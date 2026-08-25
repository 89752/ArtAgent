"""词法检索通道：core 英文 FTS5 + 用户 PDF BM25（按语言分组）。

设计：
- 查询语言检测（zh/en/其他），只在查询语言与目标索引语言不一致时
  用 LLM 按需翻译（LEXICAL_TRANSLATE=0 可关闭，失败回退原文）；
- core：英文 FTS5（SQLite 内置，无需扩展），索引落在
  data/index/lexical.db，按 CSV 修改时间增量重建；
- 用户 PDF：内存 BM25，中文用字符二元组、英文用词 token，按 chunk
  语言分组后用同语言 query 分别打分；
- 语义通道（BGE-M3）保持原文检索，本模块只做词法补充，结果由
  HybridRetriever 统一 RRF 融合。
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.retrieval.base import RetrievalResult

LEXICAL_ENABLED = os.getenv("LEXICAL_ENABLED", "1").strip().lower() not in (
    "0", "false", "no",
)
LEXICAL_TRANSLATE = os.getenv("LEXICAL_TRANSLATE", "1").strip().lower() not in (
    "0", "false", "no",
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EN_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and",
    "or", "with", "by", "from", "is", "are",
}

_LANG_NAMES = {"zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文", "other": "英文"}


def detect_lang(text: str) -> str:
    """轻量查询/文档语言检测：zh / en / ja / ko / other。"""
    if _KANA_RE.search(text):
        return "ja"
    if _HANGUL_RE.search(text):
        return "ko"
    if _CJK_RE.search(text):
        return "zh"
    if _CYRILLIC_RE.search(text):
        return "other"
    return "en"


@lru_cache(maxsize=256)
def translate_query(query: str, target_lang: str) -> str:
    """把查询翻译成目标语言；关闭/失败/同语言时原样返回。"""
    if not LEXICAL_TRANSLATE or detect_lang(query) == target_lang:
        return query
    try:
        from src.utils.llm import get_deterministic_llm

        prompt = (
            f"把下面的用户检索查询翻译成{_LANG_NAMES.get(target_lang, target_lang)}。"
            "只输出译文本身；人名、作品名、专有名词保留原文。\n"
            f"查询：{query}"
        )
        out = str(get_deterministic_llm().invoke(prompt).content).strip()
        return out or query
    except Exception:  # noqa: BLE001 —— 翻译失败回退原文，不阻塞检索
        return query


def _tokenize(text: str) -> list[str]:
    """混合分词：英文按词、中文按字符二元组（零依赖）。"""
    text = (text or "").lower()
    toks = [t for t in _LATIN_TOKEN_RE.findall(text) if t not in _EN_STOPWORDS]
    chars = [c for c in text if _CJK_CHAR_RE.match(c)]
    toks += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return toks


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_CHAR_RE.findall(text)) / len(text)


def _bm25_scores(
    query_tokens: list[str],
    doc_tokens: list[list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """纯 Python BM25 打分（零依赖）。"""
    n = len(doc_tokens)
    if not n or not query_tokens:
        return [0.0] * n
    doc_lens = [len(toks) for toks in doc_tokens]
    avgdl = sum(doc_lens) / n
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    idf = {
        t: math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
        for t in query_tokens
    }
    scores: list[float] = []
    for i, toks in enumerate(doc_tokens):
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if f:
                denom = f + k1 * (1 - b + b * doc_lens[i] / avgdl)
                score += idf[t] * f * (k1 + 1) / denom
        scores.append(score)
    return scores


# ------------------------------------------------------------------ #
# core 英文 FTS5                                                        #
# ------------------------------------------------------------------ #

_CORE_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS core_lexical USING fts5(
    artwork_id UNINDEXED,
    title,
    artist,
    year_display UNINDEXED,
    technique,
    school,
    movement,
    timeframe,
    image_url UNINDEXED,
    dedup_key UNINDEXED,
    description
);
CREATE TABLE IF NOT EXISTS lexical_meta(key TEXT PRIMARY KEY, value TEXT);
"""


class CoreLexicalRetriever:
    """core 英文词法检索器（source="core"，参与 RRF 融合）。"""

    source = "core"
    dataset_id = "core"

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        self.csv_path = csv_path or Path(
            os.getenv("CORE_DATA_PATH", "./data/core/artworks_core.csv")
        )
        self.db_path = db_path or (
            Path(os.getenv("INDEX_DIR", "./data/index")) / "lexical.db"
        )

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_index(self) -> bool:
        if not self.csv_path.exists():
            return False
        conn = self._connect()
        try:
            conn.executescript(_CORE_FTS_SCHEMA)
            mtime = self.csv_path.stat().st_mtime
            row = conn.execute(
                "SELECT value FROM lexical_meta WHERE key='core_mtime'"
            ).fetchone()
            count = conn.execute("SELECT count(*) FROM core_lexical").fetchone()[0]
            if row and float(row[0]) >= mtime and count:
                return True
            self._rebuild(conn, mtime)
            return True
        finally:
            conn.close()

    def _rebuild(self, conn: sqlite3.Connection, mtime: float) -> None:
        import pandas as pd

        df = pd.read_csv(self.csv_path, encoding="utf-8-sig", keep_default_na=False)
        if "year_display" not in df.columns:
            inc = (
                df["inception"].astype(str).str.strip()
                if "inception" in df.columns
                else ""
            )
            yr = df["year"].astype(str).str.strip() if "year" in df.columns else ""
            df["year_display"] = inc.where(inc != "", yr)

        conn.execute("DELETE FROM core_lexical")
        rows = [
            (
                str(r.artwork_id), str(r.title), str(r.artist_name),
                str(r.year_display), str(r.material), str(r.school),
                str(r.movement), str(r.year_bucket), str(r.image_url),
                str(r.dedup_key), str(r.description),
            )
            for _, r in df.iterrows()
        ]
        conn.executemany(
            "INSERT INTO core_lexical VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        conn.execute(
            "INSERT OR REPLACE INTO lexical_meta(key, value) VALUES('core_mtime', ?)",
            (str(mtime),),
        )
        conn.commit()

    def search(
        self, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[RetrievalResult]:
        if not self._ensure_index():
            return []
        q = translate_query(query, "en")
        tokens = _tokenize(q)
        if not tokens:
            return []
        quoted = [f'"{t}"' for t in tokens if t]
        if not quoted:
            return []
        conn = self._connect()
        try:
            rows = self._match(conn, " AND ".join(quoted), top_k * 2, filters)
            if not rows and len(quoted) > 1:
                rows = self._match(conn, " OR ".join(quoted), top_k * 2, filters)
        finally:
            conn.close()
        return rows[:top_k]

    def _match(
        self,
        conn: sqlite3.Connection,
        match_expr: str,
        limit: int,
        filters: dict | None,
    ) -> list[RetrievalResult]:
        sql = (
            "SELECT artwork_id, title, artist, year_display, technique, school,"
            " movement, timeframe, image_url, dedup_key, description,"
            " bm25(core_lexical) AS b"
            " FROM core_lexical WHERE core_lexical MATCH ?"
            " ORDER BY b DESC LIMIT ?"
        )
        out: list[RetrievalResult] = []
        for row in conn.execute(sql, (match_expr, limit)):
            meta = {
                "artwork_id": row[0],
                "title": row[1],
                "artist": row[2],
                "year_display": row[3],
                "material": row[4],
                "school": row[5],
                "movement": row[6],
                "year_bucket": row[7],
                "image_url": row[8],
                "dedup_key": row[9],
                "description": row[10],
            }
            if filters and not _core_hit_filters(meta, filters):
                continue
            out.append(
                RetrievalResult(
                    content=row[10] or row[1],
                    source="core",
                    score=float(row[11]),
                    metadata=meta,
                    image_refs=[row[8]] if row[8] else [],
                )
            )
        return out


def _core_hit_filters(meta: dict, filters: dict) -> bool:
    for key in ("author", "artist", "school", "timeframe"):
        value = filters.get(key)
        if not value:
            continue
        field = str(meta.get("artist") if key in ("author", "artist") else meta.get(key) or "")
        if str(value).lower() not in field.lower():
            return False
    return True


# ------------------------------------------------------------------ #
# 用户 PDF BM25（按 chunk 语言分组）                                      #
# ------------------------------------------------------------------ #


class PdfBm25Retriever:
    """用户 PDF 文字 chunk 的词法检索器（source="user_pdf_text"）。"""

    source = "user_pdf_text"
    dataset_id = None

    def __init__(self):
        self._cache: Optional[dict] = None

    def _load_chunks(self) -> dict[str, list[tuple[str, dict]]]:
        from src.retrieval.hybrid import get_or_create_chroma_collection
        from src.retrieval.userdoc_text_retriever import (
            COLLECTION_NAME,
            FALLBACK_COLLECTION_NAME,
        )

        collections = [
            get_or_create_chroma_collection(COLLECTION_NAME),
            get_or_create_chroma_collection(FALLBACK_COLLECTION_NAME),
        ]
        count = sum(collection.count() for collection in collections)
        if self._cache and self._cache.get("count") == count and count:
            return self._cache["groups"]
        groups: dict[str, list[tuple[str, dict]]] = {"zh": [], "en": []}
        for collection in collections:
            if not collection.count():
                continue
            data = collection.get(include=["documents", "metadatas"])
            for doc, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
                lang = "zh" if _cjk_ratio(doc or "") > 0.2 else "en"
                groups[lang].append((doc or "", dict(meta or {})))
        self._cache = {"count": count, "groups": groups}
        return groups

    def search(
        self, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[RetrievalResult]:
        groups = self._load_chunks()
        out: list[RetrievalResult] = []
        for lang, items in groups.items():
            if not items:
                continue
            q = translate_query(query, lang)
            qt = _tokenize(q)
            if not qt:
                continue
            docs = [d for d, _ in items]
            scores = _bm25_scores(qt, [_tokenize(d) for d in docs])
            ranked = sorted(range(len(docs)), key=lambda i: -scores[i])
            for i in ranked[: top_k * 2]:
                if scores[i] <= 0:
                    continue
                content, meta = items[i]
                if filters and not _pdf_hit_filters(meta, filters):
                    continue
                out.append(
                    RetrievalResult(
                        content=content,
                        source="user_pdf_text",
                        score=scores[i],
                        metadata=meta,
                    )
                )
        out.sort(key=lambda r: -r.score)
        return out[:top_k]


def _pdf_hit_filters(meta: dict, filters: dict) -> bool:
    for key in ("doc_id", "kb_id"):
        value = filters.get(key)
        if value and str(meta.get(key) or "") != str(value):
            return False
    return True
