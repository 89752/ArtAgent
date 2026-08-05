"""
MinerU 精准解析 API 解析器。

定位：文字路线主力解析器（版面理解、表格/公式/图注识别）。
走官方 v4 云端 API，不引入本地重依赖（~4GB 模型 + datasets/pyarrow
冲突前科，见实施方案）；与 pdfplumber_fallback 同签名，pipeline
按 MINERU_TOKEN 是否配置选择解析器，调用失败可降级 pdfplumber。

流程（v4 契约，mineru.net/doc）：
  POST /api/v4/file-urls/batch                     申请签名上传 URL
  PUT  file_urls[0]（不设 Content-Type）            上传即自动开始解析
  GET  /api/v4/extract-results/batch/{batch_id}    轮询 state → done
  下载 full_zip_url → *_content_list.json → list[Block]

决策记录：
- 整份文档上传解析，不用 files[].page_ranges：选中页的 page_idx 语义
  （原页码 vs 重排序）文档未写明，先求正确性，解析后按 page_nos
  过滤；多模态页被多解析的 quota 在 2000 页/日额度内可接受。后续
  可实测 page_idx 语义后改按页解析省额度。
- model_version="vlm"：画册/复杂版面质量优先（2026-08-01 轻量 API 实测
  全扫描画册质量优秀，生产用 vlm 后端只强不弱）。
- image/chart 块的 caption 直接用 MinerU 从文档提取的图注
  （image_caption/image_footnote），不额外调视觉模型生成；无图注的内嵌图
  不产 Block（无文字可向量化，视觉内容已被多模态整页图路线覆盖）。
- header/footer/page_number 等页面辅助块不入库（检索噪声）。
- 超时纪律（事故教训）：所有 HTTP 调用显式 timeout；轮询容忍瞬时
  网络异常直到总超时。
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.ingestion.blocks import Block
from src.utils.logging_config import get_logger

load_dotenv()

logger = get_logger("ingestion.mineru")

API_BASE = "https://mineru.net/api/v4"
MODEL_VERSION = "vlm"  # pipeline / vlm / MinerU-HTML
POLL_INTERVAL = 5  # 轮询间隔（秒）
POLL_TIMEOUT = 1800  # 云端排队+解析总上限（大文档给 30 分钟）
UPLOAD_TIMEOUT = 600  # PUT 上传上限（应用层 50MB cap，家用宽带有余量）
DOWNLOAD_TIMEOUT = 300

# vlm 后端的页面辅助块：检索噪声，丢弃
_DISCARD_TYPES = {"header", "footer", "page_number", "aside_text", "page_footnote"}


def mineru_available() -> bool:
    """可用性探测：MINERU_TOKEN 配置即视为可用（调用失败由 pipeline 降级兑现）。"""
    return bool(os.getenv("MINERU_TOKEN", "").strip())


# ------------------------------------------------------------------ #
# content_list.json → Block（纯函数，纯单测覆盖）                        #
# ------------------------------------------------------------------ #


def blocks_from_content_list(
    content_list: list[dict], page_nos: set[int] | None = None
) -> list[Block]:
    """
    MinerU content_list 条目流转语义块。

    page_nos 给定时只保留这些页（0 基，与 page_classifier/pdfplumber 一致）；
    section 由 text_level>=1 的标题块推进，作用于同页后续块（含标题自身）。
    """
    blocks: list[Block] = []
    section = ""
    for item in content_list:
        page_idx = item.get("page_idx", -1)
        if page_nos is not None and page_idx not in page_nos:
            continue
        btype = item.get("type", "")
        bbox = tuple(item.get("bbox") or ())

        if btype in _DISCARD_TYPES:
            continue

        if btype == "text":
            text = (item.get("text") or "").strip()
            if not text:
                continue
            if item.get("text_level"):  # 标题块：推进小节，自身也入库
                section = text
            blocks.append(Block("text", text, page_idx, section, bbox))

        elif btype == "list":  # vlm 扩展：列表/参考文献
            items = [
                s.strip() for s in (item.get("list_items") or []) if s and s.strip()
            ]
            if items:
                blocks.append(Block("text", "\n".join(items), page_idx, section, bbox))

        elif btype == "code":  # vlm 扩展：代码/算法块
            caps = [
                c.strip() for c in (item.get("code_caption") or []) if c and c.strip()
            ]
            body = (item.get("code_body") or "").strip()
            parts = caps + ([body] if body else [])
            if parts:
                blocks.append(Block("text", "\n".join(parts), page_idx, section, bbox))

        elif btype == "equation":
            tex = (item.get("text") or "").strip()
            if tex:
                blocks.append(Block("equation", tex, page_idx, section, bbox))

        elif btype == "table":
            parts = [
                c.strip() for c in (item.get("table_caption") or []) if c and c.strip()
            ]
            body = (item.get("table_body") or "").strip()
            if body:
                parts.append(body)
            parts += [
                f.strip() for f in (item.get("table_footnote") or []) if f and f.strip()
            ]
            if parts:
                blocks.append(Block("table", "\n".join(parts), page_idx, section, bbox))

        elif btype in ("image", "chart"):
            caps = [
                c.strip() for c in (item.get("image_caption") or []) if c and c.strip()
            ]
            foots = [
                f.strip() for f in (item.get("image_footnote") or []) if f and f.strip()
            ]
            caption = "\n".join(caps + foots)
            if caption:
                blocks.append(Block("image", caption, page_idx, section, bbox))
            # 无图注的内嵌图不产 Block（视觉内容走多模态整页图路线）

    return blocks


# ------------------------------------------------------------------ #
# v4 API 调用（全部显式超时；轮询容忍瞬时网络异常）                        #
# ------------------------------------------------------------------ #


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['MINERU_TOKEN'].strip()}"}


def _check(resp: requests.Response, what: str) -> dict:
    """HTTP 层与业务码双层校验，返回 data 字段。"""
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        raise RuntimeError(
            f"MinerU {what}失败: code={body.get('code')} msg={body.get('msg')}"
        )
    return body["data"]


def _apply_upload_url(pdf_name: str) -> tuple[str, str]:
    """申请单文件批量上传，返回 (batch_id, file_url)。URL 24h 有效。"""
    enable_ocr = os.getenv("MINERU_OCR", "1").strip().lower() not in (
        "0", "false", "no",
    )
    data = _check(
        requests.post(
            f"{API_BASE}/file-urls/batch",
            headers={**_headers(), "Content-Type": "application/json"},
            json={
                "files": [{"name": pdf_name}],
                "model_version": MODEL_VERSION,
                "enable_formula": True,
                "enable_table": True,
                "is_ocr": enable_ocr,
                "language": "ch",
            },
            timeout=30,
        ),
        "申请上传 URL ",
    )
    return data["batch_id"], data["file_urls"][0]


def _upload(file_url: str, pdf_path: str) -> None:
    """PUT 二进制上传（官方明确不设 Content-Type），上传完成自动开始解析。"""
    with open(pdf_path, "rb") as f:
        resp = requests.put(file_url, data=f, timeout=UPLOAD_TIMEOUT)
    resp.raise_for_status()


def _poll_batch(batch_id: str) -> str:
    """轮询至 state=done 返回 full_zip_url；failed 抛错；瞬时网络异常容忍至总超时。"""
    t0 = time.time()
    last_state = ""
    while True:
        try:
            data = _check(
                requests.get(
                    f"{API_BASE}/extract-results/batch/{batch_id}",
                    headers=_headers(),
                    timeout=30,
                ),
                "轮询",
            )
            item = data["extract_result"][0]
            last_state = item.get("state", "")
            if last_state == "done":
                return item["full_zip_url"]
            if last_state == "failed":
                raise RuntimeError(
                    f"MinerU 解析失败: {item.get('err_msg') or '未知原因'}"
                )
            progress = item.get("extract_progress") or {}
            logger.info(
                "[mineru] state=%s %s/%s 页，已等待 %ds",
                last_state,
                progress.get("extracted_pages", "?"),
                progress.get("total_pages", "?"),
                int(time.time() - t0),
            )
        except requests.RequestException as e:
            logger.warning("[mineru] 轮询网络异常（继续等待）: %s", e)
        if time.time() - t0 > POLL_TIMEOUT:
            raise TimeoutError(
                f"MinerU 轮询超时（{POLL_TIMEOUT}s），最后状态: {last_state}"
            )
        time.sleep(POLL_INTERVAL)


def _download_content_list(zip_url: str, work_dir: Path | None) -> list[dict]:
    """下载结果 zip，取 *_content_list.json；内嵌图落盘到 work_dir/images/ 备用。"""
    resp = requests.get(zip_url, timeout=DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    cl_names = [n for n in names if n.endswith("_content_list.json")]
    if not cl_names:
        raise RuntimeError(f"MinerU 结果 zip 中未找到 content_list.json: {names[:10]}")
    content_list = json.loads(zf.read(cl_names[0]).decode("utf-8"))

    if work_dir is not None:
        img_dir = Path(work_dir) / "images"
        n_saved = 0
        for n in names:
            if "/images/" in n and not n.endswith("/"):
                img_dir.mkdir(parents=True, exist_ok=True)
                (img_dir / Path(n).name).write_bytes(zf.read(n))
                n_saved += 1
        if n_saved:
            logger.info("[mineru] 内嵌图落盘 %d 张 → %s", n_saved, img_dir)
    return content_list


def parse_pages(pdf_path: str, page_nos: list[int], *, work_dir=None) -> list[Block]:
    """
    MinerU 精准解析（与 pdfplumber_fallback.parse_pages 同签名 + work_dir 可选参）。

    整份文档上传云端解析，返回时按 page_nos（0 基）过滤块；
    work_dir 给定时把 MinerU 抠出的内嵌图落盘到其 images/ 子目录。
    """
    if not mineru_available():
        raise RuntimeError("MINERU_TOKEN 未配置，MinerU 解析器不可用")
    t0 = time.time()
    batch_id, file_url = _apply_upload_url(Path(pdf_path).name)
    logger.info("[mineru] batch_id=%s，上传 %s", batch_id, pdf_path)
    _upload(file_url, pdf_path)
    zip_url = _poll_batch(batch_id)
    content_list = _download_content_list(zip_url, work_dir)
    blocks = blocks_from_content_list(content_list, set(page_nos))
    logger.info(
        "[mineru] %s pages=%d → blocks=%d（%.1fs）",
        pdf_path,
        len(page_nos),
        len(blocks),
        time.time() - t0,
    )
    return blocks
