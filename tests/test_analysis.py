"""绘画分析与用户图片统一测试：引擎编排 / 指标 / 校验 / 存储 / API。

离线：monkeypatch 视觉调用与引擎；API 用 TestClient + 临时 SQLite 隔离。
"""

import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api
import src.analysis.engine as engine
import src.analysis.gate as gate_mod
import src.analysis.metrics as metrics_mod
import src.analysis.store as store
import src.analysis.validate as validate_mod
import web.analysis_service as analysis_service
from src.data import db
from src.data import documents_store
from src.memory import conversations as conv_mod
from src.memory import feedback as fb_mod
from src.memory import summary as summary_mod
from src.observability import runs as runs_mod
from src.tasks import store as tasks_mod
from src.platform import users as users_store

os.environ["RATE_LIMIT_RPM"] = "0"

_TMP = Path(tempfile.mkdtemp(prefix="artagent_analysis_test_"))
store.DB_PATH = _TMP / "user_images.db"
engine.USER_IMAGE_ROOT = _TMP / "uploads" / "user_images"
documents_store.DB_PATH = _TMP / "documents.db"
documents_store._LEGACY_STATUS_FILE = _TMP / "doc_status.json"
conv_mod._DB_PATH = _TMP / "conversations.db"
conv_mod._db_ready = False
summary_mod._DB_PATH = _TMP / "conversations.db"
summary_mod._db_ready = False
fb_mod._DB_PATH = _TMP / "feedback.db"
fb_mod._db_ready = False
runs_mod._DB_PATH = _TMP / "observability.db"
runs_mod._db_ready = False
tasks_mod._DB_PATH = _TMP / "tasks.db"
tasks_mod._db_ready = False
db.close_all()


@pytest.fixture(autouse=True)
def clean_tables(client):
    # 重新断言各存储路径：pytest 命令行顺序可能让其他文件的 fixture
    # 改写本文件的模块全局 _DB_PATH，导致跨文件数据串扰
    documents_store.DB_PATH = _TMP / "documents.db"
    documents_store._LEGACY_STATUS_FILE = _TMP / "doc_status.json"
    conv_mod._DB_PATH = _TMP / "conversations.db"
    conv_mod._db_ready = False
    summary_mod._DB_PATH = _TMP / "conversations.db"
    summary_mod._db_ready = False
    fb_mod._DB_PATH = _TMP / "feedback.db"
    fb_mod._db_ready = False
    runs_mod._DB_PATH = _TMP / "observability.db"
    runs_mod._db_ready = False
    tasks_mod._DB_PATH = _TMP / "tasks.db"
    tasks_mod._db_ready = False
    db.close_all()
    users_store._reset_for_tests(_TMP / "platform.db")
    users_store.ensure_default_user()
    client.headers.update({"Authorization": f"Bearer {users_store.issue_session_token('web_user')}"})
    store.DB_PATH = _TMP / "user_images.db"
    engine.USER_IMAGE_ROOT = _TMP / "uploads" / "user_images"
    store.init_db()
    documents_store.init_db()
    with store._connect() as conn:
        conn.execute("DELETE FROM painting_analysis_results")
        conn.execute("DELETE FROM user_images")
        conn.commit()
    engine.USER_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture(scope="module")
def client():
    store.init_db()
    with TestClient(api.app) as c:
        yield c


# ══════════════ 本地确定性度量 ══════════════
def _rand_img(w=300, h=400):
    return Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8"))


def test_metrics_full_shape():
    m = metrics_mod.analyze_metrics(_rand_img())
    assert set(m) == {
        "dominant_colors", "brightness_contrast", "saturation",
        "composition_grid", "hue_relationship", "value_tiers", "visual_weight",
    }
    assert len(m["dominant_colors"]) == 5
    assert m["brightness_contrast"]
    assert m["saturation"] in {"vivid", "muted", "moderate"}
    assert m["composition_grid"] in {"dynamic", "balanced"}
    assert m["hue_relationship"]["scheme"]
    assert m["value_tiers"]["label"]
    assert m["visual_weight"]["description"]


def test_complementary_colors_detected():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :50] = (255, 0, 0)
    arr[:, 50:] = (0, 255, 255)
    m = metrics_mod.analyze_metrics(Image.fromarray(arr))
    assert m["hue_relationship"]["scheme"] == "互补色"


def test_analogous_colors_detected():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :50] = (255, 40, 0)
    arr[:, 50:] = (255, 120, 0)
    m = metrics_mod.analyze_metrics(Image.fromarray(arr))
    assert m["hue_relationship"]["scheme"] == "邻近色"


# ══════════════ S4 校验与安全规则 ══════════════
def _base_report(framework="realistic"):
    return {
        "framework": framework,
        "overall_assessment": "ok",
        "layer1_technique": {
            "perspective": {
                "applies": True, "kind": "one_point", "assessment": "x",
                "confidence": 0.8, "evidence": [],
            },
            "composition": {
                "principles_applied": [], "visual_weight": "x", "whitespace": "x",
                "assessment": "x", "confidence": 0.8, "evidence": [],
            },
            "color": {
                "scheme": "x", "value_contrast": "中", "saturation_tendency": "x",
                "warm_cool": "x", "dominant_colors": [], "assessment": "x",
                "confidence": 0.8, "evidence": [],
            },
            "line_brushwork": {
                "applies": True, "line_quality": "x", "brushwork_style": "x",
                "skill_signs": [], "assessment": "x", "confidence": 0.8, "evidence": [],
            },
        },
        "layer2_style_mood": {"mood": "沉静", "mood_evidence": {"color": "x"}},
        "layer3_suggestions": {
            "priority_items": [
                {
                    "issue": "透视消失点不统一",
                    "principle": "经典一点/两点透视",
                    "action": "统一地平线上的消失点",
                    "difficulty": "beginner",
                    "location_hint": "地面砖缝",
                }
            ]
        },
    }


def test_missing_fields_detected():
    r = _base_report()
    del r["layer2_style_mood"]["mood"]
    r["layer1_technique"]["color"]["assessment"] = ""
    missing = validate_mod.missing_fields(r)
    assert "layer2_style_mood.mood" in missing
    assert "layer1_technique.color.assessment" in missing


def test_vague_suggestions_detected():
    r = _base_report()
    r["layer3_suggestions"]["priority_items"][0]["action"] = "画得自然一点就好"
    assert validate_mod.vague_suggestions(r)


def test_boundary_hits():
    assert validate_mod.boundary_hits("作者可能有焦虑症") == ["焦虑症"]
    assert validate_mod.boundary_hits("色彩明快") == []


def test_framework_consistency_fixed_for_abstract():
    r = _base_report(framework="abstract")
    fixed = validate_mod.fix_framework_consistency(r)
    assert fixed["layer1_technique"]["perspective"]["applies"] is False
    assert fixed["layer1_technique"]["perspective"]["kind"] == "not_applicable"


def test_sanitize_injects_disclaimer_and_fixes_framework():
    r = _base_report(framework="childlike")
    out = validate_mod.sanitize_report(r)
    assert out["disclaimer"]
    assert out["layer1_technique"]["perspective"]["applies"] is False
    assert r.get("disclaimer") is None


# ══════════════ 引擎编排 ══════════════
_GATE = {
    "framework": "realistic",
    "confidence": 0.9,
    "reason": "test",
    "quality_flags": [],
    "content_summary": "a painting",
}


def _report(framework="realistic"):
    return {
        "framework": framework,
        "overall_assessment": "ok",
        "layer1_technique": {
            "perspective": {
                "applies": framework == "realistic",
                "kind": "one_point" if framework == "realistic" else "not_applicable",
                "vanishing_points": [], "assessment": "x",
                "confidence": 0.8, "evidence": [],
            },
            "composition": {
                "principles_applied": [], "visual_weight": "x", "whitespace": "x",
                "assessment": "x", "confidence": 0.8, "evidence": [],
            },
            "color": {
                "scheme": "x", "value_contrast": "中", "saturation_tendency": "x",
                "warm_cool": "x", "dominant_colors": [], "assessment": "x",
                "confidence": 0.8, "evidence": [],
            },
            "line_brushwork": {
                "applies": True, "line_quality": "x", "brushwork_style": "x",
                "skill_signs": [], "assessment": "x", "confidence": 0.8, "evidence": [],
            },
        },
        "layer2_style_mood": {"mood": "沉静", "mood_evidence": {"color": "x"}},
        "layer3_suggestions": {"priority_items": []},
    }


def _make_image(image_id: str) -> Path:
    target = engine.USER_IMAGE_ROOT / image_id
    target.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((np.random.rand(300, 400, 3) * 255).astype("uint8"))
    path = target / "original.png"
    img.save(path)
    store.add_image(
        image_id, "sess-1", "p.png", str(path), 10, "image/png", 400, 300,
    )
    return path


def test_run_analysis_success(monkeypatch):
    _make_image("img-a")
    monkeypatch.setattr(engine, "classify_framework", lambda b64, ext: dict(_GATE))
    monkeypatch.setattr(
        engine, "generate_layered_report", lambda *a, **k: _report("realistic"),
    )
    events = list(engine.run_analysis("img-a"))
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["preprocess", "metrics", "gate", "report"]
    done = next(e for e in events if e["type"] == "done")
    assert done["report"]["framework"] == "realistic"
    assert (engine.USER_IMAGE_ROOT / "img-a" / "result.json").is_file()
    assert store.get_image("img-a")["status"] == "done"
    assert store.get_analysis("img-a")["framework"] == "realistic"


def test_run_analysis_framework_override(monkeypatch):
    _make_image("img-b")
    monkeypatch.setattr(
        engine, "generate_layered_report", lambda *a, **k: _report("abstract"),
    )
    events = list(engine.run_analysis("img-b", framework_override="abstract"))
    done = next(e for e in events if e["type"] == "done")
    assert done["gate"]["framework"] == "abstract"
    assert done["report"]["framework"] == "abstract"


def test_run_analysis_rejects_not_painting(monkeypatch):
    _make_image("img-c")
    gate = dict(_GATE, framework="not_painting")
    monkeypatch.setattr(engine, "classify_framework", lambda b64, ext: gate)
    events = list(engine.run_analysis("img-c"))
    rejected = next(e for e in events if e["type"] == "rejected")
    assert "摄影" in rejected["reason"]
    assert store.get_image("img-c")["status"] == "rejected"


def test_run_analysis_missing_image():
    events = list(engine.run_analysis("nope"))
    assert events == [{"type": "error", "message": "图片不存在或已删除"}]


def test_gate_parse_ok(monkeypatch):
    class FakeResp:
        content = '```json\n{"framework":"abstract","confidence":0.8,"reason":"x","quality_flags":["blurry"],"content_summary":"y"}\n```'

    class FakeLLM:
        def invoke(self, messages):
            return FakeResp()

    monkeypatch.setattr("src.utils.llm.get_vision_llm", lambda: FakeLLM())
    out = gate_mod.classify_framework("b64", "jpeg")
    assert out["framework"] == "abstract"
    assert out["confidence"] == 0.8
    assert out["quality_flags"] == ["blurry"]


def test_gate_parse_failure_falls_back_unknown(monkeypatch):
    class FakeResp:
        content = "not json at all"

    class FakeLLM:
        def invoke(self, messages):
            return FakeResp()

    monkeypatch.setattr("src.utils.llm.get_vision_llm", lambda: FakeLLM())
    out = gate_mod.classify_framework("b64", "jpeg")
    assert out["framework"] == "unknown"


# ══════════════ 图片/分析存储 ══════════════
def test_add_get_list_update():
    store.add_image(
        "img-1", "sess-1", "a.png", "/tmp/a.png", 10,
        "image/png", 100, 200, status="uploaded",
    )
    rec = store.get_image("img-1")
    assert rec["image_id"] == "img-1"
    assert rec["width"] == 100 and rec["height"] == 200
    assert [r["image_id"] for r in store.list_images_by_session("sess-1")] == ["img-1"]
    assert store.list_images_by_session("sess-2") == []
    store.update_image_status("img-1", "analyzing")
    assert store.get_image("img-1")["status"] == "analyzing"


def test_analysis_save_get_and_session_list():
    store.add_image("img-2", "sess-1", "b.png", "/tmp/b.png", 5, "image/png", 10, 10)
    store.save_analysis("img-2", "abstract", "/tmp/result.json", {"focus": "all"})
    row = store.get_analysis("img-2")
    assert row["framework"] == "abstract"
    assert row["metadata"]["focus"] == "all"
    rows = store.list_analysis_by_session("sess-1")
    assert [r["image_id"] for r in rows] == ["img-2"]
    assert store.list_analysis_by_session("sess-x") == []


def test_delete_cascades_analysis():
    store.add_image("img-3", "sess-1", "c.png", "/tmp/c.png", 5, "image/png", 10, 10)
    store.save_analysis("img-3", "realistic", "/tmp/result.json")
    assert store.delete_image("img-3") is True
    assert store.get_image("img-3") is None
    assert store.get_analysis("img-3") is None


def test_cleanup_expired_removes_only_old():
    store.add_image("old-img", "sess-1", "old.png", "/tmp/old.png", 5, "image/png", 10, 10)
    store.add_image("new-img", "sess-1", "new.png", "/tmp/new.png", 5, "image/png", 10, 10)
    with store._connect() as conn:
        conn.execute(
            "UPDATE user_images SET created_at = '2000-01-01 00:00:00' WHERE image_id = 'old-img'"
        )
        conn.commit()
    expired = store.cleanup_expired(ttl_days=30)
    assert expired == ["old-img"]
    assert store.get_image("old-img") is None
    assert store.get_image("new-img") is not None


# ══════════════ 图片上传 / 分析 API ══════════════
def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.fromarray((np.random.rand(120, 160, 3) * 255).astype("uint8")).save(
        buf, format="PNG"
    )
    return buf.getvalue()


def test_upload_rejects_non_image(client):
    r = client.post(
        "/api/user-images/upload",
        files={"file": ("evil.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert "仅支持" in r.json()["error"]


def test_upload_file_attach_analysis_delete(client, monkeypatch):
    matches = [
        r
        for r in api.app.routes
        if getattr(r, "path", "") == "/api/user-images/upload"
        and "POST" in (getattr(r, "methods", None) or set())
    ]
    assert matches, "未注册 POST /api/user-images/upload 路由"
    assert matches[0].endpoint.__name__ == "upload_user_image"
    png = _png_bytes()
    assert len(png) > 0
    r = client.post(
        "/api/user-images/upload",
        files={"file": ("paint.png", png, "image/png")},
        data={"session_id": "art-sess"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    image_id = body["image_id"]

    meta = client.get(f"/api/user-images/{image_id}")
    assert meta.status_code == 200
    assert meta.json()["image"]["session_id"] == "art-sess"
    f = client.get(f"/api/user-images/{image_id}/file")
    assert f.status_code == 200
    assert f.headers["content-type"].startswith("image/png")

    r = client.post(
        f"/api/user-images/{image_id}/attach",
        json={"session_id": "art-sess"},
    )
    assert r.status_code == 200
    msgs = client.get("/api/sessions/art-sess").json()["messages"]
    assert msgs and msgs[0]["kind"] == "image"

    def fake_run(image_id_, focus="all", framework_override=None):
        yield {"type": "stage", "stage": "preprocess", "label": "预处理"}
        yield {
            "type": "done",
            "image_id": image_id_,
            "gate": {"framework": "realistic"},
            "metrics": {},
            "report": {
                "framework": "realistic",
                "overall_assessment": "ok",
                "layer1_technique": {},
                "layer2_style_mood": {"mood": "沉静", "mood_evidence": {}},
                "layer3_suggestions": {"priority_items": []},
                "disclaimer": "x",
            },
            "focus": focus,
        }

    monkeypatch.setattr(analysis_service, "run_analysis", fake_run)
    with client.stream(
        "POST", f"/api/painting-analysis/{image_id}", params={"focus": "all"}
    ) as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())
    assert '"type": "stage"' in text
    assert '"type": "done"' in text

    r = client.delete(f"/api/user-images/{image_id}")
    assert r.status_code == 200
    assert client.get(f"/api/user-images/{image_id}").status_code == 404


def test_painting_analysis_rejected_event(client, monkeypatch):
    r = client.post(
        "/api/user-images/upload",
        files={"file": ("photo.png", _png_bytes(), "image/png")},
        data={"session_id": "art-sess2"},
    )
    image_id = r.json()["image_id"]

    def fake_reject(image_id_, focus="all", framework_override=None):
        yield {
            "type": "rejected",
            "reason": "这看起来不是一张绘画作品（可能是摄影、截图或图表）。",
            "guide": "请重试。",
        }

    monkeypatch.setattr(analysis_service, "run_analysis", fake_reject)
    with client.stream("POST", f"/api/painting-analysis/{image_id}") as r:
        text = "".join(r.iter_text())
    assert '"type": "rejected"' in text


def test_analysis_message_persisted(client):
    r = client.post(
        "/api/user-images/upload",
        files={"file": ("p.png", _png_bytes(), "image/png")},
        data={"session_id": "art-sess4"},
    )
    image_id = r.json()["image_id"]
    target = engine.USER_IMAGE_ROOT / image_id
    target.mkdir(parents=True, exist_ok=True)
    result_path = target / "result.json"
    result_path.write_text(
        json.dumps(
            {"image_id": image_id, "report": {"framework": "realistic", "overall_assessment": "整体构图稳定"}}
        ),
        encoding="utf-8",
    )
    store.save_analysis(image_id, "realistic", str(result_path))

    r = client.post(
        f"/api/painting-analysis/{image_id}/message",
        json={
            "session_id": "art-sess4",
            "user_text": "分析画作：测试画.png",
            "html": '<div class="md-answer">分析完成：整体构图稳定</div>',
            "title": "测试画.png",
        },
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    msgs = client.get("/api/sessions/art-sess4").json()["messages"]
    users = [m for m in msgs if m["role"] == "user"]
    assert users and users[0]["content"] == "分析画作：测试画.png"
    assistant = [m for m in msgs if m["role"] == "assistant"]
    assert assistant
    assert assistant[-1]["report"]["framework"] == "realistic"
    assert assistant[-1]["title"] == "测试画.png"
    assert assistant[-1]["analysis"] is True

    r2 = client.post(
        f"/api/painting-analysis/{image_id}/message",
        json={"session_id": "art-sess4", "user_text": "分析画作：测试画.png"},
    )
    assert r2.json()["duplicated"] is True
    msgs2 = client.get("/api/sessions/art-sess4").json()["messages"]
    assert len([m for m in msgs2 if m["role"] == "user"]) == 1
    assert len([m for m in msgs2 if m["role"] == "assistant"]) == 1


def test_analysis_rejection_persisted_without_result(client):
    r = client.post(
        "/api/user-images/upload",
        files={"file": ("photo.png", _png_bytes(), "image/png")},
        data={"session_id": "art-sess5"},
    )
    image_id = r.json()["image_id"]
    r = client.post(
        f"/api/painting-analysis/{image_id}/message",
        json={
            "session_id": "art-sess5",
            "user_text": "分析画作：photo.png",
            "html": '<div class="md-answer">😔 这看起来不是绘画作品</div>',
            "title": "photo.png",
        },
    )
    assert r.status_code == 200, r.text
    resp = client.get("/api/sessions/art-sess5")
    assert resp.status_code == 200, resp.text
    msgs = resp.json()["messages"]
    assistant = [m for m in msgs if m["role"] == "assistant"]
    assert assistant and "不是绘画作品" in assistant[-1]["content"]
    assert assistant[-1].get("report") is None
    assert assistant[-1]["analysis"] is True
