"""简单多用户模式单测：会话/记忆按 user_id 隔离，ContextVar 生效。"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import src.memory.conversations as conv
import src.memory.memory_items as mi
from src.analysis import store as an
from src.data import documents_store as docs
from src.ingestion import table_pipeline as tp
from src.memory import feedback as fb
from src.platform import users as users_store


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="multi_user_"))
    mi._reset_for_tests(tmp / "agent_memory.db")
    docs._reset_for_tests(tmp / "documents.db")
    docs.init_db()
    fb._reset_for_tests(tmp / "feedback.db")
    users_store._reset_for_tests(tmp / "platform.db")
    an.DB_PATH = tmp / "user_images.db"
    an.init_db()
    conv._DB_PATH = tmp / "conversations.db"
    conv._conn = None
    monkeypatch.delenv("MEMORY_USER_ID", raising=False)
    with patch("src.memory.memory_items._embed", return_value=None):
        try:
            yield
        finally:
            mi.clear_active_user_id()


def test_conversations_isolated_by_user():
    conv.save_conversation(
        "s1", "A的对话", [{"role": "user", "content": "你好"}], user_id="u_a"
    )
    conv.save_conversation(
        "s2", "B的对话", [{"role": "user", "content": "你好"}], user_id="u_b"
    )
    items_a, total_a = conv.list_conversations(user_id="u_a")
    assert total_a == 1
    assert items_a[0]["session_id"] == "s1"
    assert conv.list_conversations(user_id="u_b")[1] == 1
    assert conv.load_conversation("s1", "u_b") == []
    assert conv.load_conversation("s1", "u_a") != []


def test_memory_items_isolated_by_user():
    from web import service

    mi.add_memory("u_a", "用户A喜欢莫奈", entity="莫奈", importance=0.8)
    mi.add_memory("u_b", "用户B喜欢梵高", entity="梵高", importance=0.8)
    a_contents = [i["content"] for i in service.memory_items_list("u_a")]
    b_contents = [i["content"] for i in service.memory_items_list("u_b")]
    assert any("莫奈" in c for c in a_contents)
    assert not any("莫奈" in c for c in b_contents)
    assert any("梵高" in c for c in b_contents)


def test_active_user_contextvar():
    mi.set_active_user_id("u_x")
    assert mi.get_memory_user_id() == "u_x"
    mi.set_active_user_id("u_y")
    assert mi.get_memory_user_id() == "u_y"


def test_default_account_login_and_token():
    users_store.ensure_default_account()
    user = users_store.verify_login("user", "11111111")
    assert user is not None
    assert user["user_id"] == "user"
    assert users_store.verify_login("user", "wrong") is None
    token = users_store.issue_session_token(user["user_id"])
    assert users_store.get_user_by_api_key(token)["user_id"] == "user"
    assert users_store.revoke_api_key(token)
    assert users_store.get_user_by_api_key(token) is None


def test_documents_isolated_by_user():
    docs.add_document(
        "doc_a", "pdf", user_id="u_a", doc_name="A.pdf", status="done"
    )
    docs.add_document(
        "doc_b", "pdf", user_id="u_b", doc_name="B.pdf", status="done"
    )
    assert docs.get_document("doc_a", "u_a") is not None
    assert docs.get_document("doc_a", "u_b") is None
    assert [d["doc_id"] for d in docs.list_documents("u_a")] == ["doc_a"]
    assert [d["doc_id"] for d in docs.list_documents("u_b")] == ["doc_b"]
    assert docs.delete_document("doc_a", "u_b") is False
    assert docs.get_document("doc_a", "u_a") is not None


def test_feedback_isolated_by_user():
    fb.add_feedback("s1", 1, user_id="u_a")
    fb.add_feedback("s2", -1, user_id="u_b")
    items_a, total_a = fb.list_feedback(user_id="u_a")
    assert total_a == 1 and items_a[0]["session_id"] == "s1"
    assert fb.list_feedback(user_id="u_b")[1] == 1


def test_image_isolated_by_user():
    an.add_image(
        "img_a", "s1", "a.png", "/tmp/a.png", 1, "image/png", 10, 10,
        user_id="u_a",
    )
    an.add_image(
        "img_b", "s1", "b.png", "/tmp/b.png", 1, "image/png", 10, 10,
        user_id="u_b",
    )
    assert an.get_image("img_a", "u_a") is not None
    assert an.get_image("img_a", "u_b") is None
    assert len(an.list_images_by_session("s1", "u_a")) == 1
    assert len(an.list_images_by_session("s1", "u_b")) == 1
    assert an.delete_image("img_a", "u_b") is False
    assert an.get_image("img_a", "u_a") is not None
    assert an.delete_image("img_a", "u_a") is True


def test_table_dataset_id_user_scoped():
    assert tp.table_dataset_id("abc", "u_x") == "table_u_x_abc"
    assert tp.table_dataset_id("abc") == "table_abc"


def test_default_account_is_admin_and_reset_password():
    users_store.ensure_default_account()
    user = users_store.get_user_by_username("user")
    assert user is not None and user["is_admin"] == 1
    assert users_store.reset_password(user["user_id"], "newpass123")
    assert users_store.verify_login("user", "newpass123") is not None
    assert users_store.verify_login("user", "11111111") is None


def test_register_user_flow():
    result = users_store.register_user("alice_art", "password123", "Alice")
    user = result["user"]
    assert user["username"] == "alice_art"
    assert user["is_admin"] == 0
    assert "password_hash" not in users_store.public_user(user)
    assert users_store.verify_login("alice_art", "password123") is not None
    assert users_store.verify_login("alice_art", "wrong") is None


def test_register_validation():
    with pytest.raises(ValueError):
        users_store.register_user("ab", "password123")          # 用户名过短
    with pytest.raises(ValueError):
        users_store.register_user("alice!", "password123")      # 非法字符
    with pytest.raises(ValueError):
        users_store.register_user("alice_art", "short")         # 密码过短
    with pytest.raises(ValueError):
        users_store.register_user("user", "password123")        # 保留用户名
    users_store.register_user("alice_art", "password123")
    with pytest.raises(KeyError):
        users_store.register_user("alice_art", "password123")   # 重复用户名


def test_change_password_flow():
    users_store.register_user("bob_art", "password123")
    user = users_store.get_user_by_username("bob_art")
    with pytest.raises(ValueError):
        users_store.change_password(user["user_id"], "wrong-old", "newpass456")
    token_keep = users_store.issue_session_token(user["user_id"])
    token_other = users_store.issue_session_token(user["user_id"])
    assert users_store.change_password(
        user["user_id"], "password123", "newpass456", keep_token=token_keep
    )
    assert users_store.verify_login("bob_art", "password123") is None
    assert users_store.verify_login("bob_art", "newpass456") is not None
    assert users_store.get_user_by_api_key(token_keep) is not None
    assert users_store.get_user_by_api_key(token_other) is None


# ── 反馈存储与导出 ──────────────────────────────────────────
def test_add_and_list_feedback():
    fb.add_feedback("s1", 1, reason="", comment="很棒")
    fb.add_feedback("s1", -1, reason="引用不充分", comment="")
    items, total = fb.list_feedback()
    assert total == 2
    assert items[0]["rating"] == -1
    assert items[0]["reason"] == "引用不充分"
    assert items[1]["rating"] == 1
    assert items[1]["comment"] == "很棒"
    assert fb.count_feedback(1) == 1
    assert fb.count_feedback(-1) == 1


def test_invalid_rating_rejected():
    with pytest.raises(ValueError):
        fb.add_feedback("s1", 0)


def test_export_feedback_jsonl(tmp_path):
    fb.add_feedback("s1", 1)
    fb.add_feedback("s2", -1, reason="不准确")
    out = tmp_path / "feedback.jsonl"
    n = fb.export_feedback(out)
    assert n == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"rating": -1' in lines[0]
