"""记忆可见可控存储测试（G2/1.2）。"""

import os
import tempfile
from pathlib import Path

import pytest

from src.memory import store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="artagent_pref_test_"))
    monkeypatch.setenv("ARTAGENT_MEMORY_DIR", str(tmp))
    # 连接是模块级单例，重置到新路径
    store._conn = None
    store._DB_DIR = tmp
    store._DB_PATH = tmp / "preferences.db"
    yield


def test_list_and_delete_preference():
    store.upsert_preference("web_user", "artist", "Van Gogh", weight=2)
    store.upsert_preference("web_user", "style", "印象派")
    store.upsert_preference("other", "artist", "Monet")

    items = store.list_preferences("web_user")
    assert len(items) == 2
    kinds = {i["kind"] for i in items}
    assert kinds == {"artist", "style"}
    by_kind = {i["kind"]: i for i in items}
    assert by_kind["artist"]["weight"] == 2
    assert "updated_at" in by_kind["artist"]

    # 跨用户隔离：other 的偏好不可见
    assert all(i["value"] != "Monet" for i in items)

    ok = store.delete_preference("web_user", "artist", "Van Gogh")
    assert ok is True
    assert store.delete_preference("web_user", "artist", "Van Gogh") is False
    assert len(store.list_preferences("web_user")) == 1


def test_delete_invalid_kind_or_value():
    assert store.delete_preference("web_user", "movement", "x") is False
    assert store.delete_preference("web_user", "artist", "") is False
