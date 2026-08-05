"""store.py 偏好兼容层单测：读写统一落在 memory_items。"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import src.memory.memory_items as mi
from src.memory import store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="artagent_pref_test_")) / "agent_memory.db"
    mi._reset_for_tests(tmp)
    monkeypatch.setenv("MEMORY_USER_ID", "test-user")
    with patch("src.memory.memory_items._embed", return_value=None):
        yield


def test_list_and_delete_preference():
    store.upsert_preference("web_user", "artist", "Van Gogh", weight=2)
    store.upsert_preference("web_user", "style", "印象派")
    store.upsert_preference("other", "artist", "Monet")

    items = store.list_preferences("web_user")
    assert len(items) == 2
    assert {i["kind"] for i in items} == {"preference"}
    by_value = {i["value"]: i for i in items}
    assert by_value["Van Gogh"]["weight"] == 1.0  # importance 折算封顶 1.0
    assert "updated_at" in by_value["Van Gogh"]

    # 跨用户隔离：other 的偏好不可见
    assert all(i["value"] != "Monet" for i in items)

    ok = store.delete_preference("web_user", "artist", "Van Gogh")
    assert ok is True
    assert store.delete_preference("web_user", "artist", "Van Gogh") is False
    assert len(store.list_preferences("web_user")) == 1


def test_load_preferences_shape():
    store.upsert_preference("web_user", "artist", "莫奈")
    assert store.load_preferences("web_user") == {
        "artists": ["莫奈"], "styles": [],
    }


def test_delete_invalid_kind_or_value():
    assert store.delete_preference("web_user", "movement", "x") is False
    assert store.delete_preference("web_user", "artist", "") is False


def test_clear_preferences():
    store.upsert_preference("web_user", "artist", "莫奈")
    store.upsert_preference("web_user", "style", "印象派")
    assert store.clear_preferences("web_user") == 2
    assert store.list_preferences("web_user") == []
