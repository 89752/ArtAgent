"""store.py 偏好库目录迁移（data/data/memory → data/memory）单测。"""

import sqlite3
from pathlib import Path


def _make_prefs_db(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE preferences (
            user_id    TEXT NOT NULL,
            kind       TEXT NOT NULL,
            value      TEXT NOT NULL,
            weight     REAL NOT NULL DEFAULT 1.0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, kind, value)
        )
        """
    )
    for i in range(rows):
        conn.execute(
            "INSERT INTO preferences VALUES (?, 'artist', ?, 1.0, 't')",
            ("u1", f"artist{i}"),
        )
    conn.commit()
    conn.close()


def test_migrate_legacy_when_new_is_empty(tmp_path, monkeypatch):
    import src.memory.store as store

    new_db = tmp_path / "memory" / "preferences.db"
    legacy = tmp_path / "legacy" / "preferences.db"
    _make_prefs_db(legacy, 3)
    _make_prefs_db(new_db, 0)   # 新路径是空壳 → 用旧库覆盖

    monkeypatch.setattr(store, "_DB_DIR", new_db.parent)
    monkeypatch.setattr(store, "_DB_PATH", new_db)
    monkeypatch.setattr(store, "_legacy_preferences_path", lambda: legacy)

    out = store._migrate_legacy_preferences()
    assert out == new_db
    assert new_db.exists()
    assert store._pref_rows(new_db) == 3
    assert not legacy.exists()


def test_keep_new_db_when_it_has_data(tmp_path, monkeypatch):
    import src.memory.store as store

    new_db = tmp_path / "memory" / "preferences.db"
    legacy = tmp_path / "legacy" / "preferences.db"
    _make_prefs_db(legacy, 3)
    _make_prefs_db(new_db, 2)   # 新库已有数据 → 不覆盖

    monkeypatch.setattr(store, "_DB_DIR", new_db.parent)
    monkeypatch.setattr(store, "_DB_PATH", new_db)
    monkeypatch.setattr(store, "_legacy_preferences_path", lambda: legacy)

    out = store._migrate_legacy_preferences()
    assert out == new_db
    assert store._pref_rows(new_db) == 2
    assert legacy.exists()
