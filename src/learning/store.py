"""Immutable experience revisions with an evaluated promotion boundary."""
import hashlib
import json
import uuid
from pathlib import Path
from src.harness.storage import connection, now, list_artifacts


def suite_hash():
    path = Path(__file__).resolve().parents[2] / "eval" / "sets" / "learning_holdout.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def init(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS experiences (
        id TEXT PRIMARY KEY,user_id TEXT NOT NULL,source_task TEXT NOT NULL,
        title TEXT NOT NULL,instructions TEXT NOT NULL,content_hash TEXT NOT NULL,
        triggers TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'candidate',
        evaluation TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL,
        UNIQUE(user_id,content_hash))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS learning_events (
        id INTEGER PRIMARY KEY,experience_id TEXT NOT NULL,user_id TEXT NOT NULL,
        action TEXT NOT NULL,created_at TEXT NOT NULL)""")


def decode(row):
    item = dict(row)
    item["triggers"] = json.loads(item["triggers"])
    item["evaluation"] = json.loads(item["evaluation"])
    return item


def list_experiences(user_id):
    with connection() as conn:
        init(conn)
        return [decode(r) for r in conn.execute("SELECT * FROM experiences WHERE user_id=? ORDER BY created_at DESC LIMIT 200", (user_id,))]


def create_candidate(user_id, task_id, title, instructions, triggers):
    from src.tasks.store import get_task
    task = get_task(task_id)
    if not task or task["payload"].get("user_id") != user_id or task["status"] != "done":
        raise ValueError("只能从本人已通过验收的任务提炼经验")
    if not any(a["metadata"].get("verification", {}).get("passed") for a in list_artifacts(task_id,user_id)):
        raise ValueError("缺少通过验收的来源产物")
    if not instructions.strip() or len(instructions) > 6000 or not triggers:
        raise ValueError("需要有效步骤与适用条件")
    digest = hashlib.sha256(instructions.encode()).hexdigest()
    eid = "e_" + uuid.uuid4().hex
    with connection() as conn:
        init(conn)
        conn.execute("INSERT OR IGNORE INTO experiences VALUES (?,?,?,?,?,?,?,'candidate','{}',?)",
            (eid,user_id,task_id,title[:120],instructions,digest,json.dumps(triggers[:10],ensure_ascii=False),now()))
        return decode(conn.execute("SELECT * FROM experiences WHERE user_id=? AND content_hash=?",(user_id,digest)).fetchone())


def active_guidance(user_id, query):
    return [{"id":e["id"],"content_hash":e["content_hash"],"instructions":e["instructions"]}
            for e in list_experiences(user_id) if e["status"] == "active" and
            any(str(t).lower() in query.lower() for t in e["triggers"])][:3]


def save_evaluation(eid, user_id, evaluation):
    """Only server-side evaluators call this; no HTTP endpoint accepts scores."""
    with connection() as conn:
        init(conn)
        conn.execute("UPDATE experiences SET evaluation=?,status='evaluated' WHERE id=? AND user_id=? AND status IN ('candidate','evaluated')",
                     (json.dumps(evaluation,ensure_ascii=False),eid,user_id))


def change_status(eid,user_id,action):
    with connection() as conn:
        init(conn)
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM experiences WHERE id=? AND user_id=?",(eid,user_id)).fetchone()
        if not row:
            raise ValueError("经验不存在")
        item = decode(row)
        if action == "promote":
            evaluation = item["evaluation"]
            if (item["status"] != "evaluated" or not evaluation.get("passed")
                    or evaluation.get("content_hash") != item["content_hash"]
                    or evaluation.get("suite_hash") != suite_hash()):
                raise ValueError("必须先通过当前版本的独立评测")
            status = "active"
        elif action == "rollback" and item["status"] == "active":
            status = "rolled_back"
        else:
            raise ValueError("当前状态不能执行该操作")
        conn.execute("UPDATE experiences SET status=? WHERE id=?",(status,eid))
        conn.execute("INSERT INTO learning_events(experience_id,user_id,action,created_at) VALUES (?,?,?,?)",(eid,user_id,action,now()))
    return status


def delete_user(user_id):
    with connection() as conn:
        init(conn)
        for table in ("experiences","learning_events"):
            conn.execute(f"DELETE FROM {table} WHERE user_id=?",(user_id,))
