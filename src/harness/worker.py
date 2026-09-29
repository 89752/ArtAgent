"""Private subprocess entry point; execution is fenced by the parent's attempt."""
import sys


def main():
    task_id, user_id, attempt = sys.argv[1:]
    from src.tasks.store import get_task
    job = get_task(task_id)
    if job and job["type"] in {"ingest_pdf","ingest_table"}:
        from src.harness.ingestion import run_ingestion
        run_ingestion(task_id,user_id,attempt=attempt)
        return
    from src.agent.graph import get_graph
    from src.harness.runner import run_job
    run_job(get_graph(), task_id, user_id, attempt=attempt)


if __name__ == "__main__":
    main()
