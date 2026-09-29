"""Private JSON-lines protocol; ordinary application logs go to stderr."""
import json
import sys


def main():
    request = json.loads(sys.stdin.readline())
    output = sys.stdout
    sys.stdout = sys.stderr
    try:
        if request["kind"] == "chat":
            from web.service import stream_answer
            events = stream_answer(**request["payload"])
        elif request["kind"] == "analysis":
            from web.analysis_service import stream_analysis
            events = stream_analysis(**request["payload"])
        else:
            raise ValueError("unknown stream kind")
        for event in events:
            output.write(json.dumps(event, ensure_ascii=False) + "\n")
            output.flush()
    except Exception as exc:
        output.write(json.dumps({"type": "error", "message": "执行失败：" + type(exc).__name__}) + "\n")
        output.flush()


if __name__ == "__main__":
    main()
