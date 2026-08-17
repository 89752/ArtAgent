"""绘画分析框架判定评测入口。

用法：
  python eval/painting_analysis_eval.py --dry-run          # 只打印评测集统计
  python eval/painting_analysis_eval.py --limit 5          # 在线跑前 5 例（需视觉 API）
  python eval/painting_analysis_eval.py --case realistic_01

在线模式对每个已填 image_path 的用例调用框架门控，输出判定准确率与拒绝召回。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SET_PATH = BASE / "eval" / "sets" / "painting_analysis.json"


def load_cases() -> list[dict]:
    data = json.loads(SET_PATH.read_text(encoding="utf-8"))
    return data.get("cases") or []


def dry_run(cases: list[dict]) -> None:
    by_framework: dict[str, int] = {}
    with_images = 0
    for c in cases:
        by_framework[c["framework"]] = by_framework.get(c["framework"], 0) + 1
        if c.get("image_path"):
            with_images += 1
    print(f"用例总数：{len(cases)}")
    print("框架分布：" + "，".join(f"{k}={v}" for k, v in sorted(by_framework.items())))
    print(f"已配置图片：{with_images}（在线评估需为用例填写 image_path）")


def run_online(cases: list[dict], limit: int, case_id: str | None) -> None:
    from src.analysis.gate import classify_framework
    from src.utils.http import load_image_bytes

    targets = [c for c in cases if case_id is None or c["id"] == case_id]
    if limit:
        targets = targets[:limit]
    correct = 0
    rejected = 0
    total = 0
    for c in targets:
        path = c.get("image_path") or ""
        if not path or not Path(path).is_file():
            print(f"[skip] {c['id']}: 未配置 image_path")
            continue
        try:
            data, ext = load_image_bytes(path)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {c['id']}: 读取失败 {e}")
            continue
        import base64

        b64 = base64.b64encode(data).decode("ascii")
        out = classify_framework(b64, ext)
        total += 1
        ok = out["framework"] == c["framework"]
        if ok:
            correct += 1
        if c["framework"] == "not_painting" and out["framework"] == "not_painting":
            rejected += 1
        print(
            f"[{'PASS' if ok else 'FAIL'}] {c['id']}: "
            f"期望={c['framework']} 实际={out['framework']} 置信={out['confidence']:.2f}"
        )
    if total:
        print(f"\n框架判定准确率：{correct}/{total} = {correct / total:.1%}")
        neg = sum(1 for c in targets if c["framework"] == "not_painting")
        if neg:
            print(f"拒绝召回：{rejected}/{neg} = {rejected / neg:.1%}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case", default=None)
    args = parser.parse_args()
    cases = load_cases()
    if args.dry_run:
        dry_run(cases)
        return 0
    run_online(cases, args.limit, args.case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
