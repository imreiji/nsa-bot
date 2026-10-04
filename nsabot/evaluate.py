"""Offline evaluation: re-judge a reviewed set of posts with the current prompt and compare.

The review set (eval/review_set.jsonl) holds only message IDs and scores, no chat text; the posts
and their context come from the bot's own database, so nothing private lives in the repo.
"""

import json
import os
from collections import Counter

REVIEW_SET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "review_set.jsonl")
MAX_WINDOW = 120  # messages spanned by one batch; flagged posts further apart go in separate calls


def load_review_set(path: str = REVIEW_SET) -> list[dict]:
    """Rows: {"id", "should", "verdict", "safety", "bot"} (bot = the score the reviewed rubric gave)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_batches(posts: list[tuple[int, int]], count_between, batch_size: int,
                  max_window: int | None = MAX_WINDOW) -> list[tuple[int, list[int]]]:
    """posts: (channel_id, message_id). Batches stay in one channel, keep at most batch_size posts and
    span at most max_window messages (None: no limit), so each call reads a real conversation and
    not a whole channel."""
    batches: list[tuple[int, list[int]]] = []
    for channel_id, mid in sorted(posts):
        if batches:
            ch, ids = batches[-1]
            if (ch == channel_id and len(ids) < batch_size
                    and (max_window is None or count_between(channel_id, ids[0], mid) <= max_window)):
                ids.append(mid)
                continue
        batches.append((channel_id, [mid]))
    return batches


def metrics(rows: list[dict], new: dict[int, int]) -> dict:
    """rows: review set entries present in the DB; new: {message_id: score from this run}."""
    rows = [r for r in rows if int(r["id"]) in new]
    n = len(rows)
    if not n:
        return {"n": 0}

    def side(get):
        gaps = [get(r) - r["should"] for r in rows]
        safety = [r for r in rows if r.get("safety")]
        clean = [r for r in rows if r["should"] == 0]
        return {
            "avg_gap": sum(map(abs, gaps)) / n,
            "bias": sum(gaps) / n,
            "within1": sum(abs(g) <= 1 for g in gaps) / n,
            "false_flags": sum(get(r) > 0 for r in clean),
            "clean_total": len(clean),
            "safety_hits": sum(get(r) > 0 for r in safety),
            "safety_total": len(safety),
            "tens": sum(get(r) == 10 for r in rows),
            "dist": [Counter(get(r) for r in rows).get(s, 0) for s in range(11)],
        }

    before = side(lambda r: r["bot"])
    after = side(lambda r: new[int(r["id"])])
    worst = sorted(rows, key=lambda r: -abs(new[int(r["id"])] - r["should"]))[:6]
    return {"n": n, "before": before, "after": after,
            "worst": [(int(r["id"]), r["should"], new[int(r["id"])], r.get("safety")) for r in worst]}
