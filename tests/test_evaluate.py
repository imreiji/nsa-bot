"""/evaluate re-judges reviewed posts without saving; /notes feeds the judge's who's-who."""

import asyncio
import json
from types import SimpleNamespace as NS

import nsabot.bot as b
from nsabot import evaluate, scoring
from nsabot.db import DB, Message

GUILD = 10


def test_batches_stay_in_one_channel_and_a_short_window():
    db = DB(":memory:")
    db.save_batch(50, 1000, [Message(i, GUILD, 50, 5, "u", "x") for i in range(1, 1001)])
    db.save_batch(51, 5, [Message(2000 + i, GUILD, 51, 5, "u", "x") for i in range(1, 6)])
    posts = [(50, 1), (50, 30), (50, 500), (51, 2001), (50, 60)]
    batches = evaluate.group_batches(posts, db.count_between, batch_size=40)
    assert batches == [(50, [1, 30, 60]), (50, [500]), (51, [2001])]  # 500 is too far from 1
    assert evaluate.group_batches([(50, i) for i in range(1, 6)], db.count_between, 2) == [(50, [1, 2]), (50, [3, 4]), (50, [5])]


def test_metrics_compare_before_and_after():
    rows = [{"id": "1", "should": 0, "bot": 7, "safety": "self_harm"},
            {"id": "2", "should": 0, "bot": 4, "safety": None},
            {"id": "3", "should": 6, "bot": 10, "safety": None}]
    m = evaluate.metrics(rows, {1: 0, 2: 3, 3: 6})
    assert m["n"] == 3
    assert (m["before"]["false_flags"], m["after"]["false_flags"], m["before"]["clean_total"]) == (2, 1, 2)
    assert (m["before"]["safety_hits"], m["after"]["safety_hits"]) == (1, 0)
    assert (m["before"]["tens"], m["after"]["tens"]) == (1, 0)
    assert m["after"]["avg_gap"] == 1.0 and m["worst"][0] == (2, 0, 3, None)


def test_evaluation_run_saves_nothing(monkeypatch, tmp_path):
    rows = [{"id": str(i), "should": 0 if i % 2 else 6, "verdict": "ok", "safety": None, "bot": 8} for i in range(1, 11)]
    path = tmp_path / "review.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(evaluate, "REVIEW_SET", str(path))
    monkeypatch.setattr(evaluate.load_review_set, "__defaults__", (str(path),))
    b.db.conn.execute("DELETE FROM messages")
    b.db.save_batch(50, 10, [Message(i, GUILD, 50, 5, "u", "she has a boyfriend" if i % 2 == 0 else "hi") for i in range(1, 11)])
    b.db.save_verdicts([(i, 8, "old", None) for i in range(1, 11)], 2)
    sent = []

    async def create(**kwargs):
        sent.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        flagged = [{"i": m["i"], "evidence": "boyfriend", "behaviours": ["unicorn"], "target": "real",
                    "sincerity": "ambiguous", "intensity": "passing"} for m in payload["messages"]
                   if "i" in m and "boyfriend" in m["text"]]
        return NS(usage=None, choices=[NS(finish_reason="stop", message=NS(content=json.dumps({"flagged": flagged})))])

    monkeypatch.setattr(b.judge.client.chat.completions, "create", create)
    guild = NS(id=GUILD, get_channel_or_thread=lambda _: NS(name="general", topic=None, is_nsfw=lambda: False))
    report = asyncio.run(b.run_evaluation(guild))
    assert report["n"] == 10 and report["calls"] == 1
    assert report["after"]["false_flags"] == 0 and report["before"]["false_flags"] == 5
    assert report["after"]["avg_gap"] == 0.0  # unicorn 6 + real 1 - passing 1 = 6
    assert all(b.db.get_message(i)["severity"] == 8 for i in range(1, 11))  # nothing was saved
    embed = b.evaluation_embed(GUILD, report)
    assert "False flags:** 5/5 → **0**/5" in embed.description


def test_server_notes_reach_the_prompt(monkeypatch, tmp_path):
    notes = tmp_path / "server_notes.md"
    monkeypatch.setattr(b, "SERVER_NOTES_PATH", str(notes))
    b.anchor_cache.clear()
    assert "Server notes" not in b.prompt_extras(GUILD)
    notes.write_text("Members: yargas, Kan (みかんP)\nSeiyuu (adults): Hayama Fuka")
    assert "Kan (みかんP)" in b.prompt_extras(GUILD) and "trust these" in b.prompt_extras(GUILD)
