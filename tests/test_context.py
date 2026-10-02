"""The judge sees each post in context: channel, replies, attachments, surrounding conversation."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace as NS

import nsabot.bot as b
from nsabot.db import DB, Message
from nsabot.judge import build_payload, message_time

GUILD = 10


def row(i, channel=50, author=5, text="hello", scored=True, **kw):
    return Message(i, GUILD, channel, author, f"user{author}", text, scored, **kw)


def fake_message(text="", author_id=5, bot=False, attachments=(), stickers=(), embeds=(), reference=None):
    return NS(content=text, clean_content=text, author=NS(id=author_id, bot=bot, display_name="u"),
              attachments=list(attachments), stickers=list(stickers), embeds=list(embeds), reference=reference)


def test_classify_keeps_reactions_and_images_as_context():
    assert b.classify(fake_message("my oshi is my wife"), set()) is True
    assert b.classify(fake_message("w"), set()) is False
    assert b.classify(fake_message("", attachments=[NS(content_type="image/png", filename="shrine.png")]), set()) is False
    assert b.classify(fake_message(""), set()) is None
    assert b.classify(fake_message("!kimoiboard"), set()) is None
    assert b.classify(fake_message("uooh", bot=True), set()) is None
    assert b.classify(fake_message("uooh", author_id=9), {9}) is None


def test_build_payload_marks_only_scored_messages():
    db = DB(":memory:")
    db.save_batch(50, 4, [
        row(1, text="earlier chat"),
        row(2, text="my wife is in this pic", extras="image/png: bed.png"),
        row(3, text="real", reply_to_id=2),
        row(4, text="lol", scored=False),
    ])
    payload, order = build_payload({"name": "#ll-general"}, db.timeline(50, 2, 4, before=5), [2, 3])

    assert order == [2, 3]
    msgs = payload["messages"]
    assert payload["channel"] == {"name": "#ll-general"}
    assert [m.get("i") for m in msgs] == [None, 0, 1, None]
    assert msgs[1]["attachments"] == "image/png: bed.png"
    assert msgs[2]["reply_to"] == {"author": "user5", "text": "my wife is in this pic"}  # filled from the DB
    assert msgs[0]["time"] == message_time(1)


def test_timeline_limits_earlier_context():
    db = DB(":memory:")
    db.save_batch(50, 30, [row(i) for i in range(1, 31)] + [row(99, channel=51)])
    ids = [r["id"] for r in db.timeline(50, 20, 22, before=5)]
    assert ids == [15, 16, 17, 18, 19, 20, 21, 22]


def test_optout_scrubs_quoted_replies():
    db = DB(":memory:")
    db.save_batch(50, 2, [row(1, author=9, text="secret"),
                          row(2, text="lol", reply_to_id=1, reply_author_id=9, reply_author="user9", reply_text="secret")])
    db.opt_out(GUILD, 9)
    (r,) = db.timeline(50, 2, 2, before=0)
    assert r["reply_author"] is None and r["reply_text"] is None


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, channel_id INTEGER NOT NULL,"
                " author_id INTEGER NOT NULL, author_name TEXT NOT NULL, content TEXT NOT NULL, severity INTEGER, reason TEXT)")
    con.execute("INSERT INTO messages VALUES (1, 10, 50, 5, 'u', 'old post', 3, 'mild')")
    con.commit()
    con.close()
    db = DB(str(path))
    assert db.leaderboard(GUILD)[0].hits == 1
    db.save_message(row(2, reply_to_id=1))
    assert [r["reply_text"] for r in db.timeline(50, 2, 2, before=0)] == ["old post"]


def test_judge_backlog_batches_per_channel_with_context():
    calls = []

    async def fake_create(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        calls.append(payload)
        scored = [m for m in payload["messages"] if "i" in m]
        flagged = [{"i": m["i"], "severity": 7, "reason": "yes"} for m in scored if "waifu" in m["text"]]
        return NS(usage=NS(total_tokens=1), choices=[NS(finish_reason="stop", message=NS(content=json.dumps({"flagged": flagged})))])

    b.judge.client.chat.completions.create = fake_create
    b.db.conn.execute("DELETE FROM messages")
    b.db.save_batch(50, 3, [row(1, text="hi all"), row(2, text="w", scored=False), row(3, text="my waifu tho")])
    b.db.save_batch(51, 4, [row(4, channel=51, text="anyone watching tonight")])

    channels = {50: NS(name="ll-general", topic="Love Live chat", is_nsfw=lambda: False),
                51: NS(name="imas", topic=None, is_nsfw=lambda: False)}
    guild = NS(id=GUILD, get_channel_or_thread=channels.get)
    judged, flagged = asyncio.run(b.judge_backlog(guild))

    assert (judged, flagged) == (3, 1)
    assert sorted(c["channel"]["name"] for c in calls) == ["#imas", "#ll-general"]
    ll = next(c for c in calls if c["channel"]["name"] == "#ll-general")
    assert ll["channel"]["topic"] == "Love Live chat"
    assert [m.get("i") for m in ll["messages"]] == [0, None, 1]  # "w" is context only
    assert b.db.count_unjudged(GUILD) == 0
    assert b.db.leaderboard(GUILD)[0].judged == 3  # context rows never count toward stats


def _seed(n, channel=50):
    b.db.conn.execute("DELETE FROM messages")
    b.db.save_batch(channel, n, [row(i, channel=channel, text=f"post {i}") for i in range(1, n + 1)])


def _guild():
    return NS(id=GUILD, get_channel_or_thread=lambda _: NS(name="general", topic=None, is_nsfw=lambda: False))


def test_worker_pool_keeps_concurrency_full(monkeypatch):
    monkeypatch.setattr(b, "CONCURRENCY", 5)
    in_flight = peak = 0

    async def fake_create(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return NS(usage=None, choices=[NS(finish_reason="stop", message=NS(content='{"flagged": []}'))])

    b.judge.client.chat.completions.create = fake_create
    _seed(40 * 12)  # 12 batches
    assert asyncio.run(b.judge_backlog(_guild())) == (480, 0)
    assert peak == 5
    assert b.db.count_unjudged(GUILD) == 0


def test_worker_pool_stops_when_api_is_down(monkeypatch):
    monkeypatch.setattr(b, "CONCURRENCY", 3)
    calls = 0

    async def broken(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("503")

    b.judge.client.chat.completions.create = broken
    _seed(40 * 50)  # 50 batches
    assert asyncio.run(b.judge_backlog(_guild())) == (0, 0)
    assert calls <= 6  # gave up after a few failures instead of trying all 50
    assert b.db.count_unjudged(GUILD) == 2000  # nothing lost, all still queued
