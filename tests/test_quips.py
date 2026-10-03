"""The analyst cracks a joke when the judge decides the moment calls for it."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import discord

import nsabot.bot as b
from nsabot.db import Message
from nsabot.judge import QUIP_REQUEST, parse_quip

GUILD = 10


def test_parse_quip():
    assert parse_quip('{"flagged": [], "quip": "  Agent   requests\\n hazard pay. "}') == "Agent requests hazard pay."
    assert parse_quip('{"flagged": []}') is None
    assert parse_quip('{"flagged": [], "quip": 42}') is None
    assert parse_quip("not json") is None
    assert len(parse_quip(json.dumps({"quip": "x" * 999}))) == 300


def test_quip_only_requested_when_rolled():
    sent = []

    async def fake_create(**kwargs):
        sent.append(kwargs["messages"])
        return NS(usage=None, choices=[NS(finish_reason="stop",
                  message=NS(content='{"flagged": [], "quip": "Logging this under hazard pay."}'))])

    b.judge.client.chat.completions.create = fake_create
    assert asyncio.run(b.judge.judge_with_quip({"messages": []}, 0, quip=False)) == ({}, None)
    assert asyncio.run(b.judge.judge_with_quip({"messages": []}, 0, quip=True)) == ({}, "Logging this under hazard pay.")
    assert len(sent[0]) == 2 and len(sent[1]) == 3 and sent[1][2]["content"] == QUIP_REQUEST
    assert sent[0][0] == sent[1][0]  # same system prompt either way (cacheable)


def test_quips_respect_switch_and_cooldown(monkeypatch):
    monkeypatch.setattr(b, "QUIPS", True)
    monkeypatch.setattr(b, "QUIP_COOLDOWN", 1800)
    b.last_quip.clear()
    assert b.quip_allowed(GUILD)
    b.last_quip[GUILD] = __import__("time").monotonic()  # just posted one
    assert not b.quip_allowed(GUILD)
    assert b.quip_allowed(GUILD + 1)  # other servers are separate
    monkeypatch.setattr(b, "QUIPS", False)
    assert not b.quip_allowed(GUILD + 1)


def snowflake(age: timedelta) -> int:
    return discord.utils.time_snowflake(datetime.now(timezone.utc) - age)


class Chan:
    name, topic = "general", None

    def __init__(self):
        self.sent = []

    def is_nsfw(self):
        return False

    async def send(self, text):
        self.sent.append(text)


def test_quip_goes_to_live_channel_or_report_channel(monkeypatch):
    monkeypatch.setattr(b, "QUIPS", True)
    monkeypatch.setattr(b, "QUIP_COOLDOWN", 0)
    b.last_quip.clear()
    source, report = Chan(), Chan()
    guild = NS(id=GUILD, get_channel_or_thread=lambda cid: source if cid == 50 else None,
               get_channel=lambda cid: report if cid == 77 else None)
    monkeypatch.setitem(b.watching, GUILD, 77)

    asyncio.run(b.post_quip(guild, 50, snowflake(timedelta(minutes=5)), "Hazard pay requested."))
    assert source.sent == ["🕵️ Hazard pay requested."] and report.sent == []

    old = snowflake(timedelta(days=200))
    asyncio.run(b.post_quip(guild, 50, old, "This aged badly."))
    assert report.sent == [f"🕵️ *re: [this](https://discord.com/channels/{GUILD}/50/{old})* This aged badly."]


def test_one_quip_per_cooldown_during_a_sweep(monkeypatch):
    monkeypatch.setattr(b, "QUIPS", True)
    monkeypatch.setattr(b, "QUIP_COOLDOWN", 1800)
    monkeypatch.setattr(b, "CONCURRENCY", 2)
    b.last_quip.clear()
    asks = []

    async def fake_create(**kwargs):
        asked = len(kwargs["messages"]) == 3
        asks.append(asked)
        await asyncio.sleep(0)
        content = {"flagged": [], **({"quip": "Undercover and underpaid."} if asked else {})}
        return NS(usage=None, choices=[NS(finish_reason="stop", message=NS(content=json.dumps(content)))])

    b.judge.client.chat.completions.create = fake_create
    b.db.conn.execute("DELETE FROM messages")
    start = snowflake(timedelta(minutes=10))
    b.db.save_batch(50, start + 400, [Message(start + i, GUILD, 50, 5, "u", f"post {i}") for i in range(400)])
    source = Chan()
    guild = NS(id=GUILD, get_channel_or_thread=lambda cid: source, get_channel=lambda cid: None)

    judged, _ = asyncio.run(b.judge_backlog(guild))
    assert judged == 400
    assert source.sent == ["🕵️ Undercover and underpaid."]  # the model joked every time it could; one got posted
    assert asks[:2] == [True, True] and not any(asks[2:])  # after posting, it stops even offering
