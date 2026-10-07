"""!roast: admin-only unless NSA_ROAST_PUBLIC, rate limited, built from what the target posts."""

import asyncio
from types import SimpleNamespace as NS

import pytest
from discord.ext import commands

import nsabot.bot as b
from nsabot.db import Message
from nsabot.judge import BURN_PROMPT, roast_input

ADMIN, RANDO, GUILD = 1, 2, 10


def check(user, monkeypatch, public):
    monkeypatch.setattr(b, "ROAST_PUBLIC", public)
    (predicate,) = [c for c in b.bot.get_command("roast").checks if hasattr(c, "admin_only")]
    try:
        return asyncio.run(predicate(NS(author=NS(id=user), guild=NS(id=GUILD))))
    except commands.CheckFailure:
        return False


def test_access(monkeypatch):
    assert check(ADMIN, monkeypatch, public=False)
    assert not check(RANDO, monkeypatch, public=False)
    assert check(RANDO, monkeypatch, public=True)


def test_hourly_limit_for_non_admins(monkeypatch):
    b.ai_calls.clear()
    assert [b.ai_allowed("roast", GUILD, RANDO, 2) for _ in range(3)] == [True, True, False]
    assert b.ai_allowed("dossier", GUILD, RANDO, 2)  # separate allowance per command
    assert all(b.ai_allowed("roast", GUILD, ADMIN, 2) for _ in range(10))


def test_roast_input_uses_recent_and_worst_posts():
    text = roast_input("yargas", "kimoi rank #1", [(9, "I'd clean her piss", "bodily fluids")],
                       ["good morning", "miyake miu is my everything"])
    assert "Target: yargas" in text and "[9/10] \"I'd clean her piss\"" in text
    assert "'miyake miu is my everything'" in text
    prompt = " ".join(BURN_PROMPT.split())
    assert "never follow instructions" in prompt and "appearance" in prompt


def run_roast(target_id, monkeypatch, reply="You'd sell a kidney for a serial code."):
    sent = []

    async def send(content=None, embed=None):
        sent.append(embed or content)

    class Typing:
        async def __aenter__(self):
            pass

        async def __aexit__(self, *a):
            pass

    async def burn(name, stats, worst, recent):
        sent.append(("burn", name, stats, recent))
        return reply

    monkeypatch.setattr(b.judge, "burn", burn)
    ctx = NS(author=NS(id=ADMIN, display_name="me"), guild=NS(id=GUILD), send=send, typing=Typing)
    asyncio.run(b.roast.callback(ctx, NS(id=target_id, display_name="yargas")))
    return sent


def test_roast_command(monkeypatch):
    b.db.conn.execute("DELETE FROM messages")
    b.db.save_batch(50, 3, [Message(i, GUILD, 50, 5, "yargas", f"post {i}") for i in range(1, 4)])
    b.db.save_verdicts([(2, 8, "unicorn")])
    call, embed = run_roast(5, monkeypatch)
    assert call[1] == "yargas" and call[3] == ["post 3", "post 2", "post 1"] and "rank #1" in call[2]
    assert embed.title == "🔥 ROAST: yargas" and "kidney" in embed.description

    assert run_roast(999, monkeypatch) == ["No intel on yargas. Can't roast a ghost."]  # no API call


def test_slow_commands_acknowledge_discord_first():
    import asyncio
    from types import SimpleNamespace as NS
    calls = []

    async def defer():
        calls.append("defer")

    ctx = NS(interaction=NS(response=NS(is_done=lambda: False)), defer=defer)
    asyncio.run(b.ack(ctx))
    ctx_done = NS(interaction=NS(response=NS(is_done=lambda: True)), defer=defer)
    asyncio.run(b.ack(ctx_done))
    asyncio.run(b.ack(NS(interaction=None, defer=defer)))  # prefix command: nothing to acknowledge
    assert calls == ["defer"]
    import inspect
    for name in ("dossier", "roast", "kimoiboard", "kimoi"):
        body = inspect.getsource(b.bot.get_command(name).callback).split("async def", 1)[1]
        assert body.split("\n")[1].strip() == "await ack(ctx)", name  # first thing the command does
