"""Nobody outside NSA_ADMIN_IDS / NSA_GUILD_IDS can make the bot call DeepSeek."""

import asyncio
from types import SimpleNamespace as NS

import pytest
from discord.ext import commands

import nsabot.bot as b

ADMIN, RANDO, GUILD, OTHER_GUILD = 1, 2, 10, 99
API_COMMANDS = ["scan", "scanall", "watch", "unwatch", "usage", "dossier"]


@pytest.fixture(autouse=True)
def api_calls():
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return NS(usage=NS(total_tokens=1), choices=[NS(message=NS(content='{"flagged": []}'))])

    b.judge.client.chat.completions.create = fake_create
    b.watching.clear()
    b.db.conn.execute("DELETE FROM messages")
    b.spam_limiter._cache.clear()
    yield calls


def guild(gid):
    return NS(id=gid, get_channel=lambda _: None)


def ctx(user, gid=GUILD):
    return NS(author=NS(id=user), guild=guild(gid) if gid else None)


async def passes_checks(command, context) -> bool:
    try:
        for check in [b.allowed_guild, *b.bot.get_command(command).checks]:
            await check(context)
        return True
    except commands.CheckFailure:
        return False


@pytest.mark.parametrize("command", API_COMMANDS)
def test_api_commands_need_admin_in_listed_guild(command):
    assert asyncio.run(passes_checks(command, ctx(ADMIN)))
    assert not asyncio.run(passes_checks(command, ctx(RANDO)))
    assert not asyncio.run(passes_checks(command, ctx(ADMIN, OTHER_GUILD)))
    assert not asyncio.run(passes_checks(command, ctx(ADMIN, None)))  # DMs


@pytest.mark.parametrize("command", ["kimoiboard", "kimoi", "optout", "optin", "possessive", "help"])
def test_public_commands_still_need_listed_guild(command):
    assert asyncio.run(passes_checks(command, ctx(RANDO)))
    assert not asyncio.run(passes_checks(command, ctx(RANDO, OTHER_GUILD)))


def test_process_refuses_unlisted_guild(api_calls):
    with pytest.raises(PermissionError):
        asyncio.run(b.process(guild(OTHER_GUILD)))
    assert api_calls == []


def msg(i, gid=GUILD, author=RANDO):
    g = guild(gid)
    return NS(id=i, guild=g, channel=NS(id=50), content="my waifu is real", clean_content="my waifu is real",
              reference=None, attachments=[], stickers=[], embeds=[],
              author=NS(id=author, bot=False, display_name="x"))


async def flood(n, gid=GUILD):
    for i in range(n):
        await b.intercept(msg(i + 1, gid))
    await asyncio.gather(*b.background)


def test_messages_are_ignored_unless_admin_turned_on_watch(api_calls):
    asyncio.run(flood(20))
    assert api_calls == [] and b.db.count_unjudged(GUILD) == 0


def test_spam_is_rate_limited_per_user(api_calls):
    b.watching[GUILD] = 77
    b.QUEUE_TRIGGER = 10**9  # isolate the limiter from the trigger
    try:
        asyncio.run(flood(20))
    finally:
        b.QUEUE_TRIGGER = 3
    assert b.db.count_unjudged(GUILD) == 5  # NSA_USER_RATE
