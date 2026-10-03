"""VAR mode: self-deleted posts are replayed with context and aired if kimoi."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

import nsabot.bot as b
from nsabot.db import Message
from nsabot.judge import VAR_NOTE

GUILD, CHAN, REPORT = 10, 50, 77


class Chan:
    name, topic = "general", None

    def __init__(self):
        self.embeds = []

    def is_nsfw(self):
        return False

    async def send(self, content=None, embed=None):
        self.embeds.append(embed)


@pytest.fixture
def env(monkeypatch):
    report, source = Chan(), Chan()
    audit = []

    async def audit_logs(limit, action):
        for entry in audit:
            yield entry

    async def no_wait(_):
        pass

    monkeypatch.setattr(b.asyncio, "sleep", no_wait)
    b.audit_counts.clear()
    b.unclaimed_mod_deletes.clear()
    guild = NS(id=GUILD, me=NS(guild_permissions=NS(view_audit_log=True)), audit_logs=audit_logs,
               get_channel=lambda cid: report if cid == REPORT else None,
               get_channel_or_thread=lambda cid: source)
    monkeypatch.setattr(b.bot, "get_guild", lambda gid: guild if gid == GUILD else None)
    monkeypatch.setitem(b.watching, GUILD, REPORT)
    monkeypatch.setattr(b, "VAR", True)
    monkeypatch.setattr(b, "VAR_MIN_SEVERITY", 5)
    b.var_calls.clear()
    b.db.conn.execute("DELETE FROM messages")
    b.db.conn.execute("DELETE FROM optouts")
    b.db.save_batch(CHAN, 7, [Message(i, GUILD, CHAN, 5 if i == 4 else 6, "yargas" if i == 4 else "pal",
                                      "I'd clean her piss" if i == 4 else f"msg {i}") for i in range(1, 8)])

    calls = []

    def answer(severity):
        async def fake_create(**kwargs):
            calls.append(kwargs["messages"])
            flagged = [{"i": 0, "severity": severity, "reason": "caught on replay"}] if severity else []
            return NS(usage=None, choices=[NS(finish_reason="stop", message=NS(content=json.dumps({"flagged": flagged})))])
        b.judge.client.chat.completions.create = fake_create

    return NS(guild=guild, report=report, calls=calls, answer=answer, audit=audit)


def delete(mid, cached=None, guild=GUILD):
    asyncio.run(b.var_review(NS(guild_id=guild, message_id=mid, channel_id=CHAN, cached_message=cached)))


def test_deleted_post_is_replayed_with_context_and_aired(env):
    env.answer(9)
    delete(4)
    (messages,) = env.calls
    payload = json.loads(messages[1]["content"])
    assert [m.get("i") for m in payload["messages"]] == [None, None, None, 0, None, None, None]  # 3 before, 3 after
    assert messages[2]["content"] == VAR_NOTE
    (embed,) = env.report.embeds
    assert embed.title == "📺 VAR REVIEW — yargas deleted a post"
    assert "9/10" in embed.fields[0].value
    row = b.db.get_message(4)
    assert (row["severity"], row["deleted"], row["reported"]) == (9, 1, 1)
    archive, _ = b.kimoi_page(GUILD, None, 0)
    assert "📺 deleted, caught by VAR" in archive.description

    delete(4)  # a second delete event for the same post doesn't air it twice
    assert len(env.report.embeds) == 1 and len(env.calls) == 1


def test_already_scored_post_costs_nothing(env):
    env.answer(9)
    b.db.save_verdicts([(4, 7, "known")])
    delete(4)
    assert env.calls == [] and "7/10" in env.report.embeds[0].fields[0].value


def test_low_scores_stay_deleted(env):
    env.answer(2)
    delete(4)
    assert env.report.embeds == [] and b.db.get_message(4)["deleted"] == 1


def entry(eid, count=1, age=timedelta(seconds=3), target=5, channel=CHAN):
    return NS(id=eid, created_at=datetime.now(timezone.utc) - age, target=NS(id=target),
              extra=NS(channel=NS(id=channel), count=count))


def test_mod_deletions_are_skipped(env):
    env.audit.append(entry(1))
    env.answer(9)
    delete(4)
    assert env.calls == [] and env.report.embeds == []


def test_merged_mod_deletion_is_caught(env):
    """Discord bumps the count on an old entry instead of adding a new one."""
    env.audit.append(entry(1, count=1, age=timedelta(minutes=10)))
    asyncio.run(b.refresh_mod_deletes(env.guild, prime=True))  # bot started; entry already there
    env.audit[0] = entry(1, count=2, age=timedelta(minutes=10))  # same mod deletes again: count bump
    env.answer(9)
    delete(4)
    assert env.calls == [] and env.report.embeds == []


def test_old_unchanged_entries_dont_block_self_deletes(env):
    env.audit.append(entry(1, count=3, age=timedelta(minutes=10)))
    asyncio.run(b.refresh_mod_deletes(env.guild, prime=True))
    env.answer(9)
    delete(4)  # nothing new in the audit log: the author deleted it
    assert len(env.report.embeds) == 1


def test_each_mod_deletion_covers_one_delete(env):
    """One mod deletion of yargas's message doesn't hide yargas's own delete a minute later."""
    asyncio.run(b.refresh_mod_deletes(env.guild, prime=True))
    env.audit.append(entry(1))
    env.answer(9)
    from nsabot.db import Message
    b.db.save_batch(CHAN, 8, [Message(8, GUILD, CHAN, 5, "yargas", "another one")])
    delete(8)  # the mod's deletion
    delete(4)  # yargas deleting their own
    assert len(env.calls) == 1 and len(env.report.embeds) == 1


def test_without_audit_log_access_var_stays_quiet(env):
    env.guild.me.guild_permissions.view_audit_log = False
    env.answer(9)
    delete(4)
    assert env.calls == [] and env.report.embeds == []


def test_things_vars_ignores(env):
    env.answer(9)
    delete(999)  # never seen
    delete(4, guild=99)  # unlisted server
    b.db.opt_out(GUILD, 5)
    delete(4)  # opted out (and their posts are gone anyway)
    bot_msg = NS(content="beep boop", clean_content="beep boop", author=NS(id=1234, bot=True, display_name="bot"),
                 attachments=[], stickers=[], embeds=[], reference=None)
    delete(1000, cached=bot_msg)
    assert env.calls == [] and env.report.embeds == []


def test_post_and_delete_spam_is_rate_limited(env, monkeypatch):
    monkeypatch.setattr(b, "VAR_PER_USER_HOUR", 2)
    env.answer(1)
    b.db.save_batch(CHAN, 30, [Message(i, GUILD, CHAN, 5, "yargas", f"spam {i}") for i in range(20, 30)])
    for mid in range(20, 30):
        delete(mid)
    assert len(env.calls) == 2


def test_message_only_in_discords_cache(env):
    env.answer(8)
    cached = NS(id=500, guild=env.guild, channel=NS(id=CHAN), content="she's MY seiyuu",
                clean_content="she's MY seiyuu", author=NS(id=5, bot=False, display_name="yargas"),
                attachments=[], stickers=[], embeds=[], reference=None)
    delete(500, cached=cached)
    assert len(env.calls) == 1 and "8/10" in env.report.embeds[0].fields[0].value
    assert b.db.get_message(500)["content"] == "she's MY seiyuu"
