"""Reply to someone's message and @ the bot: it answers that message using the last 30 messages."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

import nsabot.bot as b

GUILD, BOT_ID = 10, 4242


class Msg:
    def __init__(self, mid, author_id, name, text, channel, reference=None, bot=False, role_mentions=()):
        self.id, self.content, self.clean_content = mid, text, text.replace(f"<@{BOT_ID}>", "@NSA Bot")
        self.author = NS(id=author_id, bot=bot, display_name=name)
        self.channel, self.guild = channel, channel.guild
        self.reference = NS(message_id=reference.id, resolved=reference) if reference else None
        self.attachments, self.stickers, self.embeds = [], [], []
        self.role_mentions = list(role_mentions)
        self.replies, self.reactions = [], []

    async def reply(self, text, mention_author=True):
        self.replies.append(text)

    async def add_reaction(self, emoji):
        self.reactions.append(emoji)


class Channel:
    id, name, topic = 50, "general", None

    def __init__(self, guild):
        self.guild, self.history_msgs = guild, []

    def is_nsfw(self):
        return False

    async def history(self, limit, before):
        for m in sorted((m for m in self.history_msgs if m.id < before.id), key=lambda m: -m.id)[:limit]:
            yield m

    def typing(self):
        class T:
            async def __aenter__(self):
                pass

            async def __aexit__(self, *a):
                pass
        return T()


@pytest.fixture
def chat(monkeypatch):
    guild = NS(id=GUILD, me=NS(id=BOT_ID, display_name="NSA Bot", name="NSA Bot"), self_role=None)
    channel = Channel(guild)
    guild.get_channel_or_thread = lambda cid: channel
    channel.history_msgs = [Msg(i, 6, "pal", f"chat {i}", channel) for i in range(1, 41)]
    target = Msg(41, 5, "yargas", "I'd throw away all my money for 10 seconds with Hayama Fuka", channel)
    channel.history_msgs.append(target)
    monkeypatch.setattr(b, "RESPOND", "on")
    monkeypatch.setattr(b.bot, "_connection", NS(user=NS(id=BOT_ID)), raising=False)
    monkeypatch.setattr(type(b.bot), "user", property(lambda self: NS(id=BOT_ID)))
    b.ai_calls.clear()
    b.db.conn.execute("DELETE FROM messages")
    b.db.conn.execute("DELETE FROM optouts")
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return NS(usage=None, choices=[NS(finish_reason="stop", message=NS(content="Ten seconds. Bold of you to assume she'd stay for ten."))])

    b.judge.client.chat.completions.create = fake_create
    return NS(guild=guild, channel=channel, target=target, calls=calls)


def ping(chat, text=f"<@{BOT_ID}> is this a unicorn take", author=(7, "ハムP"), target=None, mid=100):
    m = Msg(mid, author[0], author[1], text, chat.channel, reference=target or chat.target)
    asyncio.run(b.respond_to_ping(m))
    return m


def test_answers_the_target_with_30_messages_of_context(chat):
    ping(chat)
    (call,) = chat.calls
    payload = json.loads(call["messages"][1]["content"])
    assert len(payload["conversation"]) == 30 and payload["conversation"][-1]["text"].startswith("I'd throw away")
    assert payload["target"]["author"] == "yargas"
    assert payload["request"] == {"author": "ハムP", "text": "is this a unicorn take"}
    assert call["extra_body"]["thinking"]["type"] == "enabled"
    assert chat.target.replies == ["Ten seconds. Bold of you to assume she'd stay for ten."]


def test_kimoi_file_is_included_when_on_file(chat):
    from nsabot.db import Message
    b.db.save_batch(50, 1, [Message(900, GUILD, 50, 5, "yargas", "crying about miu again")])
    b.db.save_verdicts([(900, 8, "meltdown")])
    ping(chat)
    payload = json.loads(chat.calls[0]["messages"][1]["content"])
    assert "kimoi rank #1" in payload["target"]["kimoi_file"]


def test_needs_an_explicit_ping_and_a_reply_to_someone_else(chat):
    ping(chat, text="is this a unicorn take")  # reply without @ (Discord's reply ping doesn't count)
    bot_msg = Msg(60, BOT_ID, "NSA Bot", "🚨 report", chat.channel, bot=True)
    ping(chat, target=bot_msg)  # replying to the bot itself
    plain = Msg(101, 7, "ハムP", f"<@{BOT_ID}> hi", chat.channel)  # ping without a reply
    asyncio.run(b.respond_to_ping(plain))
    ping(chat, text=f"!roast <@{BOT_ID}>")  # commands are left to the command handler
    assert chat.calls == []


def test_opt_outs_and_modes(chat, monkeypatch):
    b.db.opt_out(GUILD, 5)
    ping(chat)  # target opted out
    b.db.opt_in(GUILD, 5)
    b.db.opt_out(GUILD, 6)
    ping(chat)
    payload = json.loads(chat.calls[0]["messages"][1]["content"])
    assert all(m["author"] != "pal" for m in payload["conversation"])  # opted-out chatter left out
    monkeypatch.setattr(b, "RESPOND", "admins")
    ping(chat)  # ハムP isn't an admin
    monkeypatch.setattr(b, "RESPOND", "off")
    ping(chat, author=(1, "admin"))
    assert len(chat.calls) == 1


def test_hourly_limit_reacts_instead_of_replying(chat, monkeypatch):
    monkeypatch.setattr(b, "RESPOND_PER_USER_HOUR", 2)
    msgs = [ping(chat, mid=200 + i) for i in range(3)]
    assert len(chat.calls) == 2 and msgs[2].reactions == ["⏳"]
    ping(chat, author=(1, "admin"), mid=300)  # admins are unlimited
    assert len(chat.calls) == 3


def test_unlimited_by_default(chat):
    assert b.RESPOND_PER_USER_HOUR == 0
    for i in range(25):
        ping(chat, mid=400 + i)
    assert len(chat.calls) == 25
