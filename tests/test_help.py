"""!help lists public commands for everyone and admin commands only for admins."""

import asyncio
from types import SimpleNamespace as NS

import nsabot.bot as b

ADMIN, RANDO = 1, 2


def run_help(user, name=None):
    sent = []

    async def send(content=None, embed=None, ephemeral=False):
        sent.append(embed or content)

    asyncio.run(b.help_.callback(NS(author=NS(id=user), send=send), name))
    return sent[0]


def test_everyone_sees_public_commands_admins_also_see_admin_ones():
    public = run_help(RANDO)
    assert [f.name for f in public.fields] == ["Everyone"]
    assert "`/archive [member]`" in public.fields[0].value and "/scan" not in public.fields[0].value
    assert "`/dossier [member]`" in public.fields[0].value  # public now

    admin = run_help(ADMIN)
    assert [f.name for f in admin.fields][:2] == ["Everyone", "Admins (spend API credit)"]
    assert all(f.name == "Admins (cont.)" for f in admin.fields[2:])  # a long list carries over
    admin_text = "".join(f.value for f in admin.fields[1:])
    for cmd in ["scan", "scanall", "watch", "unwatch", "usage", "look"]:
        assert f"`/{cmd}" in admin_text
    assert all(len(f.value) <= 1024 for f in admin.fields)  # Discord's field limit
    every_command = {c.name for c in b.bot.commands}
    listed = admin.fields[0].value + admin_text
    assert all(f"`/{name}" in listed for name in every_command)


def test_help_for_one_command():
    e = run_help(RANDO, "kimoiposts")  # old names still work
    assert e.title == "`/archive [member]`" and "`!archive`" in e.fields[0].value and "`!kimoiposts`" in e.fields[0].value
    assert "No command called" in run_help(RANDO, "scan")  # admin commands stay hidden from others
    assert run_help(ADMIN, "!scan").footer.text == "Admins only"
    assert run_help(ADMIN, "/scan").footer.text == "Admins only"


def test_scan_status_is_a_normal_message_for_both_slash_and_prefix(monkeypatch):
    async def fake_scrape(ch, opted_out, progress):
        await progress("📡 Reading…", force=True)
        return 3

    async def fake_process(guild, progress):
        return 3, 1, 0

    monkeypatch.setattr(b, "scrape", fake_scrape)
    monkeypatch.setattr(b, "process", fake_process)

    for interaction in (object(), None):
        replies, channel_msgs, edits = [], [], []

        class Status:
            async def edit(self, content):
                edits.append(content)

        async def reply(content=None, ephemeral=False, **kw):
            replies.append((content, ephemeral))

        async def channel_send(content):
            channel_msgs.append(content)
            return Status()

        ctx = NS(guild=NS(id=99_000 + (interaction is None)), interaction=interaction, send=reply,
                 channel=NS(send=channel_send))
        asyncio.run(b.run_scan(ctx, [NS(name="general", mention="#general")]))
        assert channel_msgs == ["📡 Intercepting 1 channel(s)…"]  # editable for as long as the scan runs
        assert edits[-1].startswith("✅ Sweep complete: 3 new posts")
        assert replies == ([("📡 Sweep started. Progress below.", True)] if interaction else [])
