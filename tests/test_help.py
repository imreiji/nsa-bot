"""!help lists public commands for everyone and admin commands only for admins."""

import asyncio
from types import SimpleNamespace as NS

import nsabot.bot as b

ADMIN, RANDO = 1, 2


def run_help(user, name=None):
    sent = []

    async def send(content=None, embed=None):
        sent.append(embed or content)

    asyncio.run(b.help_.callback(NS(author=NS(id=user), send=send), name))
    return sent[0]


def test_everyone_sees_public_commands_admins_also_see_admin_ones():
    public = run_help(RANDO)
    assert [f.name for f in public.fields] == ["Everyone"]
    assert "`!kimoiposts [member]`" in public.fields[0].value and "!scan" not in public.fields[0].value

    admin = run_help(ADMIN)
    assert [f.name for f in admin.fields] == ["Everyone", "Admins (spend DeepSeek credit)"]
    for cmd in ["scan", "scanall", "watch", "unwatch", "usage", "dossier"]:
        assert f"`!{cmd}" in admin.fields[1].value
    assert all(len(f.value) <= 1024 for f in admin.fields)  # Discord's field limit
    every_command = {c.name for c in b.bot.commands}
    listed = admin.fields[0].value + admin.fields[1].value
    assert all(f"`!{name}" in listed for name in every_command)


def test_help_for_one_command():
    e = run_help(RANDO, "archive")  # aliases work
    assert e.title == "`!kimoiposts [member]`" and "`!archive`" in e.fields[0].value
    assert "No command called" in run_help(RANDO, "scan")  # admin commands stay hidden from others
    assert run_help(ADMIN, "!scan").footer.text == "Admins only"
