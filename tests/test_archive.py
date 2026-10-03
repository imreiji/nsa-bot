"""!kimoiposts: every flagged post, most kimoi first, paged with buttons."""

import asyncio
from types import SimpleNamespace as NS

import nsabot.bot as b
from nsabot.db import DB, Message

GUILD = 10


def seed(db, n=25, author_of=lambda i: 5 if i % 2 else 6):
    db.conn.execute("DELETE FROM messages")
    db.save_batch(50, n, [Message(i, GUILD, 50, author_of(i), f"user{author_of(i)}", f"post {i} " + "x" * 400)
                          for i in range(1, n + 1)])
    db.save_verdicts([(i, (i % 10) or 0, f"reason {i}") for i in range(1, n + 1)])


def test_query_orders_pages_and_filters():
    db = DB(":memory:")
    seed(db)
    total, rows = db.kimoi_posts(GUILD, limit=5)
    assert total == 23  # posts 10 and 20 scored 0
    assert [r["severity"] for r in rows] == [9, 9, 8, 8, 7]
    assert [r["id"] for r in rows[:2]] == [19, 9]  # newest first within a score
    total5, rows5 = db.kimoi_posts(GUILD, user_id=5, offset=0, limit=50)
    assert total5 == len(rows5) == 13 and all(r["id"] % 2 for r in rows5)
    assert db.kimoi_posts(GUILD, offset=20, limit=10)[1][-1]["severity"] == 1


def test_page_embed_fits_discord_limits():
    seed(b.db, n=60)
    embed, pages = b.kimoi_page(GUILD, None, 0)
    assert pages == 6  # 54 flagged / 10 per page
    assert len(embed.description) < 4096
    assert embed.description.startswith("`#1` **9/10** · **user")
    assert embed.footer.text == "Page 1/6 · 54 kimoi posts"
    member = NS(id=5, display_name="yargas")
    embed, _ = b.kimoi_page(GUILD, member, 0)
    assert embed.title.endswith("yargas") and "**user5**" not in embed.description


def interaction(user_id):
    sent = {}

    async def edit_message(**kw):
        sent["edit"] = kw

    async def send_message(text, ephemeral=False):
        sent["reply"] = (text, ephemeral)

    return NS(user=NS(id=user_id), response=NS(edit_message=edit_message, send_message=send_message)), sent


def test_pager_buttons():
    seed(b.db, n=60)

    async def run():
        view = b.KimoiPager(owner_id=1, guild_id=GUILD, user=None, pages=6)
        assert view.prev.disabled and view.first.disabled and not view.next.disabled

        it, sent = interaction(1)
        await view.next.callback(it)
        assert view.page == 1 and sent["edit"]["embed"].footer.text.startswith("Page 2/6")
        assert sent["edit"]["embed"].description.startswith("`#11`")

        await view.last.callback(interaction(1)[0])
        assert view.page == 5 and view.next.disabled and view.last.disabled

        await view.next.callback(interaction(1)[0])  # can't go past the end
        assert view.page == 5

        await view.first.callback(interaction(1)[0])
        assert view.page == 0 and view.prev.disabled

        stranger, sent = interaction(2)
        assert not await view.interaction_check(stranger)
        assert sent["reply"][1] is True  # only the stranger sees the "open your own" note

    asyncio.run(run())


def test_find_author_for_people_who_left():
    db = DB(":memory:")
    db.save_batch(50, 3, [Message(1, GUILD, 50, 77, "OldName", "hi"), Message(2, GUILD, 50, 77, "Yargas", "hi"),
                          Message(3, GUILD, 50, 88, "someone", "hi")])
    assert db.find_author(GUILD, "yargas")["author_id"] == 77  # case-insensitive
    assert db.find_author(GUILD, "@Yargas")["author_id"] == 77
    assert db.find_author(GUILD, "77")["author_name"] == "Yargas"  # latest name
    assert db.find_author(GUILD, "<@!77>")["author_id"] == 77
    assert db.find_author(GUILD, "nobody") is None
    assert db.find_author(99, "yargas") is None  # other servers' files stay separate


def test_archive_works_for_someone_who_left(monkeypatch):
    import pytest
    from discord.ext import commands

    async def not_a_member(self, ctx, argument):
        raise commands.MemberNotFound(argument)

    monkeypatch.setattr(commands.MemberConverter, "convert", not_a_member)
    seed(b.db, n=30)
    ctx = NS(guild=NS(id=GUILD))

    async def run():
        who = await b.Suspect().convert(ctx, "user5")
        assert (who.id, who.display_name) == (5, "user5")
        embed, pages = b.kimoi_page(GUILD, who, 0)
        assert embed.title == "🗄️ Kimoi archive: user5" and pages == 2
        with pytest.raises(commands.BadArgument, match="No one called"):
            await b.Suspect().convert(ctx, "ghost")

    asyncio.run(run())
