"""Agent read API: key-only access, rate limits, lockouts, flagged posts with context, read-only."""

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nsabot import api, scoring
from nsabot.apikey import hash_key, new_key
from nsabot.db import DB, Message

GUILD, OTHER = 10, 99
KEY, OLD_KEY = new_key(), new_key()
UNICORN = dict(behaviours=["unicorn"], target="real", sincerity="sincere")
BODILY = dict(behaviours=["bodily_servitude"], target="real", sincerity="sincere")


def make_db():
    db = DB(":memory:")
    msgs = [Message(i, GUILD, 50, 5 if i % 2 else 6, "yargas" if i % 2 else "pal", f"msg {i}") for i in range(1, 41)]
    msgs.append(Message(500, OTHER, 70, 5, "yargas", "other server"))
    db.save_batch(50, 40, msgs)
    db.save_verdicts([(i, 0, None) for i in range(1, 41)], scoring.RUBRIC_VERSION)
    db.save_verdicts([(10, 8, "purity police", UNICORN), (21, 9, "piss", BODILY), (25, 8, "again", UNICORN),
                      (500, 9, "x", BODILY)], scoring.RUBRIC_VERSION)
    return db


def run(test, per_minute=60, keys=None):
    async def go():
        app = api.build_app(make_db(), keys or {hash_key(KEY): "claude"}, {GUILD}, per_minute=per_minute,
                            channel_name=lambda g, c: "#general", guild_name=lambda g: "Idol Hell",
                            thresholds={"report": 6, "var": 7})
        async with TestClient(TestServer(app)) as client:
            await test(client)
    asyncio.run(go())


def auth(key=KEY, src="1.2.3.4"):
    return {"Authorization": f"Bearer {key}", "X-Forwarded-For": src}


def test_keys():
    k = new_key()
    assert k.startswith("nsa_") and len(k) > 40 and new_key() != k
    assert api.parse_keys(f"claude:{hash_key(k)}, other:{'a' * 64}") == {hash_key(k): "claude", "a" * 64: "other"}
    assert api.parse_keys("") == {}
    with pytest.raises(SystemExit):
        api.parse_keys(f"claude:{k}")  # a raw key instead of its hash is refused


def test_no_key_no_data():
    async def t(c):
        for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": f"Basic {KEY}"},
                        {"Authorization": f"Bearer {OLD_KEY}"}):  # not (or no longer) in NSA_API_KEYS
            r = await c.get("/v1/posts", headers=headers)
            assert r.status == 401 and "posts" not in await r.text()
    run(t)


def test_lockout_after_repeated_bad_keys():
    async def t(c):
        for _ in range(api.FAILS_BEFORE_LOCKOUT):
            assert (await c.get("/v1/posts", headers=auth("wrong", src="6.6.6.6"))).status == 401
        assert (await c.get("/v1/posts", headers=auth(src="6.6.6.6"))).status == 429  # even a good key, from there
        assert (await c.get("/v1/posts", headers=auth(src="1.2.3.4"))).status == 200  # others unaffected
    run(t)


def test_rate_limit_per_key():
    async def t(c):
        assert [(await c.get("/v1/guilds", headers=auth())).status for _ in range(4)] == [200, 200, 200, 429]
    run(t, per_minute=3)


def test_read_only():
    async def t(c):
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            assert (await c.request(method, "/v1/posts", headers=auth())).status == 405
    run(t)


def test_posts_flagged_only_sorted_and_paged():
    async def t(c):
        r = await c.get("/v1/posts", headers=auth())
        assert r.headers["Cache-Control"] == "no-store"
        body = await r.json()
        assert [p["id"] for p in body["posts"]] == ["21", "25", "10"]  # flagged only, this server only, worst first
        p = body["posts"][0]
        assert (p["severity"], p["author"], p["channel"]["name"], p["tags"]) == (9, "yargas", "#general",
                                                                                   "bodily/servitude · real person · sincere")
        assert p["url"] == f"https://discord.com/channels/{GUILD}/50/21" and body["next_cursor"] is None

        page1 = await (await c.get("/v1/posts?limit=2", headers=auth())).json()
        page2 = await (await c.get(f"/v1/posts?limit=2&cursor={page1['next_cursor']}", headers=auth())).json()
        assert [p["id"] for p in page1["posts"] + page2["posts"]] == ["21", "25", "10"]

        recent = await (await c.get("/v1/posts?order=recent", headers=auth())).json()
        assert [p["id"] for p in recent["posts"]] == ["25", "21", "10"]
        unicorns = await (await c.get("/v1/posts?behaviour=unicorn&min_severity=8", headers=auth())).json()
        assert [p["id"] for p in unicorns["posts"]] == ["25", "10"]
        assert (await c.get("/v1/posts?behaviour=DROP TABLE", headers=auth())).status == 400
        assert (await c.get(f"/v1/posts?guild={OTHER}", headers=auth())).status == 404  # not an allowed server
    run(t)


def test_one_post_with_context():
    async def t(c):
        body = await (await c.get("/v1/posts/21?before=3&after=2", headers=auth())).json()
        assert body["post"]["id"] == "21"
        assert [m["id"] for m in body["before"]] == ["18", "19", "20"] and [m["id"] for m in body["after"]] == ["22", "23"]
        big = await (await c.get("/v1/posts/21?before=500", headers=auth())).json()
        assert len(big["before"]) == 20  # capped at MAX_CONTEXT (only 20 exist before it here)
        assert (await c.get("/v1/posts/20", headers=auth())).status == 404   # not flagged: not served
        assert (await c.get("/v1/posts/500", headers=auth())).status == 404  # another server's post
    run(t)


def test_leaderboard_user_scoring_guilds():
    async def t(c):
        board = await (await c.get("/v1/leaderboard", headers=auth())).json()
        assert [(e["author"], e["kimoi_posts"]) for e in board["leaderboard"]] == [("yargas", 2), ("pal", 1)]
        user = await (await c.get("/v1/users/5", headers=auth())).json()
        assert user["rank"] == 1 and [p["id"] for p in user["worst_posts"]] == ["21", "25"]
        assert (await c.get("/v1/users/7", headers=auth())).status == 404  # nothing on file
        sc = await (await c.get("/v1/scoring", headers=auth())).json()
        assert sc["base"]["unicorn"] == scoring.BASE["unicorn"] and sc["thresholds"] == {"report": 6, "var": 7}
        g = await (await c.get("/v1/guilds", headers=auth())).json()
        assert g == {"guilds": [{"id": str(GUILD), "name": "Idol Hell"}]}
    run(t)
