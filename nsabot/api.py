"""Read-only HTTP API so trusted agents can read the posts the judge ranked, with context.

Security model:
- Every request needs `Authorization: Bearer <key>`. Keys are 256-bit random tokens made with
  `python3 -m nsabot.apikey <name>`; .env only holds their SHA-256 hashes (NSA_API_KEYS), checked in
  constant time. Remove a hash to revoke that key.
- It listens on localhost by default. To reach it from outside, put it behind HTTPS (see README);
  never expose it as plain HTTP.
- Per-key rate limit, lockout after repeated bad keys from one source, small page and context caps,
  only flagged posts and their context (no bulk dump of the chat), only allowlisted servers.
- Read-only: there is no endpoint that changes anything.
"""

import hmac
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from aiohttp import web

from . import scoring
from .apikey import hash_key
from .db import DB
from .judge import message_time

log = logging.getLogger("nsabot.api")
KEY_NAME = web.RequestKey("key_name", str)

MAX_PAGE = 100
MAX_CONTEXT = 30
FAILS_BEFORE_LOCKOUT = 10
LOCKOUT_SECONDS = 600


def parse_keys(raw: str) -> dict[str, str]:
    """NSA_API_KEYS="name:sha256hex,name2:sha256hex" -> {hash: name}."""
    keys = {}
    for part in filter(None, (p.strip() for p in raw.split(","))):
        name, _, digest = part.partition(":")
        digest = digest.strip().lower()
        if not name or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SystemExit(f"NSA_API_KEYS entry {part[:20]!r}… must be name:<64 hex chars> (see python3 -m nsabot.apikey)")
        keys[digest] = name.strip()
    return keys


class Limits:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.calls: defaultdict[str, deque] = defaultdict(deque)
        self.fails: defaultdict[str, deque] = defaultdict(deque)

    def locked_out(self, source: str) -> bool:
        fails, now = self.fails[source], time.monotonic()
        while fails and now - fails[0] > LOCKOUT_SECONDS:
            fails.popleft()
        return len(fails) >= FAILS_BEFORE_LOCKOUT

    def failed(self, source: str) -> None:
        self.fails[source].append(time.monotonic())

    def allow(self, key_name: str) -> bool:
        calls, now = self.calls[key_name], time.monotonic()
        while calls and now - calls[0] > 60:
            calls.popleft()
        if len(calls) >= self.per_minute:
            return False
        calls.append(now)
        return True


def iso(message_id: int) -> str:
    ms = (message_id >> 22) + 1420070400000
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def build_app(db: DB, keys: dict[str, str], guild_ids: set[int], *, per_minute: int = 60,
              channel_name=lambda guild_id, channel_id: None, guild_name=lambda guild_id: None,
              thresholds: dict | None = None) -> web.Application:
    limits = Limits(per_minute)

    def source(request: web.Request) -> str:
        # Behind a local reverse proxy every request comes from the proxy, so use the client it reports.
        forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        return forwarded or request.remote or "?"

    @web.middleware
    async def guard(request: web.Request, handler):
        src = source(request)
        if limits.locked_out(src):
            raise web.HTTPTooManyRequests(text='{"error": "too many bad keys, try later"}', content_type="application/json")
        header = request.headers.get("Authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        digest = hash_key(token) if token else ""
        name = next((n for h, n in keys.items() if hmac.compare_digest(h, digest)), None)
        if name is None:
            limits.failed(src)
            log.warning("API: rejected %s %s from %s", request.method, request.path, src)
            raise web.HTTPUnauthorized(text='{"error": "missing or invalid API key"}', content_type="application/json",
                                       headers={"WWW-Authenticate": "Bearer"})
        if not limits.allow(name):
            raise web.HTTPTooManyRequests(text='{"error": "rate limit"}', content_type="application/json",
                                          headers={"Retry-After": "60"})
        request[KEY_NAME] = name
        response = await handler(request)
        response.headers["Cache-Control"] = "no-store"
        log.info("API: %s %s %s -> %s", name, request.method, request.path_qs[:200], response.status)
        return response

    def bad(msg: str):
        return web.HTTPBadRequest(text=json.dumps({"error": msg}), content_type="application/json")

    def int_arg(request, name, default=None, lo=None, hi=None):
        raw = request.query.get(name)
        if raw is None or raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            raise bad(f"{name} must be an integer")
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        return value

    def guild_arg(request) -> int:
        guild_id = int_arg(request, "guild")
        if guild_id is None:
            if len(guild_ids) == 1:
                return next(iter(guild_ids))
            raise bad("guild is required (see /v1/guilds)")
        if guild_id not in guild_ids:
            raise web.HTTPNotFound(text='{"error": "unknown guild"}', content_type="application/json")
        return guild_id

    def link(guild_id, channel_id, message_id):
        return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

    def context_item(r) -> dict:
        item = {"id": str(r["id"]), "author": r["author_name"], "time": message_time(r["id"]), "text": r["content"]}
        if r["reply_author"] or r["reply_text"]:
            item["reply_to"] = {"author": r["reply_author"], "text": r["reply_text"]}
        if r["extras"]:
            item["attachments"] = r["extras"]
        return item

    def post_item(r) -> dict:
        labels = json.loads(r["labels"]) if r["labels"] else None
        item = context_item(r)
        item.update({
            "author_id": str(r["author_id"]),
            "time": iso(r["id"]),
            "channel": {"id": str(r["channel_id"]), "name": channel_name(r["guild_id"], r["channel_id"])},
            "severity": r["severity"],
            "reason": r["reason"],
            "labels": labels,
            "tags": scoring.describe(labels) if labels else None,
            "deleted": bool(r["deleted"]),
            "url": link(r["guild_id"], r["channel_id"], r["id"]),
        })
        return item

    routes = web.RouteTableDef()

    @routes.get("/v1/guilds")
    async def guilds(request):
        return web.json_response({"guilds": [{"id": str(g), "name": guild_name(g)} for g in sorted(guild_ids)]})

    @routes.get("/v1/posts")
    async def posts(request):
        guild_id = guild_arg(request)
        order = request.query.get("order", "severity")
        if order not in ("severity", "recent"):
            raise bad("order must be severity or recent")
        behaviour = request.query.get("behaviour") or None
        if behaviour and behaviour not in scoring.BASE:
            raise bad(f"behaviour must be one of {sorted(scoring.BASE)}")
        cursor = None
        if raw := request.query.get("cursor"):
            try:
                sev, _, mid = raw.partition(".")
                cursor = (int(sev), int(mid))
            except ValueError:
                raise bad("bad cursor")
        limit = int_arg(request, "limit", 25, 1, MAX_PAGE)
        rows = db.api_posts(
            guild_id, min_severity=int_arg(request, "min_severity", 1, 1, 10), author_id=int_arg(request, "author"),
            channel_id=int_arg(request, "channel"), behaviour=behaviour, order=order, cursor=cursor, limit=limit,
        )
        nxt = f"{rows[-1]['severity']}.{rows[-1]['id']}" if len(rows) == limit else None
        return web.json_response({"posts": [post_item(r) for r in rows], "next_cursor": nxt})

    @routes.get("/v1/posts/{id:\\d+}")
    async def post(request):
        guild_id = guild_arg(request)
        mid = int(request.match_info["id"])
        row = db.api_post(mid)
        if row is None or row["guild_id"] != guild_id or not row["severity"]:  # only flagged posts are served
            raise web.HTTPNotFound(text='{"error": "no flagged post with that id"}', content_type="application/json")
        before = int_arg(request, "before", 15, 0, MAX_CONTEXT)
        after = int_arg(request, "after", 5, 0, MAX_CONTEXT)
        timeline = db.timeline(row["channel_id"], mid, mid, before=before)
        return web.json_response({
            "post": post_item(row),
            "before": [context_item(r) for r in timeline if r["id"] != mid],
            "after": [context_item(r) for r in db.timeline_after(row["channel_id"], mid, after)],
        })

    @routes.get("/v1/leaderboard")
    async def leaderboard(request):
        guild_id = guild_arg(request)
        rows = db.leaderboard(guild_id, int_arg(request, "limit", 10, 1, 50))
        return web.json_response({"leaderboard": [
            {"rank": i, "author_id": str(s.author_id), "author": s.author_name, "points": round(s.points, 1),
             "kimoi_posts": s.hits, "judged_posts": s.judged, "rate": round(s.rate, 4),
             "avg_severity": round(s.avg_severity, 2)}
            for i, s in enumerate(rows, start=1)
        ]})

    @routes.get("/v1/users/{id:\\d+}")
    async def user(request):
        guild_id = guild_arg(request)
        uid = int(request.match_info["id"])
        found = db.standing(guild_id, uid)
        if not found:
            raise web.HTTPNotFound(text='{"error": "no kimoi on file for that user"}', content_type="application/json")
        rank, s = found
        worst = db.api_posts(guild_id, author_id=uid, limit=int_arg(request, "limit", 10, 1, MAX_PAGE))
        return web.json_response({
            "author_id": str(uid), "author": s.author_name, "rank": rank, "points": round(s.points, 1),
            "kimoi_posts": s.hits, "judged_posts": s.judged, "avg_severity": round(s.avg_severity, 2),
            "worst_posts": [post_item(r) for r in worst],
        })

    @routes.get("/v1/scoring")
    async def scoring_info(request):
        return web.json_response({
            "rubric_version": scoring.RUBRIC_VERSION,
            "base": scoring.BASE, "target": scoring.TARGET, "intensity": scoring.INTENSITY,
            "sincerity": scoring.SINCERITY, "doubling_down": scoring.DOUBLING_DOWN, "multi_bonus": scoring.MULTI_BONUS,
            "formula": scoring.formula_lines(), "thresholds": thresholds or {},
        })

    app = web.Application(middlewares=[guard], client_max_size=1024)
    app.add_routes(routes)
    return app


async def start(app: web.Application, host: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    log.info("API listening on %s:%d", host, port)
    return runner

