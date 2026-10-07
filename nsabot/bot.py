"""Discord front end: watch channels, feed posts to the model judge, report and rank the kimoi."""

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict, deque
from datetime import timedelta
from types import SimpleNamespace
from typing import Literal

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv

from . import api, evaluate, scoring
from .db import DB, Message
from .judge import (PARSE_STATS, VAR_NOTE, Judge, Refused, Truncated, Verdict, anchors_text, build_payload,
                    message_time)

log = logging.getLogger("nsabot")

BATCH_SIZE = 40          # messages per DeepSeek call
MIN_SPLIT = 5            # smallest batch worth splitting again when the model runs out of tokens
MAX_FAILURES = 20        # consecutive failed calls before a sweep gives up (API down, out of credit)
SAVE_EVERY = 500         # scraped messages per DB write / cursor checkpoint
REPORT_MAX_PER_RUN = 20  # report-channel posts per sweep; the rest are summarised


def id_set(name: str) -> set[int]:
    raw = os.getenv(name, "").replace(" ", "")
    try:
        return {int(x) for x in raw.split(",") if x}
    except ValueError:
        raise SystemExit(f"{name} must be comma-separated Discord IDs, got {raw!r}") from None


load_dotenv()
ADMIN_IDS = id_set("NSA_ADMIN_IDS")            # the only users who can spend API credits
GUILD_IDS = id_set("NSA_GUILD_IDS")            # the only servers the bot will stay in
IGNORE_CHANNEL_IDS = id_set("NSA_IGNORE_CHANNEL_IDS")
SCAN_LIMIT = int(os.getenv("NSA_SCAN_LIMIT", "5000"))
QUEUE_TRIGGER = int(os.getenv("NSA_QUEUE_TRIGGER", "40"))
HEARTBEAT_MINUTES = float(os.getenv("NSA_HEARTBEAT_MINUTES", "10"))
REPORT_MIN_SEVERITY = int(os.getenv("NSA_REPORT_MIN_SEVERITY", "5"))
ROAST_PUBLIC = os.getenv("NSA_ROAST_PUBLIC", "off").lower() in ("on", "1", "true")  # let everyone use /roast
ROAST_PER_USER_HOUR = int(os.getenv("NSA_ROAST_PER_USER_HOUR", "3"))  # when public; admins are exempt
DOSSIER_PUBLIC = os.getenv("NSA_DOSSIER_PUBLIC", "on").lower() in ("on", "1", "true")  # let everyone use /dossier
DOSSIER_PER_USER_HOUR = int(os.getenv("NSA_DOSSIER_PER_USER_HOUR", "3"))
API_KEYS = api.parse_keys(os.getenv("NSA_API_KEYS", ""))  # agent read API; off without keys
API_HOST = os.getenv("NSA_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("NSA_API_PORT", "8787"))
API_RATE = int(os.getenv("NSA_API_RATE_PER_MINUTE", "60"))
RESPOND = os.getenv("NSA_RESPOND", "on").lower()  # on = everyone, admins, off
RESPOND_PER_USER_HOUR = int(os.getenv("NSA_RESPOND_PER_USER_HOUR", "0"))  # 0 = unlimited
RESPOND_THINKING = os.getenv("NSA_RESPOND_THINKING", "on").lower() in ("on", "1", "true")
RESPOND_CONTEXT = 30  # messages before the ping shown as context
VAR = os.getenv("NSA_VAR", "on").lower() not in ("off", "0", "false")  # review self-deleted posts
VAR_MIN_SEVERITY = int(os.getenv("NSA_VAR_MIN_SEVERITY") or REPORT_MIN_SEVERITY)
VAR_PER_USER_HOUR = int(os.getenv("NSA_VAR_PER_USER_HOUR", "5"))  # DeepSeek reviews per person per hour
VAR_AFTER = 5  # messages after the deleted one shown as context (how people reacted)
MIN_CHARS = int(os.getenv("NSA_MIN_CHARS", "3"))
IMAGES = os.getenv("NSA_IMAGES", "on").lower() not in ("off", "0", "false")  # let the judge see pictures
IMAGES_PER_MESSAGE = 2
IMAGES_PER_BATCH = int(os.getenv("NSA_IMAGES_PER_BATCH", "10"))  # up to ~1k tokens each
IMAGE_MAX_BYTES = 8 * 1024 * 1024
IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")  # no GIFs or stickers
QUIPS = os.getenv("NSA_QUIPS", "on").lower() not in ("off", "0", "false")  # the judge decides when to joke
QUIP_COOLDOWN = float(os.getenv("NSA_QUIP_COOLDOWN_MINUTES", "30")) * 60  # at most one per server this often
CONCURRENCY = max(1, int(os.getenv("NSA_CONCURRENCY", "16")))  # DeepSeek calls in flight at once
CONTEXT_MESSAGES = int(os.getenv("NSA_CONTEXT_MESSAGES", "15"))  # earlier messages shown before each batch
AFTER_MESSAGES = 5  # later messages shown after each batch when they exist (how people reacted)
USER_RATE = int(os.getenv("NSA_USER_RATE", "10"))  # live posts queued per user per minute; extra spam is dropped
PREFIX = os.getenv("NSA_PREFIX", "!")

DB_PATH = os.getenv("NSA_DB_PATH", "nsa.db")
db = DB(DB_PATH)
# Admin-maintained who's-who for the judge (members, seiyuu, characters, running bits). Lives next to
# the database, outside the repo, and is managed with /notes.
SERVER_NOTES_PATH = os.getenv("NSA_SERVER_NOTES", os.path.join(os.path.dirname(DB_PATH) or ".", "server_notes.md"))
SERVER_NOTES_MAX = 8000
PROVIDER = os.getenv("NSA_PROVIDER", "anthropic").lower()  # anthropic (Claude) or deepseek
if PROVIDER == "anthropic":
    judge = Judge(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        thinking=os.getenv("NSA_THINKING", "on").lower() not in ("off", "0", "false", "disabled"),
        effort=os.getenv("NSA_EFFORT") or None,  # low, medium, high, xhigh, max; blank = model default
        thinking_tokens=int(os.getenv("NSA_THINKING_TOKENS", "32000")),
        db=db,
        provider="anthropic",
    )
else:
    judge = Judge(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        thinking=os.getenv("DEEPSEEK_THINKING", "on").lower() not in ("off", "0", "false", "disabled"),
        effort=os.getenv("DEEPSEEK_REASONING_EFFORT") or None,
        thinking_tokens=int(os.getenv("DEEPSEEK_THINKING_TOKENS", "32000")),
        db=db,
    )

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(
    command_prefix=PREFIX,
    intents=intents,
    # Reposted messages must never ping anyone (@everyone, roles, users).
    allowed_mentions=discord.AllowedMentions.none(),
    help_command=None,  # replaced by the !help command below
)
guild_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
watching: dict[int, int] = {}       # guild_id -> report channel id
spam_limiter = commands.CooldownMapping.from_cooldown(USER_RATE, 60, commands.BucketType.member)
ai_calls: defaultdict[tuple[str, int, int], deque] = defaultdict(deque)  # (command, guild, user) -> call times
var_calls: defaultdict[tuple[int, int], deque] = defaultdict(deque)  # (guild, user) -> review times
last_quip: dict[int, float] = {}  # guild_id -> monotonic time of the last posted quip
background: set[asyncio.Task] = set()  # keep references so tasks aren't garbage-collected


# --- access control ---------------------------------------------------------
#
# DeepSeek is only ever called from two places, and both are gated:
#   - process(): run by !scan/!scanall (admin UIDs only), or by live watching, which only an
#     admin can switch on (!watch) and only in an allowlisted server. process() re-checks the
#     allowlist itself as a last line of defence.
#   - !dossier: admin UIDs only.
# Everyone else's commands (!kimoiboard, !kimoi, !optout, !optin) only touch the local database.

@bot.check
async def allowed_guild(ctx: commands.Context) -> bool:
    if ctx.guild is None or ctx.guild.id not in GUILD_IDS:
        raise commands.CheckFailure("not an allowed server")
    return True


def deployer_only():
    """Anything that spends DeepSeek credits or changes what the bot watches."""
    async def predicate(ctx: commands.Context) -> bool:
        if ctx.author.id not in ADMIN_IDS:
            raise commands.CheckFailure("You lack clearance for that.")
        return True
    predicate.admin_only = True  # lets !help sort commands into public and admin
    return commands.check(predicate)


def public_ai(flag: str):
    """Admins always; everyone else only while the named *_PUBLIC flag is on (hourly limits in the command)."""
    async def predicate(ctx: commands.Context) -> bool:
        if ctx.author.id in ADMIN_IDS or globals()[flag]:
            return True
        raise commands.CheckFailure("You lack clearance for that.")
    predicate.admin_only = not globals()[flag]
    return commands.check(predicate)


def ai_allowed(kind: str, guild_id: int, user_id: int, per_hour: int) -> bool:
    """Non-admins get `per_hour` uses of a DeepSeek-backed fun command per hour (0 = unlimited)."""
    if user_id in ADMIN_IDS or per_hour <= 0:
        return True
    calls, now = ai_calls[(kind, guild_id, user_id)], time.monotonic()
    while calls and now - calls[0] > 3600:
        calls.popleft()
    if len(calls) >= per_hour:
        return False
    calls.append(now)
    return True


def admin_only(command: commands.Command) -> bool:
    return any(getattr(check, "admin_only", False) for check in command.checks)


async def leave_if_unlisted(guild: discord.Guild) -> None:
    if guild.id not in GUILD_IDS:
        log.warning("leaving unlisted guild %s (%s)", guild.name, guild.id)
        await guild.leave()


# --- helpers ----------------------------------------------------------------

def jump_url(guild_id: int, channel_id: int, message_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def watchable(guild_id: int, channel_id: int) -> bool:
    return channel_id not in IGNORE_CHANNEL_IDS and channel_id != watching.get(guild_id)


CUSTOM_EMOJI = re.compile(r"<a?:(\w+):\d+>")


def plain_text(m: discord.Message) -> str:
    """Mentions resolved to names, custom emoji shortened to :name:."""
    return CUSTOM_EMOJI.sub(r":\1:", m.clean_content).strip()


def describe_extras(m: discord.Message) -> str | None:
    """Attachments, stickers, link previews and forwards as text the judge can read."""
    parts = [f"{a.content_type or 'file'}: {a.filename}" for a in m.attachments]
    parts += [f"sticker: {s.name}" for s in m.stickers]
    for e in m.embeds:
        bits = [x for x in (e.provider.name if e.provider else None, e.title, e.description) if x]
        if bits:
            parts.append("link preview: " + clip(" - ".join(bits), 150))
    for snap in getattr(m, "message_snapshots", None) or []:
        if snap.content:
            parts.append("forwarded message: " + clip(snap.content, 200))
    return "; ".join(parts) or None


def image_urls(m: discord.Message) -> list[str]:
    """Pictures the judge can look at: image attachments and link-preview pictures (no GIFs, stickers or video)."""
    urls = [a.url for a in m.attachments
            if (a.content_type or "").split(";")[0] in IMAGE_TYPES and (a.size or 0) <= IMAGE_MAX_BYTES]
    for e in m.embeds:
        if e.type in ("gifv", "video"):
            continue
        pic = e.image or e.thumbnail
        if pic and (pic.proxy_url or pic.url):
            urls.append(pic.proxy_url or pic.url)
    return urls[:IMAGES_PER_MESSAGE]


def classify(m: discord.Message, opted_out: set[int]) -> bool | None:
    """True = score it, False = keep only as context for its neighbours, None = ignore."""
    if m.author.bot or m.author.id in opted_out:
        return None
    text = m.content.strip()
    if text.startswith(PREFIX):  # bot commands
        return None
    if len(text) >= MIN_CHARS:
        return True
    if IMAGES and any((a.content_type or "").split(";")[0] in IMAGE_TYPES for a in m.attachments):
        return True  # a picture post can be judged on the picture
    if text or m.attachments or m.stickers or m.embeds or getattr(m, "message_snapshots", None):
        return False  # "w", "lol", image-only posts: tells the judge how people reacted
    return None


def to_row(m: discord.Message, scored: bool, opted_out: set[int]) -> Message:
    row = Message(m.id, m.guild.id, m.channel.id, m.author.id, m.author.display_name, plain_text(m), scored)
    if m.reference and m.reference.message_id:
        row.reply_to_id = m.reference.message_id
        target = m.reference.resolved
        if isinstance(target, discord.Message):
            row.reply_author_id = target.author.id
            if target.author.id not in opted_out:  # never store an opted-out user's words
                row.reply_author = target.author.display_name
                row.reply_text = clip(plain_text(target) or describe_extras(target) or "", 300)
    row.extras = describe_extras(m)
    row.images = json.dumps(image_urls(m))
    return row


# --- pictures for the judge -----------------------------------------------------

http_session: aiohttp.ClientSession | None = None


async def download_image(url: str) -> str | None:
    """A picture as a data: URL (the model can't fetch Discord's signed links itself), or None."""
    global http_session
    if http_session is None or http_session.closed:
        http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
    try:
        async with http_session.get(url) as resp:
            kind = (resp.headers.get("Content-Type") or "").split(";")[0]
            if resp.status != 200 or kind not in IMAGE_TYPES or (resp.content_length or 0) > IMAGE_MAX_BYTES:
                return None
            data = bytearray()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                data += chunk
                if len(data) > IMAGE_MAX_BYTES:
                    return None
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None
    return f"data:{kind};base64,{base64.b64encode(bytes(data)).decode()}"


async def fresh_image_urls(guild: discord.Guild, channel_id: int, message_id: int) -> list[str]:
    """Re-read a message for its current picture links (Discord's expire after about a day)."""
    channel = guild.get_channel_or_thread(channel_id)
    if channel is None:
        return []
    try:
        message = await channel.fetch_message(message_id)
    except discord.HTTPException:
        db.set_images(message_id, [])  # gone or unreadable: don't try again
        return []
    urls = image_urls(message)
    db.set_images(message_id, urls)
    return urls


async def batch_images(guild: discord.Guild, channel_id: int, order: list[int]) -> list[tuple[int, str]]:
    """(index, data URL) for the pictures in a batch's scored messages, at most IMAGES_PER_BATCH."""
    if not IMAGES:
        return []
    out: list[tuple[int, str]] = []
    rows = db.images_for(order)
    for i, mid in enumerate(order):
        row = rows.get(mid)
        if row is None:
            continue
        if row["images"] is None:  # scraped before pictures were recorded: look only if it had any
            extras = row["extras"] or ""
            urls = await fresh_image_urls(guild, channel_id, mid) if ("image/" in extras or "link preview" in extras) else []
        else:
            urls = json.loads(row["images"])
        refreshed = False
        for url in urls:
            if len(out) >= IMAGES_PER_BATCH:
                return out
            data = await download_image(url)
            if data is None and not refreshed and row["images"] is not None:
                refreshed = True  # probably an expired link: get fresh ones and retry this message once
                fresh = await fresh_image_urls(guild, channel_id, mid)
                for again in fresh:
                    if len(out) < IMAGES_PER_BATCH and (data := await download_image(again)):
                        out.append((i, data))
                break
            if data:
                out.append((i, data))
    return out


def channel_info(guild: discord.Guild, channel_id: int) -> dict:
    ch = guild.get_channel_or_thread(channel_id)
    if ch is None:
        return {"name": "#unknown"}
    parent = getattr(ch, "parent", None)
    info = {"name": f"#{parent.name} > {ch.name}" if isinstance(ch, discord.Thread) and parent else f"#{ch.name}"}
    topic = getattr(ch, "topic", None) or getattr(parent, "topic", None)
    if topic:
        info["topic"] = clip(topic, 200)
    info["nsfw"] = bool(ch.is_nsfw()) if hasattr(ch, "is_nsfw") else False
    return info


class Suspect(commands.Converter):
    """A server member, or someone who has left but is still on file (by ID, mention or old name)."""

    async def convert(self, ctx: commands.Context, argument: str):
        try:
            return await commands.MemberConverter().convert(ctx, argument)
        except commands.BadArgument:
            pass
        row = db.find_author(ctx.guild.id, argument)
        if row is None:
            raise commands.BadArgument(f'No one called "{argument}" on file.')
        return SimpleNamespace(id=row["author_id"], display_name=row["author_name"])


def quip_allowed(guild_id: int) -> bool:
    """Whether the judge may joke right now: quips on and this server's cooldown has passed."""
    return QUIPS and (guild_id not in last_quip or time.monotonic() - last_quip[guild_id] >= QUIP_COOLDOWN)


async def post_quip(guild: discord.Guild, channel_id: int, last_message_id: int, text: str) -> None:
    """Blurt it into the channel if the chat is live; old history goes to the report channel with a link.

    Parallel batches can each come back with a joke; only the first one after the cooldown is posted.
    """
    if not quip_allowed(guild.id):
        return
    last_quip[guild.id] = time.monotonic()
    fresh = discord.utils.utcnow() - discord.utils.snowflake_time(last_message_id) < timedelta(hours=1)
    try:
        if fresh and (ch := guild.get_channel_or_thread(channel_id)):
            await ch.send(f"🕵️ {text}")
        elif report_ch := guild.get_channel(watching.get(guild.id, 0)):
            await report_ch.send(f"🕵️ *re: [this]({jump_url(guild.id, channel_id, last_message_id)})* {text}")
        else:
            return
        log.info("quip in %s: %s", channel_id, text)
    except discord.HTTPException:
        log.warning("couldn't post a quip in %s", channel_id)


anchor_cache: dict[int, tuple[float, str]] = {}


def tags(labels_json: str | None) -> str:
    return scoring.describe(json.loads(labels_json)) if labels_json else ""


_notes_cache: tuple[float, str] = (-1.0, "")


def server_notes() -> str:
    """The admins' server notes, re-read when the file changes."""
    global _notes_cache
    try:
        mtime = os.path.getmtime(SERVER_NOTES_PATH)
    except OSError:
        return ""
    if mtime != _notes_cache[0]:
        with open(SERVER_NOTES_PATH, encoding="utf-8") as f:
            _notes_cache = (mtime, f.read()[:SERVER_NOTES_MAX])
    return _notes_cache[1]


def prompt_extras(guild_id: int) -> str:
    """Server notes plus calibration examples, appended to the judge's system prompt."""
    notes = server_notes().strip()
    head = ("\n\nServer notes from the admins (who's who; trust these over your own guesses):\n" + notes) if notes else ""
    return head + anchors_for(guild_id)


def anchors_for(guild_id: int) -> str:
    """Up to 2 admin-scored posts per score band where the formula agrees (within 1), for the prompt."""
    cached = anchor_cache.get(guild_id)
    if cached and time.monotonic() - cached[0] < 600:
        return cached[1]
    bands: dict[int, list] = {0: [], 1: [], 4: [], 7: [], 9: []}
    for r in db.gold_rows(guild_id):
        if r["rubric_version"] != scoring.RUBRIC_VERSION:
            continue  # judged under an older rubric: its labels don't apply
        labels = json.loads(r["labels"]) if r["labels"] else None
        bot_score = scoring.score(labels) if labels else 0
        admin = r["admin_score"]
        if abs(bot_score - admin) > 1:
            continue
        band = 0 if admin == 0 else 1 if admin <= 3 else 4 if admin <= 6 else 7 if admin <= 8 else 9
        if len(bands[band]) < 2:
            bands[band].append((r["content"], labels, admin))
    text = anchors_text([example for band in bands.values() for example in band])
    anchor_cache[guild_id] = (time.monotonic(), text)
    return text


# --- pipeline: scrape -> judge -> report ------------------------------------

class Progress:
    """Edits a status message at most every `every` seconds (Discord rate-limits edits) and logs it."""

    def __init__(self, message: discord.Message | None = None, every: float = 10.0):
        self.message = message
        self.every = every
        self.last: float | None = None

    async def __call__(self, text: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and self.last is not None and now - self.last < self.every:
            return
        self.last = now
        log.info("progress: %s", text)
        if self.message:
            try:
                await self.message.edit(content=text)
            except discord.HTTPException:
                pass  # status message deleted or uneditable; the scan carries on



# Anything with a message history: text channels, the built-in text chat of voice and stage
# channels, and threads.
Scannable = discord.TextChannel | discord.VoiceChannel | discord.StageChannel | discord.Thread


async def scrape_recent(channel: Scannable, opted_out: set[int], count: int) -> int:
    """Pull only the channel's latest `count` messages, leaving its scan cursor where it was, so a
    later full /scan still reads the older history. Posts already on file keep their verdicts."""
    log.info("reading the latest %d messages of #%s", count, channel.name)
    rows = [to_row(m, scored, opted_out) async for m in channel.history(limit=count)
            if (scored := classify(m, opted_out)) is not None]
    rows.reverse()
    known = db.images_for([r.id for r in rows]) if rows else {}
    new = [r for r in rows if r.id not in known]
    db.save_messages(new)
    return sum(r.scored for r in new)


async def scrape(channel: Scannable, opted_out: set[int], progress: Progress | None = None) -> int:
    """Pull messages newer than the channel's cursor (oldest first), up to SCAN_LIMIT.

    Returns how many were queued for scoring (context-only messages are stored but not counted).
    """
    cursor = db.get_cursor(channel.id)
    after = discord.Object(cursor) if cursor else None
    log.info("reading #%s from %s, up to %d messages", channel.name, "the start" if cursor is None else cursor, SCAN_LIMIT)
    pending: list[Message] = []
    last_id = cursor or 0
    read = to_score = 0
    async for m in channel.history(limit=SCAN_LIMIT, after=after, oldest_first=True):
        last_id = m.id
        read += 1
        scored = classify(m, opted_out)
        if scored is not None:
            pending.append(to_row(m, scored, opted_out))
            to_score += scored
        if len(pending) >= SAVE_EVERY:
            db.save_batch(channel.id, last_id, pending)
            pending = []
        if progress and read % 100 == 0:
            await progress(f"📡 Reading #{channel.name}: {read:,} messages so far ({to_score:,} to analyse)…")
    if last_id:
        db.save_batch(channel.id, last_id, pending)
    log.info("read #%s: %d messages, %d queued for scoring", channel.name, read, to_score)
    return to_score


async def judge_backlog(guild: discord.Guild, progress=None, max_window: int | None = None) -> tuple[int, int]:
    """Run every queued message through DeepSeek. Returns (judged, flagged).

    A pool of CONCURRENCY workers each pulls the next batch as soon as it finishes its last one,
    so one slow call never holds the others up. Batches are one stretch of one channel; with
    max_window, a batch also spans at most that many messages (for scattered posts, so a call
    doesn't drag in everything between them as context).
    """
    rows = db.unjudged(guild.id)
    batches: asyncio.Queue[tuple[int, list[int]]] = asyncio.Queue()
    for batch in evaluate.group_batches([(r["channel_id"], r["id"]) for r in rows], db.count_between,
                                        BATCH_SIZE, max_window):
        batches.put_nowait(batch)
    total = len(rows)
    if not total:
        return 0, 0
    log.info("judging %d posts in %d batches with %d workers", total, batches.qsize(), CONCURRENCY)
    started = time.monotonic()
    judged = flagged = failed_in_a_row = 0
    dead = asyncio.Event()

    async def worker() -> None:
        nonlocal judged, flagged, failed_in_a_row
        while not dead.is_set():
            try:
                channel_id, ids = batches.get_nowait()
            except asyncio.QueueEmpty:
                return
            timeline = (db.timeline(channel_id, ids[0], ids[-1], before=CONTEXT_MESSAGES)
                        + db.timeline_after(channel_id, ids[-1], AFTER_MESSAGES))
            payload, order = build_payload(channel_info(guild, channel_id), timeline, ids)
            t0 = time.monotonic()
            quip = quip_allowed(guild.id)
            try:
                pictures = await batch_images(guild, channel_id, order)
                result, joke = await judge.judge_with_quip(payload, len(order), quip, anchors=prompt_extras(guild.id),
                                                           images=pictures)
            except Truncated:
                if len(ids) >= MIN_SPLIT * 2:  # too much to think about at once: retry as two halves
                    half = len(ids) // 2
                    log.warning("batch of %d ran out of tokens, retrying as %d + %d", len(ids), half, len(ids) - half)
                    batches.put_nowait((channel_id, ids[:half]))
                    batches.put_nowait((channel_id, ids[half:]))
                    continue
                log.error("batch of %d ran out of tokens even after splitting; leaving it queued", len(ids))
                failed = True
            except Refused as e:
                if len(ids) > 1:  # find the post it objects to: retry as two halves, down to single posts
                    half = len(ids) // 2
                    log.warning("batch of %d %s, retrying as %d + %d", len(ids), e, half, len(ids) - half)
                    batches.put_nowait((channel_id, ids[:half]))
                    batches.put_nowait((channel_id, ids[half:]))
                    continue
                # One post the model won't judge: file it as 0 so it isn't retried on every sweep.
                log.warning("post %s %s; filed as not judged (0)", ids[0], e)
                db.save_verdicts([(ids[0], 0, "declined by the model")], scoring.RUBRIC_VERSION)
                continue
            except Exception:
                log.exception("judge batch failed; leaving %d messages queued", len(order))
                failed = True
            else:
                failed = False
            if failed:
                failed_in_a_row += 1
                if failed_in_a_row >= MAX_FAILURES:  # API down or out of credit: stop instead of burning the queue
                    log.error("%d batches failed in a row, stopping this sweep", failed_in_a_row)
                    dead.set()
                continue
            failed_in_a_row = 0
            if joke:
                await post_quip(guild, channel_id, ids[-1], joke)
            db.save_verdicts([(mid, *result.get(i, Verdict(0, None))) for i, mid in enumerate(order)],
                             scoring.RUBRIC_VERSION)
            judged += len(order)
            hits = sum(v.severity > 0 for v in result.values())
            flagged += hits
            log.info("judged %d posts in %s: %d flagged (%.0fs)",
                     len(order), payload["channel"]["name"], hits, time.monotonic() - t0)
            if progress:
                rate = judged / (time.monotonic() - started)
                eta = (total - judged) / rate / 60 if rate else 0
                await progress(f"🕵️ Analysed {judged:,}/{total:,} posts, {flagged:,} flagged as kimoi, ~{eta:.0f} min left…")

    await asyncio.gather(*(worker() for _ in range(min(CONCURRENCY, batches.qsize()))))
    return judged, flagged


async def report(guild: discord.Guild) -> int:
    """Post newly flagged posts to the guild's report channel. Returns how many were posted."""
    channel = guild.get_channel(watching.get(guild.id, 0))
    if channel is None:
        return 0
    rows = db.unreported(guild.id, REPORT_MIN_SEVERITY)
    for r in rows[:REPORT_MAX_PER_RUN]:
        embed = discord.Embed(
            title=f"🚨 {r['severity']}/10 kimoi — {r['author_name']}",
            description=f">>> {clip(discord.utils.escape_markdown(r['content']), 1000)}",
            url=jump_url(guild.id, r["channel_id"], r["id"]),
            color=0xE91E63 if r["severity"] < 8 else 0x8B0000,
        )
        embed.add_field(name="Analyst note", value=clip(r["reason"] or "no comment", 200), inline=False)
        if profile := tags(r["labels"]):
            embed.add_field(name="Profile", value=profile, inline=False)
        embed.add_field(name="Location", value=f"<#{r['channel_id']}>", inline=False)
        await channel.send(embed=embed)
    if len(rows) > REPORT_MAX_PER_RUN:
        await channel.send(f"…and {len(rows) - REPORT_MAX_PER_RUN} more kimoi posts filed. See `{PREFIX}kimoiboard`.")
    db.mark_reported([r["id"] for r in rows])
    return len(rows)


async def process(guild: discord.Guild, progress=None) -> tuple[int, int, int]:
    """Judge the queue and report results. Caller must hold the guild lock."""
    if guild.id not in GUILD_IDS:
        raise PermissionError(f"refusing to call the model for unlisted guild {guild.id}")
    judged, flagged = await judge_backlog(guild, progress)
    posted = await report(guild)
    return judged, flagged, posted


async def process_in_background(guild: discord.Guild) -> None:
    lock = guild_locks[guild.id]
    if lock.locked():
        return
    async with lock:
        try:
            await process(guild)
        except Exception:
            log.exception("background sweep failed in %s", guild.id)


# --- live watching ----------------------------------------------------------

@bot.listen("on_message")
async def intercept(m: discord.Message):
    if m.guild is None or m.guild.id not in watching or not watchable(m.guild.id, m.channel.id):
        return
    opted_out = db.opted_out(m.guild.id)
    scored = classify(m, opted_out)
    if scored is None:
        return
    if spam_limiter.update_rate_limit(m):  # flooding can't buy extra API calls
        return
    db.save_message(to_row(m, scored, opted_out))
    if db.count_unjudged(m.guild.id) >= QUEUE_TRIGGER:
        task = asyncio.create_task(process_in_background(m.guild))
        background.add(task)
        task.add_done_callback(background.discard)


@tasks.loop(minutes=HEARTBEAT_MINUTES)
async def heartbeat():
    """Flush partial queues so quiet servers still get judged."""
    for guild_id in list(watching):
        guild = bot.get_guild(guild_id)
        if guild and db.count_unjudged(guild_id):
            await process_in_background(guild)


# --- respond: reply-and-ping the bot to get its take --------------------------

def pings_bot(m: discord.Message) -> bool:
    """An explicit @mention of the bot (or its role), not just Discord's reply ping."""
    me = m.guild.me
    if me is None:
        return False
    in_text = f"<@{me.id}>" in m.content or f"<@!{me.id}>" in m.content
    role = getattr(m.guild, "self_role", None)  # the bot's own managed role
    return in_text or bool(role and role in m.role_mentions)


def chat_item(m: discord.Message, limit: int = 300) -> dict:
    item = {"author": m.author.display_name, "time": message_time(m.id), "text": clip(plain_text(m), limit)}
    ref = m.reference.resolved if m.reference else None
    if isinstance(ref, discord.Message):
        item["reply_to"] = {"author": ref.author.display_name, "text": clip(plain_text(ref), 150)}
    if extras := describe_extras(m):
        item["attachments"] = clip(extras, 200)
    return item


def kimoi_file(guild_id: int, user_id: int) -> str | None:
    found = db.standing(guild_id, user_id)
    if not found:
        return None
    rank, s = found
    worst = db.worst_posts(guild_id, user_id, 2)
    lines = [f"kimoi rank #{rank}, {s.hits} kimoi posts out of {s.judged}, avg severity {s.avg_severity:.1f}/10"]
    lines += [f"worst: [{p['severity']}/10] {clip(p['content'], 120)}" for p in worst]
    return "; ".join(lines)


async def build_respond_payload(trigger: discord.Message, target: discord.Message, opted_out: set[int]) -> dict:
    history = [m async for m in trigger.channel.history(limit=RESPOND_CONTEXT, before=trigger)]
    conversation = [
        item for m in reversed(history)
        if m.author.id not in opted_out and not m.content.startswith(PREFIX)
        and ((item := chat_item(m))["text"] or "attachments" in item)
    ]
    me = trigger.guild.me
    request = plain_text(trigger)
    for name in {me.display_name, me.name}:
        request = request.replace(f"@{name}", "").strip()
    target_item = chat_item(target, 800)
    if file := kimoi_file(trigger.guild.id, target.author.id):
        target_item["kimoi_file"] = file
    return {
        "channel": channel_info(trigger.guild, trigger.channel.id),
        "conversation": conversation,
        "target": target_item,
        "request": {"author": trigger.author.display_name, "text": request or "(no request, just respond)"},
    }


@bot.listen("on_message")
async def respond_to_ping(m: discord.Message):
    """Someone replied to a message and pinged the bot: answer that message in context."""
    if RESPOND == "off" or m.guild is None or m.guild.id not in GUILD_IDS or m.author.bot:
        return
    if not m.reference or not m.reference.message_id or m.content.startswith(PREFIX) or not pings_bot(m):
        return
    if m.channel.id in IGNORE_CHANNEL_IDS:
        return
    if RESPOND == "admins" and m.author.id not in ADMIN_IDS:
        return
    target = m.reference.resolved
    if target is None or isinstance(target, discord.DeletedReferencedMessage):  # not sent along: fetch it
        try:
            target = await m.channel.fetch_message(m.reference.message_id)
        except discord.HTTPException:
            return
    if target.author.id == bot.user.id:
        return  # replies to the bot ping it automatically; only answer replies to other people
    opted_out = db.opted_out(m.guild.id)
    if m.author.id in opted_out or target.author.id in opted_out:
        return
    if not ai_allowed("respond", m.guild.id, m.author.id, RESPOND_PER_USER_HOUR):
        try:
            await m.add_reaction("⏳")
        except discord.HTTPException:
            pass
        return
    log.info("%s (%s) asked for a response to %s in #%s", m.author, m.author.id, target.id, getattr(m.channel, "name", "?"))
    try:
        async with m.channel.typing():
            payload = await build_respond_payload(m, target, opted_out)
            text = await judge.respond(payload, thinking=RESPOND_THINKING)
    except Exception:
        log.exception("respond failed for %s", m.id)
        return
    if not text:
        return
    try:
        await target.reply(clip(text, 1900), mention_author=False)
    except discord.HTTPException:  # target vanished: answer the ping instead
        await m.reply(clip(text, 1900), mention_author=False)


# --- VAR: replay self-deleted posts -----------------------------------------

def var_allowed(guild_id: int, user_id: int) -> bool:
    """At most VAR_PER_USER_HOUR paid reviews per person per hour, so post-and-delete spam can't burn credit."""
    calls, now = var_calls[(guild_id, user_id)], time.monotonic()
    while calls and now - calls[0] > 3600:
        calls.popleft()
    if len(calls) >= VAR_PER_USER_HOUR:
        return False
    calls.append(now)
    return True


audit_counts: dict[int, dict[int, int]] = {}  # guild -> {message-delete audit entry id: count}
unclaimed_mod_deletes: defaultdict[int, defaultdict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
audit_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
warned_no_audit: set[int] = set()
MOD_DELETE_TTL = 120  # seconds an unclaimed mod deletion stays attributable


async def refresh_mod_deletes(guild: discord.Guild, prime: bool = False) -> None:
    """Diff the recent message-delete audit entries against the last look.

    Self-deletes never appear in the audit log; a mod deleting someone else's message does. Discord
    merges repeated deletes (same mod, same author, same channel) into one entry and bumps its
    count, so a deletion shows up either as a new entry or as a higher count on an old one. Each one
    is recorded as an unclaimed (author, channel) mod deletion for the next matching delete event.
    """
    async with audit_locks[guild.id]:
        entries = [e async for e in guild.audit_logs(limit=50, action=discord.AuditLogAction.message_delete)]
        seen, now = audit_counts.get(guild.id), time.monotonic()
        for e in entries if not prime else []:
            count = getattr(e.extra, "count", None) or 1
            if seen is None:  # first look since startup: only trust entries created just now
                new = count if (discord.utils.utcnow() - e.created_at) < timedelta(minutes=2) else 0
            else:
                new = count - seen.get(e.id, 0)
            key = (getattr(e.target, "id", None), getattr(getattr(e.extra, "channel", None), "id", None))
            unclaimed_mod_deletes[guild.id][key] += [now] * max(0, new)
        audit_counts[guild.id] = {e.id: getattr(e.extra, "count", None) or 1 for e in entries}
        for key, times in list(unclaimed_mod_deletes[guild.id].items()):
            times[:] = [t for t in times if now - t < MOD_DELETE_TTL]
            if not times:
                del unclaimed_mod_deletes[guild.id][key]


def claim_mod_delete(guild_id: int, author_id: int, channel_id: int) -> bool:
    times = unclaimed_mod_deletes[guild_id].get((author_id, channel_id))
    if times:
        times.pop(0)
        return True
    return False


async def deleted_by_mod(guild: discord.Guild, channel_id: int, author_id: int) -> bool | None:
    """True if a mod (or another bot) removed it, False if the author did, None if we can't tell."""
    if not guild.me.guild_permissions.view_audit_log:
        if guild.id not in warned_no_audit:
            warned_no_audit.add(guild.id)
            log.warning("no View Audit Log in %s: VAR can't spot mod deletes, so it reviews every delete", guild.id)
        return None
    try:
        for wait in (2, 3):  # the audit entry lands a moment after the delete event
            await asyncio.sleep(wait)
            await refresh_mod_deletes(guild)
            if claim_mod_delete(guild.id, author_id, channel_id):
                return True
    except discord.HTTPException:
        return None
    return False


async def post_var(guild: discord.Guild, row, severity: int, reason: str | None) -> None:
    channel = guild.get_channel(watching.get(guild.id, 0))
    if channel is None:
        return
    before = db.timeline(row["channel_id"], row["id"], row["id"], before=1)
    scene = (f"[just before it]({jump_url(guild.id, row['channel_id'], before[0]['id'])}) in "
             if before and before[0]["id"] != row["id"] else "")
    embed = discord.Embed(
        title=f"📺 VAR REVIEW — {row['author_name']} deleted a post",
        description=f">>> {clip(discord.utils.escape_markdown(row['content']), 1000)}",
        color=0x8B0000 if severity >= 8 else 0xE91E63,
    )
    embed.add_field(name="Decision", value=f"**{severity}/10 kimoi.** Deletion overturned. The post stands.", inline=False)
    embed.add_field(name="Analyst note", value=clip(reason or "no comment", 200), inline=False)
    if profile := tags(row["labels"]):
        embed.add_field(name="Profile", value=profile, inline=False)
    embed.add_field(name="Scene", value=f"{scene}<#{row['channel_id']}>", inline=False)
    await channel.send(embed=embed)


@bot.listen("on_raw_message_delete")
async def var_review(event: discord.RawMessageDeleteEvent):
    """Someone deleted their own message: replay it with context and air it if it was kimoi."""
    gid, mid, cid = event.guild_id, event.message_id, event.channel_id
    if not VAR or gid is None or gid not in GUILD_IDS or gid not in watching or not watchable(gid, cid):
        return
    guild = bot.get_guild(gid)
    if guild is None:
        return
    opted_out = db.opted_out(gid)
    row, cached = db.get_message(mid), event.cached_message
    if cached is not None:
        if classify(cached, opted_out) is not True:  # bots, commands, "lol", opted-out users
            return
        author_id = cached.author.id
    elif row is not None and row["scored"] and row["author_id"] not in opted_out:
        author_id = row["author_id"]
    else:
        return  # never saw it, or not worth reviewing
    if await deleted_by_mod(guild, cid, author_id):  # None (can't tell) is reviewed as a self-delete
        log.info("VAR: message %s was removed by a mod, not reviewing", mid)
        return
    if row is None:  # only in Discord's cache: store it so it has context and a place on the board
        db.save_message(to_row(cached, True, opted_out))
        row = db.get_message(mid)
    db.mark_deleted(mid)
    if row["reported"]:
        return  # already aired
    severity, reason = row["severity"], row["reason"]
    if severity is None:  # not judged yet: replay it now
        if not var_allowed(gid, author_id):
            log.info("VAR: %s hit the review limit, skipping %s", author_id, mid)
            return
        timeline = db.timeline(cid, mid, mid, before=CONTEXT_MESSAGES) + db.timeline_after(cid, mid, VAR_AFTER)
        payload, _ = build_payload(channel_info(guild, cid), timeline, [mid])
        try:
            pictures = await batch_images(guild, cid, [mid])
            result, _ = await judge.judge_with_quip(payload, 1, quip=False, note=VAR_NOTE, anchors=prompt_extras(gid),
                                                    images=pictures)
        except Exception:
            log.exception("VAR review of %s failed", mid)
            return
        verdict = result.get(0, Verdict(0, None))
        severity, reason = verdict.severity, verdict.reason
        db.save_verdicts([(mid, *verdict)], scoring.RUBRIC_VERSION)
        row = db.get_message(mid)
    log.info("VAR: %s deleted %s, %d/10", row["author_name"], mid, severity)
    if severity >= VAR_MIN_SEVERITY:
        try:
            await post_var(guild, row, severity, reason)
        except discord.HTTPException:
            log.warning("couldn't post VAR review of %s", mid)
            return
        db.mark_reported([mid])


# --- commands ---------------------------------------------------------------

async def run_scan(ctx: commands.Context, channels: list[discord.abc.Messageable], recent: int | None = None) -> None:
    lock = guild_locks[ctx.guild.id]
    if lock.locked():
        await ctx.send("A surveillance sweep is already running in this server.")
        return
    async with lock:
        if ctx.interaction:
            await ctx.send("📡 Sweep started. Progress below.", ephemeral=True)
        status = await ctx.channel.send(f"📡 Intercepting {len(channels)} channel(s)…")
        progress = Progress(status)
        opted_out = db.opted_out(ctx.guild.id)
        scraped = 0
        for ch in channels:
            try:
                scraped += await (scrape_recent(ch, opted_out, recent) if recent else scrape(ch, opted_out, progress))
            except discord.Forbidden:
                log.warning("no permission to read #%s", ch.name)
                await ctx.send(f"No clearance for {ch.mention}, skipping.")
        await progress(f"📡 Intercepted {scraped:,} new posts. Handing them to the analyst…", force=True)
        judged, flagged, posted = await process(ctx.guild, progress)
        left = db.count_unjudged(ctx.guild.id)
        notes = []
        if left:
            notes.append(f"{left} posts failed (API errors), rerun to retry.")
        if ctx.guild.id not in watching:
            notes.append(f"No report channel set (`{PREFIX}watch #channel`), so nothing was posted.")
        log.info("sweep complete: %d queued, %d judged, %d flagged, %d posted, %d left", scraped, judged, flagged, posted, left)
        await status.edit(
            content=f"✅ Sweep complete: {scraped} new posts intercepted, {judged} analysed, {flagged} kimoi "
            f"(all go on the leaderboard), {posted} at ≥{REPORT_MIN_SEVERITY}/10 posted to the report channel. " + " ".join(notes)
        )


@bot.hybrid_command(help="Scan a channel's history for kimoi posts (default: this one). Voice chats and threads work too.")
@app_commands.describe(channel="Channel, voice chat or thread to scan (default: this one)",
                       recent="Only the latest N messages (default: everything new since the last scan)")
@deployer_only()
async def scan(ctx: commands.Context, channel: Scannable = None, recent: commands.Range[int, 1, 10000] = None):
    target = channel or ctx.channel
    if not watchable(ctx.guild.id, target.id):
        await ctx.send("That channel is off-limits (the report channel or an ignored channel).", ephemeral=True)
        return
    await run_scan(ctx, [target], recent)


@bot.hybrid_command(help="Scan every text channel and voice/stage chat the bot can read.")
@app_commands.describe(recent="Only the latest N messages of each channel (default: everything new since the last scan)")
@deployer_only()
async def scanall(ctx: commands.Context, recent: commands.Range[int, 1, 10000] = None):
    me = ctx.guild.me
    channels = [
        c for c in [*ctx.guild.text_channels, *ctx.guild.voice_channels, *ctx.guild.stage_channels]
        if c.permissions_for(me).read_message_history and watchable(ctx.guild.id, c.id)
    ]
    await run_scan(ctx, channels, recent)


@bot.hybrid_command(help="Start live surveillance, posting kimoi to the given report channel.")
@app_commands.describe(report_channel="Where kimoi reports, VAR reviews and quips about old chat go")
@deployer_only()
async def watch(ctx: commands.Context, report_channel: discord.TextChannel):
    perms = report_channel.permissions_for(ctx.guild.me)
    if not (perms.send_messages and perms.embed_links):
        await ctx.send(f"I can't post embeds in {report_channel.mention}.")
        return
    db.set_watch(ctx.guild.id, report_channel.id)
    watching[ctx.guild.id] = report_channel.id
    await ctx.send(
        f"👁️ Live surveillance on. Judging every {QUEUE_TRIGGER} posts (or every {HEARTBEAT_MINUTES:g} min), "
        f"reporting severity ≥ {REPORT_MIN_SEVERITY} to {report_channel.mention}."
    )


@bot.hybrid_command(help="Stop live surveillance.")
@deployer_only()
async def unwatch(ctx: commands.Context):
    db.set_watch(ctx.guild.id, None)
    watching.pop(ctx.guild.id, None)
    await ctx.send("Live surveillance off. Queued posts will be judged on the next scan.")


@bot.hybrid_command(help="Show model token usage and the queue.")
@deployer_only()
async def usage(ctx: commands.Context):
    await ctx.send(
        f"💸 {db.tokens_today():,} {judge.model} tokens used today (UTC). "
        f"{db.count_unjudged(ctx.guild.id)} posts queued. "
        f"Live watch: {'on' if ctx.guild.id in watching else 'off'}."
    )


# --- model / scoring info ---------------------------------------------------

def on_off(flag: bool) -> str:
    return "on" if flag else "off"


@bot.hybrid_command(help="Which model the bot uses (Claude or DeepSeek), and how.")
async def model(ctx: commands.Context):
    effort = judge.effort or ("default (medium)" if judge.provider == "anthropic" else "default (high)")
    embed = discord.Embed(title="🧠 Analyst hardware", color=0x5865F2)
    embed.add_field(name="Model", value=f"`{judge.model}` via `{judge.client.base_url.host}` ({judge.provider})",
                    inline=False)
    embed.add_field(
        name="Scoring posts",
        value=f"thinking {on_off(judge.thinking)}"
        + (f" · effort {effort} · up to {judge.thinking_tokens:,} thinking tokens" if judge.thinking else "")
        + f"\n{BATCH_SIZE} posts per call · {CONCURRENCY} calls at once · {CONTEXT_MESSAGES} messages of context",
        inline=False,
    )
    embed.add_field(name="Chat replies", value=f"thinking {on_off(RESPOND_THINKING)} · {RESPOND_CONTEXT} messages of context",
                    inline=False)
    embed.add_field(name="Roasts and dossiers", value=f"thinking {on_off(judge.thinking)}", inline=False)
    await ctx.send(embed=embed)


def calibration_stats(guild_id: int) -> dict:
    """How far the formula lands from the admins' own scores (current rubric only)."""
    diffs, outdated = [], 0
    for r in db.gold_rows(guild_id):
        if r["rubric_version"] != scoring.RUBRIC_VERSION:
            outdated += 1
            continue
        labels = json.loads(r["labels"]) if r["labels"] else None
        bot_score = scoring.score(labels) if labels else 0
        diffs.append((bot_score - r["admin_score"], r, bot_score, labels))
    n = len(diffs)
    return {
        "n": n,
        "outdated": outdated,
        "mae": sum(abs(d) for d, *_ in diffs) / n if n else 0.0,
        "bias": sum(d for d, *_ in diffs) / n if n else 0.0,
        "within1": sum(abs(d) <= 1 for d, *_ in diffs) / n if n else 0.0,
        "worst": sorted(diffs, key=lambda x: -abs(x[0]))[:5],
    }


@bot.hybrid_command(name="scoring", help="How posts are scored: the formula behind every kimoi score.")
async def scoring_(ctx: commands.Context):
    embed = discord.Embed(
        title=f"📐 Kimoi scoring (rubric v{scoring.RUBRIC_VERSION})",
        description="The model labels each kimoi post; this formula turns the labels into a score.\n\n"
        + "\n".join(scoring.formula_lines()),
        color=0x5865F2,
    )
    stats = calibration_stats(ctx.guild.id)
    if stats["n"]:
        embed.add_field(
            name="Calibration",
            value=f"{stats['n']} admin-scored posts · off by {stats['mae']:.1f} on average · "
            f"{stats['within1']:.0%} within 1 point",
            inline=False,
        )
    embed.set_footer(text=f"Report threshold {REPORT_MIN_SEVERITY}/10 · VAR threshold {VAR_MIN_SEVERITY}/10")
    await ctx.send(embed=embed)



# --- calibration (admins, in DMs) --------------------------------------------

class CalibrationView(discord.ui.View):
    """0-10 buttons for one post. Answering saves the admin's score, reveals the bot's and moves on."""

    def __init__(self, admin_id: int, guild_id: int, row, done: int):
        super().__init__(timeout=900)
        self.admin_id, self.guild_id, self.row, self.done = admin_id, guild_id, row, done
        for n in range(11):
            button = discord.ui.Button(label=str(n), row=0 if n < 5 else 1 if n < 10 else 2,
                                       style=discord.ButtonStyle.danger if n >= 7 else discord.ButtonStyle.secondary)
            button.callback = self._scorer(n)
            self.add_item(button)
        for label, handler in (("Skip", self._skip), ("Stop", self._stop)):
            button = discord.ui.Button(label=label, row=2, style=discord.ButtonStyle.primary)
            button.callback = handler
            self.add_item(button)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.admin_id

    def _close(self) -> None:
        for item in self.children:
            item.disabled = True
        self.stop()

    def _scorer(self, n: int):
        async def callback(interaction: discord.Interaction):
            db.save_gold(self.guild_id, self.row["id"], self.admin_id, n)
            anchor_cache.pop(self.guild_id, None)
            labels = json.loads(self.row["labels"]) if self.row["labels"] else None
            bot_score = scoring.score(labels) if labels else (self.row["severity"] or 0)
            embed = interaction.message.embeds[0]
            embed.add_field(name="You", value=f"**{n}**/10", inline=True)
            embed.add_field(name="Bot", value=f"**{bot_score}**/10" + (f" · {scoring.describe(labels)}" if labels else ""),
                            inline=True)
            self._close()
            await interaction.response.edit_message(embed=embed, view=self)
            await send_calibration(interaction.channel, self.admin_id, self.guild_id, self.done + 1)
        return callback

    async def _skip(self, interaction: discord.Interaction):
        self._close()
        await interaction.response.edit_message(content="Skipped.", view=self)
        await send_calibration(interaction.channel, self.admin_id, self.guild_id, self.done)

    async def _stop(self, interaction: discord.Interaction):
        self._close()
        await interaction.response.edit_message(
            content=f"Session over: {self.done} scored. `/calibration` shows how the formula compares.", view=self)


async def send_calibration(channel, admin_id: int, guild_id: int, done: int) -> None:
    """Send the next post to score: alternately one the bot flagged and a random one."""
    row = db.calibration_sample(guild_id, flagged=done % 2 == 0) or db.calibration_sample(guild_id, flagged=False)
    if row is None:
        await channel.send(f"Nothing left to score. {done} done this session. `/calibration` for the report.")
        return
    before = [r for r in db.timeline(row["channel_id"], row["id"], row["id"], before=3) if r["id"] != row["id"]]
    context = "\n".join(f"-# {clip(r['author_name'], 30)}: {clip(r['content'], 120)}" for r in before)
    embed = discord.Embed(
        title=f"Calibration #{done + 1}: how kimoi is this, 0-10?",
        description=(context + "\n" if context else "")
        + f"**{discord.utils.escape_markdown(row['author_name'])}**: {clip(discord.utils.escape_markdown(row['content']), 900)}",
        url=jump_url(guild_id, row["channel_id"], row["id"]),
        color=0x5865F2,
    )
    embed.set_footer(text="0 = normal fandom · 10 = call the actual NSA · the bot's score is shown after you answer")
    await channel.send(embed=embed, view=CalibrationView(admin_id, guild_id, row, done))


@bot.hybrid_command(help="Score posts yourself in DMs, so the bot can be checked and tuned against you.")
@deployer_only()
async def calibrate(ctx: commands.Context):
    try:
        dm = await ctx.author.create_dm()
        await dm.send(f"🎯 Calibration for **{ctx.guild.name}**. Score each post 0-10 by your own gut; "
                      "Stop whenever you like. Your scores tune the bot's examples and power `/calibration`.")
        await send_calibration(dm, ctx.author.id, ctx.guild.id, 0)
    except discord.Forbidden:
        await ctx.send("I can't DM you. Allow DMs from server members and try again.", ephemeral=True)
        return
    await ctx.send("📬 Check your DMs.", ephemeral=True)


@bot.hybrid_command(help="How the scoring formula compares with the admins' own scores (sent to your DMs).")
@deployer_only()
async def calibration(ctx: commands.Context):
    stats = calibration_stats(ctx.guild.id)
    if not stats["n"]:
        note = f" ({stats['outdated']} are from an older rubric; run `/rescore`)" if stats["outdated"] else ""
        await ctx.send(f"No admin scores to compare yet{note}. Start with `/calibrate`.", ephemeral=True)
        return
    direction = "too harsh" if stats["bias"] > 0.25 else "too soft" if stats["bias"] < -0.25 else "about right"
    embed = discord.Embed(
        title="🎯 Calibration report",
        description=f"**{stats['n']}** admin-scored posts\n"
        f"Off by **{stats['mae']:.1f}** points on average · **{stats['within1']:.0%}** within 1 point\n"
        f"Overall the bot is **{direction}** ({stats['bias']:+.1f})"
        + (f"\n{stats['outdated']} more were scored under an older rubric (`/rescore` to include them)"
           if stats["outdated"] else ""),
        color=0x5865F2,
    )
    for diff, r, bot_score, labels in stats["worst"]:
        if diff == 0:
            break
        embed.add_field(
            name=f"You {r['admin_score']} · bot {bot_score}" + (f" · {scoring.describe(labels)}" if labels else " · not flagged"),
            value=f"[{clip(discord.utils.escape_markdown(r['content']), 150)}]({jump_url(ctx.guild.id, r['channel_id'], r['id'])})",
            inline=False,
        )
    try:
        await ctx.author.send(embed=embed)
    except discord.Forbidden:
        await ctx.send("I can't DM you. Allow DMs from server members and try again.", ephemeral=True)
        return
    await ctx.send("📬 Report sent to your DMs.", ephemeral=True)


@bot.hybrid_command(help="Re-judge posts with the current prompt: everything on an older rubric, or only flagged posts.")
@app_commands.describe(scope="all: every post on an older rubric · flagged: only flagged posts, much cheaper")
@deployer_only()
async def rescore(ctx: commands.Context, scope: Literal["all", "flagged"] = "all"):
    flagged_only = scope == "flagged"
    pending = db.count_unjudged(ctx.guild.id)  # left over from a rescore cut short: finish those first
    if flagged_only:
        outdated = pending or db.count_flagged(ctx.guild.id)
        if not outdated:
            await ctx.send("Nothing is flagged yet.", ephemeral=True)
            return
    else:
        outdated = pending + db.count_outdated(ctx.guild.id, scoring.RUBRIC_VERSION)
        if not outdated:
            await ctx.send(f"Everything is already on rubric v{scoring.RUBRIC_VERSION}. "
                           "`/rescore scope:flagged` re-checks the flagged posts anyway.", ephemeral=True)
            return
    lock = guild_locks[ctx.guild.id]
    if lock.locked():
        await ctx.send("A sweep is already running in this server.", ephemeral=True)
        return
    async with lock:
        if ctx.interaction:
            await ctx.send("🔁 Rescore started. Progress below.", ephemeral=True)
        what = ("leftover posts" if pending else "flagged posts") if flagged_only else "posts"
        status = await ctx.channel.send(f"🔁 Re-scoring {outdated:,} {what} under rubric v{scoring.RUBRIC_VERSION}…")
        if flagged_only and not pending:
            db.queue_rescore_flagged(ctx.guild.id)
        else:
            db.queue_rescore(ctx.guild.id, scoring.RUBRIC_VERSION)
        judged, flagged = await judge_backlog(ctx.guild, Progress(status),
                                              evaluate.MAX_WINDOW if flagged_only else None)
        left = db.count_unjudged(ctx.guild.id)
        log.info("rescore complete: %d judged, %d flagged, %d left", judged, flagged, left)
        await status.edit(content=f"✅ Rescore done: {judged:,} posts re-scored, {flagged:,} kimoi."
                          + (f" {left:,} still queued (API errors), run the same `/rescore` again to finish them."
                             if left else "")
                          + " Old history isn't re-posted to the report channel.")


# --- prompt tuning: evaluation and server notes (admins) ----------------------

async def run_evaluation(guild: discord.Guild, progress=None, limit: int | None = None) -> dict:
    """Re-judge the reviewed posts with the current prompt (nothing is saved) and score the result."""
    rows = evaluate.load_review_set()
    posts, wanted = [], {}
    for r in rows:
        stored = db.get_message(int(r["id"]))
        if stored and stored["guild_id"] == guild.id:
            posts.append((stored["channel_id"], stored["id"]))
            wanted[stored["id"]] = r
    if limit:
        posts = posts[:limit]
    batches = evaluate.group_batches(posts, db.count_between, BATCH_SIZE)
    new: dict[int, int] = {}
    verdicts: dict[int, Verdict] = {}
    sem, done = asyncio.Semaphore(CONCURRENCY), 0
    PARSE_STATS.clear()
    tokens_before = db.tokens_today()
    extras = prompt_extras(guild.id)

    failed: Counter = Counter()  # posts left out of the report, by cause

    async def one(channel_id: int, ids: list[int]) -> None:
        nonlocal done
        timeline = (db.timeline(channel_id, ids[0], ids[-1], before=CONTEXT_MESSAGES)
                    + db.timeline_after(channel_id, ids[-1], AFTER_MESSAGES))
        payload, order = build_payload(channel_info(guild, channel_id), timeline, ids)
        result, refused, cause = None, False, None
        async with sem:
            pictures = await batch_images(guild, channel_id, order)
            for attempt in range(3):
                try:
                    result, _ = await judge.judge_with_quip(payload, len(order), quip=False, anchors=extras,
                                                            images=pictures)
                    break
                except Refused as e:
                    log.warning("evaluation: batch of %d %s", len(ids), e)
                    refused = True
                    break
                except Exception as e:
                    log.exception("evaluation batch failed (attempt %d)", attempt + 1)
                    cause = type(e).__name__
        if refused and len(ids) > 1:  # narrow it down to the post it objects to, like a sweep does
            half = len(ids) // 2
            await asyncio.gather(one(channel_id, ids[:half]), one(channel_id, ids[half:]))
            return
        if refused:  # the one post the model won't judge: scored 0, as a sweep would file it
            result = {}
            failed["declined by the model (scored 0)"] += 1
        elif result is None:
            failed[cause] += len(order)
            return
        for i, mid in enumerate(order):
            verdicts[mid] = result.get(i, Verdict(0, None))
            new[mid] = verdicts[mid].severity
        done += len(order)
        if progress:
            await progress(f"🧪 Evaluating: {done}/{len(posts)} posts…")

    await asyncio.gather(*(one(c, ids) for c, ids in batches))
    report = evaluate.metrics([wanted[m] for _, m in posts], new)
    report.update(calls=len(batches), tokens=db.tokens_today() - tokens_before, stats=dict(PARSE_STATS),
                  failed=dict(failed), asked=len(posts),
                  details=evaluation_details(guild.id, [(m, wanted[m]) for _, m in posts], verdicts))
    log.info("evaluation: %s", {k: v for k, v in report.items() if k not in ("worst", "details")})
    return report


def evaluation_details(guild_id: int, rows: list[tuple[int, dict]], verdicts: dict[int, Verdict]) -> str:
    """Every post's review score next to the new verdict, misses first, for an admin (or agent) to read."""
    lines = []
    for mid, r in sorted(rows, key=lambda x: -abs(verdicts[x[0]].severity - x[1]["should"]) if x[0] in verdicts else 0):
        v, m = verdicts.get(mid), db.get_message(mid)
        if v is None or m is None:
            continue
        labels = v.labels or {}
        lines.append(" | ".join([
            f"review {r['should']} · was {r['bot']} · now {v.severity}" + (f" · {r['safety']}" if r.get("safety") else ""),
            f"{m['author_name']}: {clip(m['content'].replace(chr(10), ' '), 200)}",
            "distress" if labels.get("distress") else (scoring.describe(labels) or "not flagged"),
            f"quote: {labels['evidence']}" if labels.get("evidence") else "",
            v.reason or "",
            jump_url(guild_id, m["channel_id"], mid),
        ]))
    return "\n".join(lines)


def evaluation_embed(guild_id: int, r: dict) -> discord.Embed:
    b, a = r["before"], r["after"]

    def row(label, key, fmt="{}", total=None):
        tot = f"/{a[total]}" if total else ""
        return f"**{label}:** {fmt.format(b[key])}{tot} → **{fmt.format(a[key])}**{tot}"

    embed = discord.Embed(
        title=f"🧪 Evaluation, rubric v{scoring.RUBRIC_VERSION}",
        description="\n".join([
            f"{r['n']} reviewed posts, re-judged with the current prompt (nothing saved). Before → after:",
            row("Average gap", "avg_gap", "{:.2f}"),
            row("Within 1 point", "within1", "{:.0%}"),
            row("False flags", "false_flags", total="clean_total"),
            row("Safety posts still flagged", "safety_hits", total="safety_total"),
            row("Scored 10", "tens"),
            f"**Score spread now:** `{' '.join(str(x) for x in a['dist'])}` (0→10)",
            f"Dropped for no quote: {r['stats'].get('no_evidence', 0)} · distress: {r['stats'].get('distress', 0)} · "
            f"{r['calls']} calls · {r['tokens']:,} tokens",
            *([f"⚠️ **{r['asked'] - r['n']} of {r['asked']} posts left out** (the API kept failing): "
               + ", ".join(f"{k} ×{v}" for k, v in r["failed"].items() if "declined" not in k)]
              if r.get("asked", r["n"]) > r["n"] else []),
            *([f"Declined by the model: {r['failed'][k]}" for k in r.get("failed", {}) if "declined" in k]),
        ]),
        color=0x5865F2,
    )
    for mid, should, got, safety in r["worst"]:
        m = db.get_message(mid)
        if m is None or got == should:
            continue
        embed.add_field(
            name=f"Review {should} · now {got}" + (f" · {safety}" if safety else ""),
            value=f"[{clip(discord.utils.escape_markdown(m['author_name'] + ': ' + m['content']), 140)}]"
            f"({jump_url(guild_id, m['channel_id'], mid)})",
            inline=False,
        )
    return embed


@bot.hybrid_command(name="evaluate", help="Test the current prompt on the reviewed posts (calls the model, nothing saved; report in DMs).")
@app_commands.describe(limit="Only the first N posts, for a quick cheap check (default: all)")
@deployer_only()
async def evaluate_(ctx: commands.Context, limit: int | None = None):
    lock = guild_locks[ctx.guild.id]
    if lock.locked():
        await ctx.send("A sweep is already running in this server.", ephemeral=True)
        return
    try:
        dm = await ctx.author.create_dm()
        status = await dm.send("🧪 Starting evaluation…")
    except discord.Forbidden:
        await ctx.send("I can't DM you. Allow DMs from server members and try again.", ephemeral=True)
        return
    await ctx.send("🧪 Evaluation started. Results will arrive in your DMs.", ephemeral=True)
    async with lock:
        report = await run_evaluation(ctx.guild, Progress(status), limit)
    if not report.get("n"):
        await status.edit(content="None of the reviewed posts are in this server's database.")
        return
    await status.edit(content="🧪 Evaluation done.")
    await dm.send(embed=evaluation_embed(ctx.guild.id, report),
                  file=discord.File(io.BytesIO(report["details"].encode()), filename="evaluation.txt"))



@bot.hybrid_command(help="Show the judge's server notes, or replace them by attaching a .md/.txt file.")
@app_commands.describe(file="A text file to replace the notes with (leave empty to get the current notes)")
@deployer_only()
async def notes(ctx: commands.Context, file: discord.Attachment | None = None):
    if file is None:
        current = server_notes()
        if not current:
            await ctx.send("No server notes yet. Attach a .md or .txt file to `/notes` to add them.", ephemeral=True)
            return
        try:
            await ctx.author.send("Current server notes:",
                                  file=discord.File(io.BytesIO(current.encode()), filename="server_notes.md"))
        except discord.Forbidden:
            await ctx.send("I can't DM you. Allow DMs from server members and try again.", ephemeral=True)
            return
        await ctx.send("📬 Sent to your DMs.", ephemeral=True)
        return
    if file.size > SERVER_NOTES_MAX * 4:
        await ctx.send(f"That file is too big (keep it under {SERVER_NOTES_MAX:,} characters).", ephemeral=True)
        return
    try:
        text = (await file.read()).decode("utf-8")
    except UnicodeDecodeError:
        await ctx.send("That doesn't look like a UTF-8 text file.", ephemeral=True)
        return
    if len(text) > SERVER_NOTES_MAX:
        await ctx.send(f"That's {len(text):,} characters; keep it under {SERVER_NOTES_MAX:,}.", ephemeral=True)
        return
    os.makedirs(os.path.dirname(SERVER_NOTES_PATH) or ".", exist_ok=True)
    with open(SERVER_NOTES_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("%s (%s) replaced the server notes (%d chars)", ctx.author, ctx.author.id, len(text))
    await ctx.send(f"📝 Server notes updated ({len(text):,} characters). They apply to the next judging call; "
                   "run `/evaluate` to see the effect.", ephemeral=True)


@bot.hybrid_command(help="Have the analyst write up a classified dossier on someone's kimoi record.")
@app_commands.describe(member="Who to investigate (leave empty for yourself)")
@public_ai("DOSSIER_PUBLIC")
@commands.cooldown(1, 30, commands.BucketType.guild)
async def dossier(ctx: commands.Context, member: Suspect = None):
    member = member or ctx.author
    found = db.standing(ctx.guild.id, member.id)
    if not found:
        await ctx.send(f"Insufficient evidence on {member.display_name}.")
        return
    if not ai_allowed("dossier", ctx.guild.id, ctx.author.id, DOSSIER_PER_USER_HOUR):
        await ctx.send(f"Dossier limit reached ({DOSSIER_PER_USER_HOUR}/hour). Records office is closed.", ephemeral=True)
        return
    rank, s = found
    posts = [(p["severity"], p["content"], p["reason"] or "") for p in db.worst_posts(ctx.guild.id, member.id, 8)]
    stats = f"rank #{rank}, {s.hits} kimoi posts out of {s.judged}, avg severity {s.avg_severity:.1f}/10"
    async with ctx.typing():
        text = await judge.roast(member.display_name, stats, posts)
    await ctx.send(f"**CLASSIFIED — {member.display_name}**\n{clip(text, 1900) or 'File redacted.'}")


@bot.hybrid_command(help="Get the analyst to roast someone (or yourself) based on what they post.")
@app_commands.describe(member="Who to roast (leave empty to roast yourself)")
@public_ai("ROAST_PUBLIC")
@commands.cooldown(1, 20, commands.BucketType.channel)
async def roast(ctx: commands.Context, member: Suspect = None):
    member = member or ctx.author
    recent = [r["content"] for r in db.recent_posts(ctx.guild.id, member.id)]
    if not recent:
        await ctx.send(f"No intel on {member.display_name}. Can't roast a ghost.")
        return
    if not ai_allowed("roast", ctx.guild.id, ctx.author.id, ROAST_PER_USER_HOUR):
        await ctx.send(f"Roast limit reached ({ROAST_PER_USER_HOUR}/hour). The analyst needs water.", ephemeral=True)
        return
    found = db.standing(ctx.guild.id, member.id)
    if found:
        rank, s = found
        stats = f"kimoi rank #{rank}, {s.hits} kimoi posts out of {s.judged}, avg severity {s.avg_severity:.1f}/10"
    else:
        stats = "no kimoi posts on file (suspiciously clean)"
    worst = [(p["severity"], p["content"], p["reason"] or "") for p in db.worst_posts(ctx.guild.id, member.id, 6)]
    async with ctx.typing():
        text = await judge.burn(member.display_name, stats, worst, recent)
    if not text:
        await ctx.send("The analyst opened their mouth and nothing came out. Try again.")
        return
    embed = discord.Embed(title=f"🔥 ROAST: {member.display_name}", description=clip(text, 1500), color=0xFF4500)
    embed.set_footer(text="Neckbeard Surveillance Agency · comedy division")
    await ctx.send(embed=embed)


@bot.hybrid_command(aliases=["kimoirank"], help="The kimoi leaderboard.")
@commands.cooldown(1, 10, commands.BucketType.channel)
async def kimoiboard(ctx: commands.Context):
    rows = db.leaderboard(ctx.guild.id)
    if not rows:
        await ctx.send("No kimoi on file yet (or this server is suspiciously clean).")
        return
    medals = ["🥇", "🥈", "🥉"]
    lines = []
    for i, s in enumerate(rows):
        rank = medals[i] if i < len(medals) else f"`#{i + 1}`"
        lines.append(
            f"{rank} **{discord.utils.escape_markdown(s.author_name)}** — {s.points:.1f} pts · "
            f"{s.hits} kimoi posts ({s.rate:.1%}) · avg {s.avg_severity:.1f}/10"
        )
    embed = discord.Embed(title="🚨 NSA Kimoi Watchlist", description="\n".join(lines), color=0xE91E63)
    embed.set_footer(text="Points = Σ severity² / 10. Frequency counts, but severity counts more.")
    await ctx.send(embed=embed)


POSTS_PER_PAGE = 10


def kimoi_page(guild_id: int, user: discord.Member | None, page: int) -> tuple[discord.Embed, int]:
    """One page of the kimoi archive. Returns the embed and the number of pages."""
    total, rows = db.kimoi_posts(guild_id, user.id if user else None, page * POSTS_PER_PAGE, POSTS_PER_PAGE)
    pages = max(1, -(-total // POSTS_PER_PAGE))
    title = f"🗄️ Kimoi archive: {user.display_name}" if user else "🗄️ Kimoi archive"
    lines = []
    for n, r in enumerate(rows, start=page * POSTS_PER_PAGE + 1):
        where = "📺 deleted, caught by VAR" if r["deleted"] else f"[jump]({jump_url(guild_id, r['channel_id'], r['id'])})"
        who = "" if user else f" · **{discord.utils.escape_markdown(r['author_name'])}**"
        lines.append(
            f"`#{n}` **{r['severity']}/10**{who} · {where}\n"
            f"> {clip(discord.utils.escape_markdown(r['content']), 160)}\n"
            f"*{clip(r['reason'] or 'no comment', 90)}*" + (f" · `{tags(r['labels'])}`" if r["labels"] else "")
        )
    embed = discord.Embed(
        title=title,
        description="\n\n".join(lines) or "Nothing on file. Suspiciously clean.",
        color=0xE91E63,
    )
    embed.set_footer(text=f"Page {page + 1}/{pages} · {total:,} kimoi posts")
    return embed, pages


class KimoiPager(discord.ui.View):
    """◀ ▶ buttons for the archive. Only the person who opened it can flip pages."""

    def __init__(self, owner_id: int, guild_id: int, user: discord.Member | None, pages: int):
        super().__init__(timeout=300)
        self.owner_id, self.guild_id, self.user = owner_id, guild_id, user
        self.page, self.pages = 0, pages
        self.message: discord.Message | None = None
        self._sync()

    def _sync(self) -> None:
        self.first.disabled = self.prev.disabled = self.page == 0
        self.next.disabled = self.last.disabled = self.page >= self.pages - 1

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "Open your own with `/archive`.", ephemeral=True
            )
            return False
        return True

    async def _show(self, interaction: discord.Interaction, page: int) -> None:
        embed, self.pages = kimoi_page(self.guild_id, self.user, page)  # re-count: scans may add posts
        self.page = min(page, self.pages - 1)
        if self.page != page:
            embed, _ = kimoi_page(self.guild_id, self.user, self.page)
        self._sync()
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(emoji="⏮️", style=discord.ButtonStyle.secondary)
    async def first(self, interaction: discord.Interaction, _):
        await self._show(interaction, 0)

    @discord.ui.button(emoji="◀️", style=discord.ButtonStyle.primary)
    async def prev(self, interaction: discord.Interaction, _):
        await self._show(interaction, max(0, self.page - 1))

    @discord.ui.button(emoji="▶️", style=discord.ButtonStyle.primary)
    async def next(self, interaction: discord.Interaction, _):
        await self._show(interaction, self.page + 1)

    @discord.ui.button(emoji="⏭️", style=discord.ButtonStyle.secondary)
    async def last(self, interaction: discord.Interaction, _):
        await self._show(interaction, self.pages - 1)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass


@bot.hybrid_command(name="archive", aliases=["kimoiposts", "kimoilist"],
                    help="Every kimoi post, most kimoi first, with page buttons.")
@app_commands.describe(member="Only this person's posts (leave empty for everyone)")
@commands.cooldown(1, 10, commands.BucketType.user)
async def kimoiposts(ctx: commands.Context, member: Suspect = None):
    embed, pages = kimoi_page(ctx.guild.id, member, 0)
    if pages <= 1:
        await ctx.send(embed=embed)
        return
    view = KimoiPager(ctx.author.id, ctx.guild.id, member, pages)
    view.message = await ctx.send(embed=embed, view=view)


@bot.hybrid_command(help="A user's kimoi file: rank, stats and worst posts.")
@app_commands.describe(member="Whose file to open (leave empty for yours)")
@commands.cooldown(1, 10, commands.BucketType.user)
async def kimoi(ctx: commands.Context, member: Suspect = None):
    member = member or ctx.author
    found = db.standing(ctx.guild.id, member.id)
    if not found:
        await ctx.send(f"{member.display_name} has a clean record. For now.")
        return
    rank, s = found
    embed = discord.Embed(
        title=f"🗂️ File: {member.display_name}",
        description=f"**Rank #{rank}** · {s.points:.1f} pts · {s.hits}/{s.judged} posts kimoi "
        f"({s.rate:.1%}) · avg severity {s.avg_severity:.1f}/10",
        color=0xE91E63,
    )
    for p in db.worst_posts(ctx.guild.id, member.id):
        link = jump_url(ctx.guild.id, p["channel_id"], p["id"])
        embed.add_field(
            name=f"{p['severity']}/10 — {clip(p['reason'] or 'no comment', 80)}"
            + (f" · {tags(p['labels'])}" if p["labels"] else ""),
            value=f"[{clip(discord.utils.escape_markdown(p['content']), 150)}]({link})",
            inline=False,
        )
    await ctx.send(embed=embed)


POSSESSIVE = (
    "If my experiences hadn’t included the IRL possessive nature and entitlement of seiyuu and also being "
    "spit on when someone cheered their name, I would think differently but instead, half the content here "
    "includes poorly socialized people with their masturbatory fantasies about seiyuu that are just acting, "
    "not actually interested.\nCreepy and possessive."
)


@bot.hybrid_command(help="Post the possessive copypasta.")
@commands.cooldown(1, 30, commands.BucketType.channel)
async def possessive(ctx: commands.Context):
    await ctx.send(POSSESSIVE)


@bot.hybrid_command(help="Remove yourself from surveillance and delete your stored posts.")
async def optout(ctx: commands.Context):
    db.opt_out(ctx.guild.id, ctx.author.id)
    await ctx.send(f"{ctx.author.display_name} has been removed from the watchlist and their file shredded.")


@bot.hybrid_command(help="Rejoin the kimoi rankings (takes effect for new messages).")
async def optin(ctx: commands.Context):
    db.opt_in(ctx.guild.id, ctx.author.id)
    await ctx.send(f"{ctx.author.display_name} is back under surveillance. Brave.")


async def suspect_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    """Suggest everyone on file, including people who have left (value is their ID)."""
    rows = db.author_names(interaction.guild_id or 0, current.lstrip("@"), 25)
    return [app_commands.Choice(name=r["author_name"][:100], value=str(r["author_id"])) for r in rows]


for _cmd in (dossier, roast, kimoiposts, kimoi):
    _cmd.autocomplete("member")(suspect_autocomplete)


def usage_line(command: commands.Command) -> str:
    return f"`/{command.name}{' ' + command.signature if command.signature else ''}`"


@bot.hybrid_command(name="help", aliases=["commands"], help="List every command, or explain one: /help archive")
@app_commands.describe(name="A command to explain")
@commands.cooldown(1, 5, commands.BucketType.user)
async def help_(ctx: commands.Context, name: str | None = None):
    if name:
        command = bot.get_command(name.lstrip(PREFIX).lstrip("/"))
        if command is None or (admin_only(command) and ctx.author.id not in ADMIN_IDS):
            await ctx.send(f"No command called `{name}`. Try `/help`.", ephemeral=True)
            return
        embed = discord.Embed(title=usage_line(command), description=command.help or "", color=0xE91E63)
        if command.aliases:
            embed.add_field(name="Also works as", value=", ".join(f"`{PREFIX}{a}`" for a in [command.name, *command.aliases]))
        if admin_only(command):
            embed.set_footer(text="Admins only")
        await ctx.send(embed=embed)
        return

    def section(cmds) -> str:
        return "\n".join(f"{usage_line(c)} {c.help or ''}" for c in sorted(cmds, key=lambda c: c.name))

    visible = [c for c in bot.commands if not c.hidden]
    embed = discord.Embed(
        title="🕵️ NSA field manual",
        description="The Neckbeard Surveillance Agency reads the chat and ranks the kimoi.",
        color=0xE91E63,
    )
    embed.add_field(name="Everyone", value=section(c for c in visible if not admin_only(c)), inline=False)
    if ctx.author.id in ADMIN_IDS:
        embed.add_field(name="Admins (spend API credit)", value=section(c for c in visible if admin_only(c)),
                        inline=False)
    embed.set_footer(text=f"/help <command> for details · every command also works with {PREFIX} · [optional] <required>")
    await ctx.send(embed=embed)


# --- lifecycle --------------------------------------------------------------

def describe_invocation(ctx: commands.Context) -> str:
    if ctx.interaction:  # slash: no message text, rebuild it from the options
        args = " ".join(f"{k}:{v}" for k, v in (ctx.interaction.namespace.__dict__ or {}).items())
        return repr(f"/{ctx.command.qualified_name} {args}".strip()) if ctx.command else "a slash command"
    return repr(ctx.message.content[:100])


@bot.listen("on_command")
async def log_command(ctx: commands.Context):
    log.info("%s (%s) ran %s in #%s", ctx.author, ctx.author.id, describe_invocation(ctx), getattr(ctx.channel, "name", "DM"))


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.CheckFailure):
        log.info("refused %s from %s (%s): %s", describe_invocation(ctx), ctx.author, ctx.author.id, error)
        if ctx.guild and ctx.guild.id in GUILD_IDS:  # stay silent in unlisted servers and DMs
            await ctx.send(str(error), ephemeral=True)
    elif isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"The analyst is busy. Try again in {error.retry_after:.0f}s.", ephemeral=True)
    elif isinstance(error, commands.UserInputError):
        await ctx.send(str(error), ephemeral=True)
    else:
        log.error("command failed", exc_info=error)
        await ctx.send("Something went wrong in the field office.")


async def sync_slash_commands() -> None:
    """Register slash commands only in the allowed servers: they appear instantly and nowhere else."""
    for guild_id in GUILD_IDS:
        guild = discord.Object(guild_id)
        bot.tree.copy_global_to(guild=guild)
        try:
            synced = await bot.tree.sync(guild=guild)
            log.info("registered %d slash commands in %s", len(synced), guild_id)
        except discord.HTTPException as e:
            log.warning("couldn't register slash commands in %s (%s). Re-invite the bot with the"
                        " applications.commands scope; ! commands still work.", guild_id, e)


def channel_name(guild_id: int, channel_id: int) -> str | None:
    guild = bot.get_guild(guild_id)
    ch = guild.get_channel_or_thread(channel_id) if guild else None
    return f"#{ch.name}" if ch else None


def guild_name(guild_id: int) -> str | None:
    guild = bot.get_guild(guild_id)
    return guild.name if guild else None


async def setup() -> None:
    await sync_slash_commands()
    if API_KEYS:
        app = api.build_app(db, API_KEYS, GUILD_IDS, per_minute=API_RATE, channel_name=channel_name,
                            guild_name=guild_name,
                            thresholds={"report": REPORT_MIN_SEVERITY, "var": VAR_MIN_SEVERITY})
        await api.start(app, API_HOST, API_PORT)
        log.info("agent API on for %d key(s): %s", len(API_KEYS), ", ".join(sorted(API_KEYS.values())))


bot.setup_hook = setup


@bot.event
async def on_guild_join(guild: discord.Guild):
    await leave_if_unlisted(guild)


@bot.event
async def on_ready():
    for guild in bot.guilds:
        await leave_if_unlisted(guild)
    for guild in bot.guilds:  # baseline the audit log so VAR can spot merged mod deletions from the start
        if guild.id in GUILD_IDS and guild.me.guild_permissions.view_audit_log:
            try:
                await refresh_mod_deletes(guild, prime=True)
            except discord.HTTPException:
                pass
    watching.clear()
    watching.update({g: c for g, c in db.watched().items() if g in GUILD_IDS})
    if not heartbeat.is_running():
        heartbeat.start()
    log.info("logged in as %s; watching %d guild(s)", bot.user, len(watching))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if changed := db.recompute_scores(scoring.score):  # formula weights changed since last run
        log.info("re-applied the scoring formula: %d scores changed", changed)
    if not ADMIN_IDS or not GUILD_IDS:
        raise SystemExit("NSA_ADMIN_IDS and NSA_GUILD_IDS must both be set (comma-separated Discord IDs).")
    bot.run(os.environ["DISCORD_TOKEN"], log_handler=None)


if __name__ == "__main__":
    main()
