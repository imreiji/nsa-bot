"""Discord front end: watch channels, feed posts to the DeepSeek judge, report and rank the kimoi."""

import asyncio
import logging
import os
from collections import defaultdict

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from .db import DB, Message
from .judge import Judge

log = logging.getLogger("nsabot")

BATCH_SIZE = 40          # messages per DeepSeek call
CONCURRENCY = 4          # parallel DeepSeek calls
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
MIN_CHARS = int(os.getenv("NSA_MIN_CHARS", "3"))
USER_RATE = int(os.getenv("NSA_USER_RATE", "10"))  # live posts queued per user per minute; extra spam is dropped
PREFIX = os.getenv("NSA_PREFIX", "!")

db = DB(os.getenv("NSA_DB_PATH", "nsa.db"))
judge = Judge(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    db=db,
)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(
    command_prefix=PREFIX,
    intents=intents,
    # Reposted messages must never ping anyone (@everyone, roles, users).
    allowed_mentions=discord.AllowedMentions.none(),
)
guild_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
watching: dict[int, int] = {}       # guild_id -> report channel id
spam_limiter = commands.CooldownMapping.from_cooldown(USER_RATE, 60, commands.BucketType.member)
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
    return commands.check(predicate)


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


def worth_judging(m: discord.Message, opted_out: set[int]) -> bool:
    text = m.content.strip()
    return (
        not m.author.bot
        and m.author.id not in opted_out
        and len(text) >= MIN_CHARS
        and not text.startswith(PREFIX)  # bot commands
    )


def to_row(m: discord.Message) -> Message:
    return Message(m.id, m.guild.id, m.channel.id, m.author.id, m.author.display_name, m.content)


# --- pipeline: scrape -> judge -> report ------------------------------------

async def scrape(channel: discord.TextChannel, opted_out: set[int]) -> int:
    """Pull messages newer than the channel's cursor (oldest first), up to SCAN_LIMIT."""
    cursor = db.get_cursor(channel.id)
    after = discord.Object(cursor) if cursor else None
    pending: list[Message] = []
    last_id = cursor or 0
    saved = 0
    async for m in channel.history(limit=SCAN_LIMIT, after=after, oldest_first=True):
        last_id = m.id
        if worth_judging(m, opted_out):
            pending.append(to_row(m))
        if len(pending) >= SAVE_EVERY:
            db.save_batch(channel.id, last_id, pending)
            saved += len(pending)
            pending = []
    if last_id:
        db.save_batch(channel.id, last_id, pending)
    return saved + len(pending)


async def judge_backlog(guild_id: int, progress=None) -> tuple[int, int]:
    """Run queued messages through DeepSeek. Returns (judged, flagged)."""
    sem = asyncio.Semaphore(CONCURRENCY)
    judged = flagged = 0

    async def run(rows) -> tuple[int, int]:
        async with sem:
            try:
                result = await judge.judge([(r["author_name"], r["content"]) for r in rows])
            except Exception:
                log.exception("judge batch failed; leaving %d messages queued", len(rows))
                return 0, 0
        db.save_verdicts([(r["id"], *result.get(i, (0, None))) for i, r in enumerate(rows)])
        return len(rows), len(result)

    while True:
        rows = db.unjudged(guild_id, BATCH_SIZE * CONCURRENCY * 4)
        if not rows:
            break
        batches = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
        results = await asyncio.gather(*(run(b) for b in batches))
        round_judged = sum(j for j, _ in results)
        judged += round_judged
        flagged += sum(f for _, f in results)
        if progress:
            await progress(f"🕵️ Analysed {judged} posts, {flagged} flagged as kimoi…")
        if round_judged == 0:  # every batch failed; don't spin on a dead API
            break
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
        embed.add_field(name="Location", value=f"<#{r['channel_id']}>", inline=False)
        await channel.send(embed=embed)
    if len(rows) > REPORT_MAX_PER_RUN:
        await channel.send(f"…and {len(rows) - REPORT_MAX_PER_RUN} more kimoi posts filed. See `{PREFIX}kimoiboard`.")
    db.mark_reported([r["id"] for r in rows])
    return len(rows)


async def process(guild: discord.Guild, progress=None) -> tuple[int, int, int]:
    """Judge the queue and report results. Caller must hold the guild lock."""
    if guild.id not in GUILD_IDS:
        raise PermissionError(f"refusing to call DeepSeek for unlisted guild {guild.id}")
    judged, flagged = await judge_backlog(guild.id, progress)
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
    if not worth_judging(m, db.opted_out(m.guild.id)):
        return
    if spam_limiter.update_rate_limit(m):  # flooding can't buy extra API calls
        return
    db.save_message(to_row(m))
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


# --- commands ---------------------------------------------------------------

async def run_scan(ctx: commands.Context, channels: list[discord.abc.Messageable]) -> None:
    lock = guild_locks[ctx.guild.id]
    if lock.locked():
        await ctx.send("A surveillance sweep is already running in this server.")
        return
    async with lock:
        status = await ctx.send(f"📡 Intercepting {len(channels)} channel(s)…")
        opted_out = db.opted_out(ctx.guild.id)
        scraped = 0
        for ch in channels:
            try:
                scraped += await scrape(ch, opted_out)
            except discord.Forbidden:
                await ctx.send(f"No clearance for {ch.mention}, skipping.")
        await status.edit(content=f"📡 Intercepted {scraped} new posts. Handing them to the analyst…")
        judged, flagged, posted = await process(ctx.guild, lambda text: status.edit(content=text))
        left = db.count_unjudged(ctx.guild.id)
        notes = []
        if left:
            notes.append(f"{left} posts failed (API errors), rerun to retry.")
        if ctx.guild.id not in watching:
            notes.append(f"No report channel set (`{PREFIX}watch #channel`), so nothing was posted.")
        await status.edit(
            content=f"✅ Sweep complete: {scraped} new posts intercepted, {judged} analysed, {flagged} kimoi "
            f"(all go on the leaderboard), {posted} at ≥{REPORT_MIN_SEVERITY}/10 posted to the report channel. " + " ".join(notes)
        )


@bot.command(help="Scan channels' history for kimoi posts (default: this channel).")
@deployer_only()
async def scan(ctx: commands.Context, *channels: discord.TextChannel):
    targets = [c for c in (channels or [ctx.channel]) if watchable(ctx.guild.id, c.id)]
    await run_scan(ctx, targets)


@bot.command(help="Scan every text channel the bot can read.")
@deployer_only()
async def scanall(ctx: commands.Context):
    me = ctx.guild.me
    channels = [
        c for c in ctx.guild.text_channels
        if c.permissions_for(me).read_message_history and watchable(ctx.guild.id, c.id)
    ]
    await run_scan(ctx, channels)


@bot.command(help="Start live surveillance, posting kimoi to the given report channel.")
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


@bot.command(help="Stop live surveillance.")
@deployer_only()
async def unwatch(ctx: commands.Context):
    db.set_watch(ctx.guild.id, None)
    watching.pop(ctx.guild.id, None)
    await ctx.send("Live surveillance off. Queued posts will be judged on the next scan.")


@bot.command(help="Show DeepSeek token usage and the queue.")
@deployer_only()
async def usage(ctx: commands.Context):
    await ctx.send(
        f"💸 {db.tokens_today():,} DeepSeek tokens used today (UTC). "
        f"{db.count_unjudged(ctx.guild.id)} posts queued. "
        f"Live watch: {'on' if ctx.guild.id in watching else 'off'}."
    )


@bot.command(help="Have the analyst write up a classified dossier on someone.")
@deployer_only()
@commands.cooldown(1, 30, commands.BucketType.guild)
async def dossier(ctx: commands.Context, member: discord.Member | None = None):
    member = member or ctx.author
    found = db.standing(ctx.guild.id, member.id)
    if not found:
        await ctx.send(f"Insufficient evidence on {member.display_name}.")
        return
    rank, s = found
    posts = [(p["severity"], p["content"], p["reason"] or "") for p in db.worst_posts(ctx.guild.id, member.id, 8)]
    stats = f"rank #{rank}, {s.hits} kimoi posts out of {s.judged}, avg severity {s.avg_severity:.1f}/10"
    async with ctx.typing():
        text = await judge.roast(member.display_name, stats, posts)
    await ctx.send(f"**CLASSIFIED — {member.display_name}**\n{clip(text, 1900)}")


@bot.command(aliases=["kimoirank"], help="The kimoi leaderboard.")
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


@bot.command(help="A user's kimoi file: rank, stats and worst posts.")
@commands.cooldown(1, 10, commands.BucketType.user)
async def kimoi(ctx: commands.Context, member: discord.Member | None = None):
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
            name=f"{p['severity']}/10 — {clip(p['reason'] or 'no comment', 80)}",
            value=f"[{clip(discord.utils.escape_markdown(p['content']), 150)}]({link})",
            inline=False,
        )
    await ctx.send(embed=embed)


@bot.command(help="Remove yourself from surveillance and delete your stored posts.")
async def optout(ctx: commands.Context):
    db.opt_out(ctx.guild.id, ctx.author.id)
    await ctx.send(f"{ctx.author.display_name} has been removed from the watchlist and their file shredded.")


@bot.command(help="Rejoin the kimoi rankings (takes effect for new messages).")
async def optin(ctx: commands.Context):
    db.opt_in(ctx.guild.id, ctx.author.id)
    await ctx.send(f"{ctx.author.display_name} is back under surveillance. Brave.")


# --- lifecycle --------------------------------------------------------------

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.CheckFailure):
        if ctx.guild and ctx.guild.id in GUILD_IDS:  # stay silent in unlisted servers and DMs
            await ctx.send(str(error))
    elif isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"The analyst is busy. Try again in {error.retry_after:.0f}s.")
    elif isinstance(error, commands.UserInputError):
        await ctx.send(str(error))
    else:
        log.error("command failed", exc_info=error)
        await ctx.send("Something went wrong in the field office.")


@bot.event
async def on_guild_join(guild: discord.Guild):
    await leave_if_unlisted(guild)


@bot.event
async def on_ready():
    for guild in bot.guilds:
        await leave_if_unlisted(guild)
    watching.clear()
    watching.update({g: c for g, c in db.watched().items() if g in GUILD_IDS})
    if not heartbeat.is_running():
        heartbeat.start()
    log.info("logged in as %s; watching %d guild(s)", bot.user, len(watching))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not ADMIN_IDS or not GUILD_IDS:
        raise SystemExit("NSA_ADMIN_IDS and NSA_GUILD_IDS must both be set (comma-separated Discord IDs).")
    bot.run(os.environ["DISCORD_TOKEN"], log_handler=None)


if __name__ == "__main__":
    main()
