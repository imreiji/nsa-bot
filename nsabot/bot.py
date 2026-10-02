"""Discord front end: scrape channels, feed them to the DeepSeek judge, show the kimoi rankings."""

import asyncio
import logging
import os
from collections import defaultdict

import discord
from discord.ext import commands
from dotenv import load_dotenv

from .db import DB, Message
from .judge import Judge

log = logging.getLogger("nsabot")

BATCH_SIZE = 40       # messages per DeepSeek call
CONCURRENCY = 4       # parallel DeepSeek calls
SAVE_EVERY = 500      # scraped messages per DB write / cursor checkpoint

load_dotenv()
db = DB(os.getenv("NSA_DB_PATH", "nsa.db"))
judge = Judge(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
SCAN_LIMIT = int(os.getenv("NSA_SCAN_LIMIT", "5000"))

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=os.getenv("NSA_PREFIX", "!"), intents=intents)
guild_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def jump_url(guild_id: int, channel_id: int, message_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


async def scrape(channel: discord.TextChannel, opted_out: set[int]) -> int:
    """Pull messages newer than the channel's cursor (oldest first), up to SCAN_LIMIT."""
    cursor = db.get_cursor(channel.id)
    after = discord.Object(cursor) if cursor else None
    pending: list[Message] = []
    last_id = cursor or 0
    saved = 0
    async for m in channel.history(limit=SCAN_LIMIT, after=after, oldest_first=True):
        last_id = m.id
        if not m.author.bot and m.author.id not in opted_out and m.content.strip():
            pending.append(Message(m.id, channel.guild.id, channel.id, m.author.id, m.author.display_name, m.content))
        if len(pending) >= SAVE_EVERY:
            db.save_batch(channel.id, last_id, pending)
            saved += len(pending)
            pending = []
    if last_id:
        db.save_batch(channel.id, last_id, pending)
    return saved + len(pending)


async def judge_backlog(guild_id: int, progress) -> tuple[int, int]:
    """Run every unjudged message in the guild through DeepSeek. Returns (judged, flagged)."""
    sem = asyncio.Semaphore(CONCURRENCY)
    judged = flagged = 0

    async def run(rows) -> tuple[int, int]:
        async with sem:
            try:
                result = await judge.judge([(r["author_name"], r["content"]) for r in rows])
            except Exception:
                log.exception("judge batch failed; leaving %d messages for the next scan", len(rows))
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
        await progress(f"🕵️ Analysed {judged} posts, {flagged} flagged as kimoi…")
        if round_judged == 0:  # every batch failed; don't spin on a dead API
            break
    return judged, flagged


async def run_scan(ctx: commands.Context, channels: list[discord.TextChannel]) -> None:
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
        judged, flagged = await judge_backlog(ctx.guild.id, lambda text: status.edit(content=text))
        left = db.count_unjudged(ctx.guild.id)
        tail = f" {left} posts still pending (API errors), rerun to retry." if left else ""
        await status.edit(
            content=f"✅ Sweep complete: {scraped} new posts intercepted, {judged} analysed, "
            f"{flagged} kimoi.{tail} See `{ctx.prefix}kimoiboard`."
        )


@bot.command(help="Scan channels for kimoi posts (default: this channel). Admin only.")
@commands.guild_only()
@commands.has_guild_permissions(manage_guild=True)
async def scan(ctx: commands.Context, *channels: discord.TextChannel):
    await run_scan(ctx, list(channels) or [ctx.channel])


@bot.command(help="Scan every text channel the bot can read. Admin only.")
@commands.guild_only()
@commands.has_guild_permissions(manage_guild=True)
async def scanall(ctx: commands.Context):
    me = ctx.guild.me
    channels = [c for c in ctx.guild.text_channels if c.permissions_for(me).read_message_history]
    await run_scan(ctx, channels)


@bot.command(aliases=["kimoirank"], help="The kimoi leaderboard.")
@commands.guild_only()
async def kimoiboard(ctx: commands.Context):
    rows = db.leaderboard(ctx.guild.id)
    if not rows:
        await ctx.send(f"No kimoi on file. Run `{ctx.prefix}scan` first (or this server is suspiciously clean).")
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
@commands.guild_only()
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


@bot.command(help="Have the analyst write up a classified dossier on someone.")
@commands.guild_only()
@commands.cooldown(1, 30, commands.BucketType.user)
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


@bot.command(help="Remove yourself from surveillance and delete your stored posts.")
@commands.guild_only()
async def optout(ctx: commands.Context):
    db.opt_out(ctx.guild.id, ctx.author.id)
    await ctx.send(f"{ctx.author.display_name} has been removed from the watchlist and their file shredded.")


@bot.command(help="Rejoin the kimoi rankings (takes effect from the next scan of new messages).")
@commands.guild_only()
async def optin(ctx: commands.Context):
    db.opt_in(ctx.guild.id, ctx.author.id)
    await ctx.send(f"{ctx.author.display_name} is back under surveillance. Brave.")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, (commands.MissingPermissions, commands.NoPrivateMessage)):
        await ctx.send("You lack clearance for that.")
    elif isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"The analyst is busy. Try again in {error.retry_after:.0f}s.")
    elif isinstance(error, (commands.BadArgument, commands.UserInputError)):
        await ctx.send(str(error))
    elif not isinstance(error, commands.CommandNotFound):
        log.error("command failed", exc_info=error)
        await ctx.send("Something went wrong in the field office.")


@bot.event
async def on_ready():
    log.info("logged in as %s", bot.user)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    bot.run(os.environ["DISCORD_TOKEN"], log_handler=None)


if __name__ == "__main__":
    main()
