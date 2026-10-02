# nsa-bot

Neckbeard Surveillance Agency. A Discord bot that scrapes channels, has a DeepSeek agent read every
post and score how **kimoi** (キモい) it is, then ranks the server's otaku by frequency and severity.

## How it works

1. `!watch #kimoi-reports` turns on live surveillance. Every new message in the server (except the
   report channel, ignored channels, bots, commands and opted-out users) goes into a queue.
2. When the queue hits `NSA_QUEUE_TRIGGER` posts (default 40), or every `NSA_HEARTBEAT_MINUTES`
   (default 10) if anything is waiting, DeepSeek judges the queue in one batch.
3. Anything scoring `NSA_REPORT_MIN_SEVERITY`+ (default 5/10) is posted to the report channel
   with the quote, the analyst's note and a link to the original. Reposts never ping anyone.
4. Every judged post feeds the leaderboard.

`!scan` / `!scanall` do the same for channel history (incremental, `NSA_SCAN_LIMIT` per channel per
run, oldest first). At most 20 reports are posted per sweep; the rest are summarised in one line.

## Commands

| Command | Who | What |
|---|---|---|
| `!watch #channel` / `!unwatch` | admin UIDs | Start / stop live surveillance and set the report channel |
| `!scan [#ch ...]` / `!scanall` | admin UIDs | Judge channel history (default: current channel) |
| `!usage` | admin UIDs | DeepSeek tokens used today, queue size |
| `!dossier [@user]` | admin UIDs | DeepSeek writes a classified report roasting the user's worst posts |
| `!kimoiboard` | anyone | Top 10 leaderboard (no API cost) |
| `!kimoi [@user]` | anyone | Rank, stats and worst posts with links (no API cost) |
| `!possessive` | anyone | Posts the possessive copypasta (30s cooldown per channel) |
| `!optout` / `!optin` | anyone | Leave the rankings (deletes your stored posts) / rejoin |

## Who can trigger API calls

DeepSeek is only called from two code paths, and both are locked down:

- **Judging** (`process()` in `nsabot/bot.py`) runs from `!scan` / `!scanall`, which only
  `NSA_ADMIN_IDS` can use, or from live watching, which only an admin can turn on with `!watch`.
  It also refuses to run for any server not in `NSA_GUILD_IDS`.
- **`!dossier`** is admin-only.

Everything else anyone can run (`!kimoiboard`, `!kimoi`, `!optout`, `!optin`) only reads the
local database. Other protections:

- **Hardcoded admins**: Discord roles and permissions grant nothing; only the IDs in
  `NSA_ADMIN_IDS` count. Commands in DMs are refused.
- **Server allowlist**: the bot leaves any server not in `NSA_GUILD_IDS`, so nobody can invite it
  elsewhere. It refuses to start unless both lists are set.
- **Spam can't buy API calls**: in live mode each user can queue at most `NSA_USER_RATE` posts per
  minute (default 10); anything beyond that is dropped, not judged. Bots and commands are never
  queued.
- **Bounded requests**: output is capped per call, retries are limited to 2, messages are truncated
  to 800 characters, and very short messages are skipped.
- **Prompt injection**: posts are sent as JSON data, and the model is told never to follow
  instructions inside them. Worst case, someone games their own score.
- **Usage**: `!usage` shows today's token count (no cap; it's just for watching the bill).
- **Outside the bot**: turn off **Public Bot** in the Discord developer portal so only you can
  invite it, keep the DeepSeek balance small (it's prepaid), and never commit `.env`.

## Scoring

DeepSeek reads messages in batches of 40 (with surrounding context) and gives each a severity of
0-10. A user's score is **Σ severity² / 10**: every kimoi post counts, but one 10/10 post (10 pts)
beats ten 3/10 posts (9 pts). The board also shows hit rate (kimoi posts ÷ judged posts) and average
severity. The rubric is in `nsabot/judge.py` (`JUDGE_PROMPT`); edit it to fit your server.

## Setup

### 1. Discord bot

1. <https://discord.com/developers/applications> → **New Application**.
2. **Bot** tab: **Reset Token** and copy it (`DISCORD_TOKEN`). Turn on **Message Content Intent**.
   Turn off **Public Bot**.
3. Invite it (replace `APP_ID` with the Application ID from **General Information**):
   `https://discord.com/oauth2/authorize?client_id=APP_ID&scope=bot&permissions=84992`
   (View Channels, Send Messages, Embed Links, Read Message History.)
4. In Discord, **Settings → Advanced → Developer Mode** on, then right-click yourself → **Copy User
   ID** (`NSA_ADMIN_IDS`) and right-click the server → **Copy Server ID** (`NSA_GUILD_IDS`).
5. Create a channel for reports, e.g. `#kimoi-reports`. Hide the bot from channels it shouldn't read.

### 2. DeepSeek

Get an API key at <https://platform.deepseek.com> (`DEEPSEEK_API_KEY`) and top up a small balance.

### 3. Run it (any always-on Linux box with Docker)

```sh
git clone https://github.com/imreiji/nsa-bot && cd nsa-bot
cp .env.example .env && nano .env      # tokens, NSA_ADMIN_IDS, NSA_GUILD_IDS
chmod 600 .env
docker compose up -d --build
docker compose logs -f                 # expect "logged in as ..."
```

Then in Discord: `!watch #kimoi-reports`, and optionally `!scanall` to go through history.

- **Update**: `git pull && docker compose up -d --build`
- **Back up the database**: `docker compose cp nsa-bot:/data/nsa.db ./nsa-backup.db`
- The database lives in the `nsa-data` volume, so it survives rebuilds. `restart: unless-stopped`
  brings the bot back after crashes and reboots.

Without Docker: `python -m venv .venv && .venv/bin/pip install -r requirements.txt` and run
`.venv/bin/python -m nsabot.bot` under systemd, tmux or similar.

Tests: `.venv/bin/pip install pytest && .venv/bin/python -m pytest`
