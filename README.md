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

Every command is a slash command (`/archive`) and also works with the prefix (`!archive`). Slash
commands are registered only in the servers in `NSA_GUILD_IDS` when the bot starts.

| Command | Who | What |
|---|---|---|
| `/watch #channel` / `/unwatch` | admin UIDs | Start / stop live surveillance and set the report channel |
| `/scan [channel]` / `/scanall` | admin UIDs | Judge history of a channel, voice chat or thread (default: current) / of everything |
| `/usage` | admin UIDs | DeepSeek tokens used today, queue size |
| `/dossier [@user]` | everyone (`NSA_DOSSIER_PUBLIC`, 3/hour each; admins unlimited) | Deadpan classified report on their kimoi record |
| `/roast [@user]` | admin UIDs (everyone with `NSA_ROAST_PUBLIC=on`, 3/hour each) | Comedy roast built from their recent posts, kimoi stats and worst posts |
| `/kimoiboard` | anyone | Top 10 leaderboard |
| `/kimoi [@user]` | anyone | Rank, stats and worst posts with links |
| `/archive [@user]` | anyone | Every kimoi post, most kimoi first, 10 per page with ⏮️ ◀️ ▶️ ⏭️ buttons |
| `/possessive` | anyone | Posts the possessive copypasta (30s cooldown per channel) |
| `/optout` / `/optin` | anyone | Leave the rankings (deletes your stored posts) / rejoin |
| `/help [command]` | anyone | Lists every command you can use (admins also see admin ones), or explains one |

Person options autocomplete from everyone on file, including people who have left the server.
Only `/dossier` and `/roast` call DeepSeek on behalf of non-admins, and only within their hourly
limits.

## Who can trigger API calls

DeepSeek is only called from these code paths:

- **Judging** (`process()` in `nsabot/bot.py`) runs from `/scan` / `/scanall`, which only
  `NSA_ADMIN_IDS` can use, or from live watching, which only an admin can turn on with `/watch`.
  It also refuses to run for any server not in `NSA_GUILD_IDS`.
- **VAR** reviews of self-deleted posts, only in watched servers, capped per person per hour.
- **`/dossier` and `/roast`**: admins always; everyone else only while `NSA_DOSSIER_PUBLIC` /
  `NSA_ROAST_PUBLIC` is on, and then a few per person per hour.

Everything else anyone can run (`/kimoiboard`, `/kimoi`, `/archive`, `/optout`, ...) only reads the
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

## Context

DeepSeek never sees a post on its own. Each batch is one stretch of one channel, and every message
carries:

- **Channel**: name, topic and NSFW flag (threads show as `#parent > thread`).
- **Conversation**: the previous `NSA_CONTEXT_MESSAGES` messages (default 15) plus everything in
  between, in order with timestamps.
- **Replies**: the author and text of the message being replied to, looked up from the database if
  Discord didn't include it.
- **Attachments**: file names and types, stickers, link-preview titles (YouTube, X, etc.) and
  forwarded messages. Images themselves aren't sent.
- **Reactions**: short posts ("w", "lol") and image-only posts are stored as context but never
  scored, so they don't count toward anyone's rate.

Opted-out users are never stored or quoted, even as reply context. Context costs roughly 30-50%
more tokens per batch than judging posts on their own.

## Ask the analyst

Reply to someone's message and **@mention the bot** (optionally with a request, e.g. "@NSA Bot is
this a unicorn take?"). It reads the 30 messages before your ping, plus the target author's kimoi
file if they have one, and replies to that message in character, in the language you wrote in.

- Only explicit @mentions count. Replying to the bot's own messages (which pings it automatically)
  is ignored, and so are plain pings that aren't replies.
- Thinking is off for these (`NSA_RESPOND_THINKING`), so replies take seconds and cost about $0.001.
- `NSA_RESPOND=on` (everyone, `NSA_RESPOND_PER_USER_HOUR` each, default 10; admins unlimited),
  `admins`, or `off`. Over the limit the bot reacts ⏳ instead of replying.
- Opted-out users' messages are left out of the context, and their posts are never answered.

## VAR mode

When someone deletes their own message in a watched server, the bot replays it to DeepSeek with
the 15 messages before it and 5 after, and if it scores `NSA_VAR_MIN_SEVERITY`+ (default: the
report threshold) posts a **📺 VAR REVIEW** to the report channel. It also counts on the board.

- Content comes from the bot's database, so even old deletions can be reviewed.
- Already-scored posts are aired without an API call; nothing is aired twice.
- Low scores stay silent, and the judge is told to score 0 anything that looks deleted because it
  was private (addresses, phone numbers, real names, personal stuff).
- Mod deletions are skipped when the bot has **View Audit Log**; without it, every single delete
  is treated as a self-delete. Bulk purges are ignored.
- At most `NSA_VAR_PER_USER_HOUR` (default 5) paid reviews per person per hour.
- Turn off with `NSA_VAR=off`.

## Quips

While judging, DeepSeek may add a one-line joke when it thinks the moment calls for it: the
chat just did something absurd, ironic or very kimoi. It stays quiet otherwise, and never jokes
when someone is upset. Jokes about live chat (under an hour old) go straight into that channel;
jokes about old history from a scan go to the report channel with a link. At most one per server
every `NSA_QUIP_COOLDOWN_MINUTES` (default 30). Turn off with `NSA_QUIPS=off`. No extra API calls.

## Scoring

DeepSeek reads messages in batches of up to 40 scored posts per channel and gives each a severity
of 0-10. A user's score is **Σ severity² / 10**: every kimoi post counts, but one 10/10 post (10 pts)
beats ten 3/10 posts (9 pts). The board also shows hit rate (kimoi posts ÷ judged posts) and average
severity. The rubric is in `nsabot/judge.py` (`JUDGE_PROMPT`); edit it to fit your server.

## Setup

### 1. Discord bot

1. <https://discord.com/developers/applications> → **New Application**.
2. **Bot** tab: **Reset Token** and copy it (`DISCORD_TOKEN`). Turn on **Message Content Intent**.
   Turn off **Public Bot**.
3. Invite it (replace `APP_ID` with the Application ID from **General Information**):
   `https://discord.com/oauth2/authorize?client_id=APP_ID&scope=bot+applications.commands&permissions=85120`
   (View Channels, Send Messages, Embed Links, Read Message History, View Audit Log, plus slash
   commands.)
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
