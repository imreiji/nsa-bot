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
| `/scan [channel] [recent]` / `/scanall [recent]` | admin UIDs | Judge history of a channel, voice chat or thread (default: current) / of everything. `recent:N` reads only the latest N messages (per channel) and leaves the scan position alone, so a later full scan still covers older history; posts already judged keep their scores |
| `/usage` | admin UIDs | DeepSeek tokens used today, queue size |
| `/calibrate` / `/calibration` | admin UIDs | Score posts yourself in DMs / DM report comparing the formula with your scores |
| `/evaluate [limit]` | admin UIDs | Test the current prompt on the reviewed posts, report in DMs (nothing saved) |
| `/notes [file]` | admin UIDs | Get the judge's server notes in DMs, or replace them with an attached file |
| `/rescore [scope]` | admin UIDs | Re-judge posts scored under an older rubric, or with `scope:flagged` only the flagged ones (much cheaper). Re-runs DeepSeek; old history isn't re-posted |
| `/look <message link>` | admin UIDs | Test picture judging on one message: which pictures the bot finds, whether they download, what the model says it sees, and the judge's verdict (nothing saved) |
| `/model` | anyone | Which model (Claude or DeepSeek) and settings the bot uses |
| `/scoring` | anyone | The scoring formula, thresholds and calibration accuracy |
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
- Thinking is on (`NSA_RESPOND_THINKING`), so replies are more considered but take longer.
- `NSA_RESPOND=on` (everyone, unlimited), `admins`, or `off`. Set `NSA_RESPOND_PER_USER_HOUR` to cap
  it per person; over the cap the bot reacts ⏳ instead of replying.
- Opted-out users' messages are left out of the context, and their posts are never answered.

## VAR mode

When someone deletes their own message in a watched server, the bot replays it to DeepSeek with
the 15 messages before it and 5 after, and if it scores `NSA_VAR_MIN_SEVERITY`+ (default: the
report threshold) posts a **📺 VAR REVIEW** to the report channel. It also counts on the board.

- Content comes from the bot's database, so even old deletions can be reviewed.
- Already-scored posts are aired without an API call; nothing is aired twice.
- Low scores stay silent, and the judge is told to score 0 anything that looks deleted because it
  was private (addresses, phone numbers, real names, personal stuff).
- Self-deletes never appear in the audit log, but a mod deleting someone else's message does, so
  VAR treats a delete as the author's own only when no matching mod entry turns up. Discord merges
  repeated deletes by the same mod (same author, same channel) into one entry and bumps its count,
  so the bot tracks counts and treats a new entry *or* a higher count as a mod deletion. Each mod
  deletion covers exactly one delete event. Bulk purges are ignored.
- Give it **View Audit Log**. Without it VAR can't spot mod deletions, so it reviews every delete as
  if the author made it (a mod removing a kimoi post would get a VAR review too).
- At most `NSA_VAR_PER_USER_HOUR` (default 5) paid reviews per person per hour.
- Turn off with `NSA_VAR=off`.

## Quips

While judging, DeepSeek may add a one-line joke when it thinks the moment calls for it: the
chat just did something absurd, ironic or very kimoi. It stays quiet otherwise, and never jokes
when someone is upset. Jokes about live chat (under an hour old) go straight into that channel;
jokes about old history from a scan go to the report channel with a link. At most one per server
every `NSA_QUIP_COOLDOWN_MINUTES` (default 30). Turn off with `NSA_QUIPS=off`. No extra API calls.

## Scoring

DeepSeek doesn't pick numbers. It **labels** each kimoi post, and a fixed formula in
`nsabot/scoring.py` turns the labels into a score (`/scoring` shows it in Discord):

- **behaviours**: worship 2, spending 4, gachikoi 4, horny 5, unicorn 6, life impact 7,
  bodily/servitude 7, stalking/harassment 7 (start from the highest)
- **target**: real person +1, minor character +1 · **intensity**: passing −1, graphic +1 · **sincerity**: obvious bit −1,
  sincere +1 · **doubling down** +1 · **two or more behaviours** +1
- kept between 1 and 10; an oshi spiral (crying, drinking, drunk quitting, death jokes over a seiyuu)
  is at most 8; horny or bodily/servitude stuff about a 2D character, minors included, is at most 7
  (it's fiction); only pointing at someone else's kimoi, and real distress outside the fandom, is 0

Labels are stored with every verdict, so changing a weight re-applies to old posts on the next start
without any API calls. Changing the labels or their definitions in the prompt means bumping
`RUBRIC_VERSION` and running `/rescore`. `/rescore scope:flagged` re-judges only the posts currently
flagged (plus ones set aside as distress), whatever rubric judged them: a few hundred posts instead of
the whole history, for when the prompt or server notes changed. Posts the bot already let through
keep their verdicts.

A user's leaderboard score is **Σ severity² / 10**, so one 10/10 post outweighs ten 3/10 posts.

### How the judge decides (rubric v3)

The prompt is an ordered checklist: first what is never flagged (distress, pointing at someone
else's kimoi, normal fandom, normal life, friend banter between members, racial or ethnic remarks,
quotes, links and emotes), then **evidence**, then labels. Jokes are still flagged (as a "bit", which
keeps the score low), and a line that carries on the same author's kimoi burst counts with it. Every flag must quote the exact words
from the post itself that show the behaviour, and the bot checks the quote really is in the post;
flags that can't point at the words are dropped. An oshi spiral (crying, drinking, drunk "I'm
quitting seiyuu", isolating, even death jokes over a seiyuu) is a running joke here, so it is kimoi,
labelled `spiral` and capped at 8. Wanting to die or self-harm that has nothing to do with the
fandom, or someone who seems genuinely not okay, scores 0, is marked as distress, and is kept out of
roasts, dossiers and quips.

### Pictures

The judge sees pictures too (`NSA_IMAGES=on`): image attachments and link-preview pictures on the
posts being judged, at most 2 per post and `NSA_IMAGES_PER_BATCH` (default 10) per call, each up to
about 1k tokens. GIFs, stickers and video are skipped. Picture-only posts are judged instead of
being kept as context. A flag can cite `"[image]"` as its evidence, but only for a post that
actually carries a picture. Lewd art, doujin pages and creepshot-style crops count; official art,
screenshots, memes and ordinary merch photos don't. The bot downloads each picture itself and sends
it inline, because Discord's links are signed and expire after about a day; an expired link is
refreshed by re-reading the message once. Posts scraped before this was added have no picture links
stored, so the bot re-reads those messages from Discord the first time they're judged again.

### Server notes

`/notes` (admins) shows the judge's who's-who, and attaching a `.md` or `.txt` file to `/notes`
replaces it: who the members are, which names are seiyuu (adults) or characters (and which are
minors), and the server's running bits. The file lives next to the database (`NSA_SERVER_NOTES`,
default `/data/server_notes.md` in Docker), never in the repo, and applies to the next judging call.

### Evaluation

`/evaluate` (admins) re-judges the reviewed posts in `eval/review_set.jsonl` with the current prompt,
saves nothing, and DMs a before/after report: average gap to the review, share within 1 point,
false flags, safety posts still flagged, how many 10s, and the worst misses, plus `evaluation.txt` with
every post's review score, new score, labels, quote and reason (biggest misses first). The review set holds
only message IDs and scores; the posts come from the bot's own database. `/evaluate limit:100` is a
quick, cheap check. A full run is a few hundred calls, roughly $1-3 with thinking on.

### Calibration

`/calibrate` (admins) sends you posts **in DMs** with 0-10 buttons: alternately ones the bot flagged
and random ones. You answer first, then see the bot's score and labels. Your scores:

- become examples in the judge's prompt (up to 2 per score band, where the formula agreed with you)
- power `/calibration`, a DM report of how far the formula lands from your scores on average, whether
  it's too harsh or too soft, and its worst misses with their labels, so you can see which weight to change

## Agent API (read-only)

Lets trusted agents (e.g. Claude) read the posts the judge ranked, with their context, without
access to the box or the database. It's off until you create a key.

**Security:** every request needs a key; `.env` stores only each key's SHA-256 hash; per-key rate
limit (`NSA_API_RATE_PER_MINUTE`, default 60); 10 bad keys from one source locks it out for 10
minutes; only flagged posts and their context are served (no bulk dump of the chat), only from
`NSA_GUILD_IDS`, at most 100 posts per page and 30 messages of context; nothing can be written;
every request is logged with the key's name. Opted-out users are never stored, so never served.

### 1. Make a key per agent

```sh
python3 -m nsabot.apikey claude        # prints the key (once) and a name:hash line
```

Put the `name:hash` line in `.env` as `NSA_API_KEYS=claude:<hash>` (comma-separate several), restart
with `docker compose up -d`, and give the key itself to the agent. To revoke a key, delete its entry
and restart. The logs show `agent API on for 1 key(s): claude`.

### 2. Reach it

It listens on `127.0.0.1:8787` on the box only. Test from the box:

```sh
curl -H "Authorization: Bearer nsa_..." http://127.0.0.1:8787/v1/guilds
```

For agents elsewhere, publish it **over HTTPS only**, through the web server already on the box,
on its own subdomain (add a DNS record for it first). Caddy:

```
nsa-api.example.com {
    reverse_proxy 127.0.0.1:8787
}
```

nginx (with a certificate from certbot):

```
server {
    listen 443 ssl;
    server_name nsa-api.example.com;
    ssl_certificate     /etc/letsencrypt/live/nsa-api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/nsa-api.example.com/privkey.pem;
    location / {
        proxy_pass http://127.0.0.1:8787;
        proxy_set_header X-Forwarded-For $remote_addr;
    }
}
```

Open port 443 in the Lightsail firewall if it isn't already; never open 8787.

### 3. Endpoints

All `GET`, all JSON, all need `Authorization: Bearer <key>`. `guild` can be left out when the bot
serves one server.

| Endpoint | Returns |
|---|---|
| `/v1/guilds` | Servers the bot serves |
| `/v1/posts` | Flagged posts. Query: `min_severity` (1-10), `author` (user ID), `channel` (ID), `behaviour` (`unicorn`, `bodily_servitude`, ...), `order` (`severity` or `recent`), `limit` (max 100), `cursor` (from `next_cursor`) |
| `/v1/posts/{id}` | One flagged post plus `before` (default 15) and `after` (default 5) messages of context, max 30 each |
| `/v1/leaderboard` | Top users (`limit` max 50) |
| `/v1/users/{id}` | A user's rank, stats and worst posts |
| `/v1/scoring` | The formula weights, rubric version and thresholds |

Each post has `id`, `author`, `author_id`, `time`, `channel`, `text`, `reply_to`, `attachments`,
`severity`, `reason`, `labels`, `tags`, `deleted` (caught by VAR) and `url` (Discord jump link).

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

### 2. Model

The bot runs on **Claude Haiku 5.5** by default (`NSA_PROVIDER=anthropic`). Get an API key at
<https://console.anthropic.com> (`ANTHROPIC_API_KEY`) and set a monthly spend limit there.

- Haiku 5.5 costs $0.10 / $0.50 per million input / output tokens for prompts up to 100k tokens
  (judge batches are well under that). The judge prompt is cached, so most of each call's input is
  billed at $0.01 per million.
- Thinking is adaptive; `NSA_EFFORT` (low/medium/high/xhigh/max, blank = medium) sets how hard it
  thinks. Chat replies with thinking off run at effort `low`.
- Claude can decline a request (a `refusal`). A judge batch that's declined is split in half and
  retried until the one post it objects to is found; that post is filed as 0 ("declined by the
  model") so it isn't retried forever.

#### Judging through a Console-built agent (optional)

With `NSA_PROVIDER=anthropic`, judge batches can run on a **Managed Agent** you build in the Claude
Console instead of direct API calls. Roasts, dossiers and chat replies stay on direct calls.

1. Console → **Agents → Environments → New**: type cloud, networking **limited** with nothing allowed
   (the judge reads the batch and answers; it never touches the network). Copy the `env_...` ID.
2. Console → **Agents → New agent**: name it (e.g. "NSA judge"), pick the model (`claude-haiku-5-5`)
   and effort, and give it **no tools, MCP servers or skills**. Leave the system prompt empty: the bot
   writes its judge prompt into the agent on startup and keeps it in sync on every deploy (a new agent
   version only when the prompt changed). Copy the `agent_...` ID.
3. `.env`: `NSA_AGENT_ID=agent_...`, `NSA_ENVIRONMENT_ID=env_...`, then restart.

Each batch becomes one session with a hard spend cap (`NSA_AGENT_BUDGET_USD`, default $0.25), and is
deleted once judged unless `NSA_AGENT_KEEP_SESSIONS=on`; failed sessions are kept so you can open them
in the Console's session viewer. Sessions add $0.08 per session-hour of run time on top of tokens
(seconds per batch). Server notes and calibration examples are sent with each batch rather than in the
agent's prompt. `/model` shows the agent in use. The agent runs the same Claude model, so judging
quality is the model's: Haiku 5.5 measured below DeepSeek on `/evaluate`.

To go back to DeepSeek, set `NSA_PROVIDER=deepseek` and `DEEPSEEK_API_KEY` (from
<https://platform.deepseek.com>) and restart; the `DEEPSEEK_*` settings still apply there.

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
