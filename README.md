# nsa-bot

Neckbeard Surveillance Agency. A Discord bot that scrapes channels, has a DeepSeek agent read every
post and score how **kimoi** (キモい) it is, then ranks the server's otaku by frequency and severity.

## Commands

| Command | Who | What |
|---|---|---|
| `!scan [#ch ...]` | Manage Server | Scrape new messages in the given channels (default: current one) and judge them |
| `!scanall` | Manage Server | Same, for every text channel the bot can read |
| `!kimoiboard` | anyone | Top 10 leaderboard |
| `!kimoi [@user]` | anyone | Rank, stats and worst posts (with jump links) |
| `!dossier [@user]` | anyone | DeepSeek writes a classified report roasting the user's worst posts |
| `!optout` / `!optin` | anyone | Leave the rankings (deletes your stored posts) / rejoin |

Scans are incremental: each channel keeps a cursor, so rerunning `!scan` only reads new messages.
A scan reads at most `NSA_SCAN_LIMIT` messages per channel (oldest first), so run it again to keep
crawling a long history.

## Scoring

DeepSeek reads messages in batches of 40 (with surrounding context) and gives each a severity of
0-10. A user's score is **Σ severity² / 10**: every kimoi post counts, but one 10/10 post (10 pts)
beats ten 3/10 posts (9 pts). The board also shows hit rate (kimoi posts ÷ judged posts) and average
severity. The rubric is in `nsabot/judge.py` (`JUDGE_PROMPT`); edit it to fit your server.

## Setup

1. Create a bot at <https://discord.com/developers/applications>, enable the **Message Content**
   intent, and invite it with Read Messages, Read Message History, Send Messages, Embed Links.
2. Get a DeepSeek API key at <https://platform.deepseek.com>.
3. Run it:

```sh
cp .env.example .env   # fill in DISCORD_TOKEN and DEEPSEEK_API_KEY
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m nsabot.bot
```

Tests: `.venv/bin/pip install pytest && .venv/bin/python -m pytest`
