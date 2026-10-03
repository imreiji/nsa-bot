"""SQLite storage for scraped messages, DeepSeek verdicts, scan cursors and opt-outs."""

import json
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,   -- Discord message id
    guild_id    INTEGER NOT NULL,
    channel_id  INTEGER NOT NULL,
    author_id   INTEGER NOT NULL,
    author_name TEXT NOT NULL,
    content     TEXT NOT NULL,
    severity    INTEGER,               -- NULL = not judged yet, 0 = clean, 1-10 = kimoi
    reason      TEXT,
    reported    INTEGER NOT NULL DEFAULT 0,
    scored      INTEGER NOT NULL DEFAULT 1,  -- 0 = stored only as context for its neighbours
    reply_to_id INTEGER,
    reply_author_id INTEGER,
    reply_author    TEXT,
    reply_text      TEXT,
    extras      TEXT,                  -- attachments, stickers, embeds, forwards, as text
    deleted     INTEGER NOT NULL DEFAULT 0, -- deleted by its author (VAR reviewed)
    labels      TEXT,                  -- JSON labels from the judge (formula scoring)
    rubric_version INTEGER             -- rubric the verdict was made under (NULL = v1, number scores)
);
CREATE INDEX IF NOT EXISTS idx_messages_unjudged ON messages (guild_id, severity);
CREATE INDEX IF NOT EXISTS idx_messages_author ON messages (guild_id, author_id);
CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages (channel_id, id);

CREATE TABLE IF NOT EXISTS cursors (
    channel_id INTEGER PRIMARY KEY,
    last_id    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS watch (
    guild_id          INTEGER PRIMARY KEY,
    report_channel_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS usage (
    day    TEXT PRIMARY KEY,           -- UTC date
    tokens INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS gold (
    message_id INTEGER PRIMARY KEY,    -- an admin's own 0-10 score for a post (/calibrate)
    guild_id   INTEGER NOT NULL,
    admin_id   INTEGER NOT NULL,
    score      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS optouts (
    guild_id INTEGER NOT NULL,
    user_id  INTEGER NOT NULL,
    PRIMARY KEY (guild_id, user_id)
);
"""

# Kimoi points: severity squared / 10, so one 10/10 post (10 pts) outweighs ten 3/10 posts (9 pts)
# while frequency still accumulates.
POINTS = "SUM(severity * severity) / 10.0"

# Columns added after the first release, migrated onto existing databases.
MIGRATIONS = {
    "reported": "INTEGER NOT NULL DEFAULT 0",
    "scored": "INTEGER NOT NULL DEFAULT 1",
    "reply_to_id": "INTEGER",
    "reply_author_id": "INTEGER",
    "reply_author": "TEXT",
    "reply_text": "TEXT",
    "extras": "TEXT",
    "deleted": "INTEGER NOT NULL DEFAULT 0",
    "labels": "TEXT",
    "rubric_version": "INTEGER",
}

INSERT = (
    "INSERT OR IGNORE INTO messages (id, guild_id, channel_id, author_id, author_name, content, scored,"
    " reply_to_id, reply_author_id, reply_author, reply_text, extras) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

# A message plus whatever it replies to; falls back to the stored target when the reply wasn't resolved.
TIMELINE = (
    "SELECT m.id, m.author_name, m.content, m.extras,"
    " COALESCE(m.reply_author, r.author_name) AS reply_author,"
    " COALESCE(m.reply_text, r.content) AS reply_text"
    " FROM messages m LEFT JOIN messages r ON r.id = m.reply_to_id"
)

# Same, plus the verdict columns the API returns.
API_SELECT = TIMELINE.replace(
    "SELECT m.id,", "SELECT m.id, m.guild_id, m.channel_id, m.author_id, m.severity, m.reason, m.labels, m.deleted,"
)


@dataclass
class Message:
    id: int
    guild_id: int
    channel_id: int
    author_id: int
    author_name: str
    content: str
    scored: bool = True
    reply_to_id: int | None = None
    reply_author_id: int | None = None
    reply_author: str | None = None
    reply_text: str | None = None
    extras: str | None = None

    def params(self) -> tuple:
        return (
            self.id, self.guild_id, self.channel_id, self.author_id, self.author_name, self.content,
            int(self.scored), self.reply_to_id, self.reply_author_id, self.reply_author, self.reply_text,
            self.extras,
        )


@dataclass
class Standing:
    author_id: int
    author_name: str
    points: float
    hits: int
    judged: int
    avg_severity: float

    @property
    def rate(self) -> float:
        return self.hits / self.judged if self.judged else 0.0


class DB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(messages)")}
        for name, decl in MIGRATIONS.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE messages ADD COLUMN {name} {decl}")
        self.conn.commit()

    # --- scraping -----------------------------------------------------------

    def get_cursor(self, channel_id: int) -> int | None:
        row = self.conn.execute("SELECT last_id FROM cursors WHERE channel_id = ?", (channel_id,)).fetchone()
        return row["last_id"] if row else None

    def save_batch(self, channel_id: int, last_id: int, messages: list[Message]) -> None:
        with self.conn:
            self.conn.executemany(INSERT, [m.params() for m in messages])
            self.conn.execute(
                "INSERT INTO cursors (channel_id, last_id) VALUES (?, ?)"
                " ON CONFLICT (channel_id) DO UPDATE SET last_id = excluded.last_id",
                (channel_id, last_id),
            )

    def save_message(self, m: Message) -> None:
        with self.conn:
            self.conn.execute(INSERT, m.params())

    # --- judging ------------------------------------------------------------

    def unjudged(self, guild_id: int, limit: int = -1) -> list[sqlite3.Row]:
        """Queued messages to score, grouped by channel in order. limit -1 = all."""
        return self.conn.execute(
            "SELECT id, channel_id FROM messages"
            " WHERE guild_id = ? AND scored = 1 AND severity IS NULL ORDER BY channel_id, id LIMIT ?",
            (guild_id, limit),
        ).fetchall()

    def count_unjudged(self, guild_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE guild_id = ? AND scored = 1 AND severity IS NULL", (guild_id,)
        ).fetchone()[0]

    def timeline(self, channel_id: int, first_id: int, last_id: int, before: int) -> list[sqlite3.Row]:
        """Everything stored in a channel from first_id to last_id, plus `before` earlier messages."""
        earlier = self.conn.execute(
            TIMELINE + " WHERE m.channel_id = ? AND m.id < ? ORDER BY m.id DESC LIMIT ?",
            (channel_id, first_id, before),
        ).fetchall()
        window = self.conn.execute(
            TIMELINE + " WHERE m.channel_id = ? AND m.id BETWEEN ? AND ? ORDER BY m.id",
            (channel_id, first_id, last_id),
        ).fetchall()
        return earlier[::-1] + window

    def timeline_after(self, channel_id: int, after_id: int, limit: int) -> list[sqlite3.Row]:
        """The next `limit` stored messages after after_id (how people reacted)."""
        return self.conn.execute(
            TIMELINE + " WHERE m.channel_id = ? AND m.id > ? ORDER BY m.id LIMIT ?", (channel_id, after_id, limit)
        ).fetchall()

    def count_between(self, channel_id: int, first_id: int, last_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE channel_id = ? AND id BETWEEN ? AND ?", (channel_id, first_id, last_id)
        ).fetchone()[0]

    def get_message(self, message_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()

    def mark_deleted(self, message_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE messages SET deleted = 1 WHERE id = ?", (message_id,))

    def save_verdicts(self, verdicts: list[tuple], rubric_version: int | None = None) -> None:
        """verdicts: (message_id, severity, reason) or (message_id, severity, reason, labels dict)."""
        rows = []
        for v in verdicts:
            mid, sev, reason, labels = (*v, None)[:4]
            rows.append((sev, reason, json.dumps(labels, ensure_ascii=False) if labels else None, rubric_version, mid))
        with self.conn:
            self.conn.executemany(
                "UPDATE messages SET severity = ?, reason = ?, labels = ?, rubric_version = ? WHERE id = ?", rows
            )

    # --- formula scoring / calibration --------------------------------------

    def recompute_scores(self, score_fn) -> int:
        """Re-apply the formula to stored labels (after weights change). Returns how many changed."""
        changed = []
        for r in self.conn.execute("SELECT id, severity, labels FROM messages WHERE labels IS NOT NULL"):
            new = score_fn(json.loads(r["labels"]))
            if new != r["severity"]:
                changed.append((new, r["id"]))
        with self.conn:
            self.conn.executemany("UPDATE messages SET severity = ? WHERE id = ?", changed)
        return len(changed)

    def count_outdated(self, guild_id: int, version: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE guild_id = ? AND scored = 1 AND severity IS NOT NULL"
            " AND COALESCE(rubric_version, 1) < ?",
            (guild_id, version),
        ).fetchone()[0]

    def queue_rescore(self, guild_id: int, version: int) -> int:
        """Send posts judged under an older rubric back to the queue. Old history isn't re-reported."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE messages SET severity = NULL, reason = NULL, labels = NULL,"
                " reported = CASE WHEN deleted = 1 THEN reported ELSE 1 END"
                " WHERE guild_id = ? AND scored = 1 AND severity IS NOT NULL AND COALESCE(rubric_version, 1) < ?",
                (guild_id, version),
            )
        return cur.rowcount

    def calibration_sample(self, guild_id: int, flagged: bool) -> sqlite3.Row | None:
        """A random judged post no admin has scored yet: a flagged one, or any one."""
        return self.conn.execute(
            "SELECT * FROM messages WHERE guild_id = ? AND scored = 1 AND severity IS NOT NULL AND deleted = 0"
            + (" AND severity > 0" if flagged else "")
            + " AND id NOT IN (SELECT message_id FROM gold) ORDER BY RANDOM() LIMIT 1",
            (guild_id,),
        ).fetchone()

    def save_gold(self, guild_id: int, message_id: int, admin_id: int, score: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO gold VALUES (?, ?, ?, ?) ON CONFLICT (message_id)"
                " DO UPDATE SET admin_id = excluded.admin_id, score = excluded.score",
                (message_id, guild_id, admin_id, score),
            )

    def gold_rows(self, guild_id: int) -> list[sqlite3.Row]:
        """Admin scores joined with the bot's current verdicts for the same posts."""
        return self.conn.execute(
            "SELECT g.score AS admin_score, m.id, m.channel_id, m.content, m.severity, m.labels, m.rubric_version"
            " FROM gold g JOIN messages m ON m.id = g.message_id WHERE g.guild_id = ? ORDER BY m.id DESC",
            (guild_id,),
        ).fetchall()

    def unreported(self, guild_id: int, min_severity: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, channel_id, author_id, author_name, content, severity, reason, labels FROM messages"
            " WHERE guild_id = ? AND reported = 0 AND severity >= ? ORDER BY id",
            (guild_id, min_severity),
        ).fetchall()

    def mark_reported(self, ids: list[int]) -> None:
        with self.conn:
            self.conn.executemany("UPDATE messages SET reported = 1 WHERE id = ?", [(i,) for i in ids])

    # --- live watch ---------------------------------------------------------

    def watched(self) -> dict[int, int]:
        """{guild_id: report_channel_id} for every guild with live watching on."""
        return {r["guild_id"]: r["report_channel_id"] for r in self.conn.execute("SELECT * FROM watch")}

    def set_watch(self, guild_id: int, report_channel_id: int | None) -> None:
        with self.conn:
            if report_channel_id is None:
                self.conn.execute("DELETE FROM watch WHERE guild_id = ?", (guild_id,))
            else:
                self.conn.execute(
                    "INSERT INTO watch VALUES (?, ?) ON CONFLICT (guild_id)"
                    " DO UPDATE SET report_channel_id = excluded.report_channel_id",
                    (guild_id, report_channel_id),
                )

    # --- API usage (informational) ---------------------------------------------------------

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def tokens_today(self) -> int:
        row = self.conn.execute("SELECT tokens FROM usage WHERE day = ?", (self._today(),)).fetchone()
        return row["tokens"] if row else 0

    def add_tokens(self, n: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO usage VALUES (?, ?) ON CONFLICT (day) DO UPDATE SET tokens = tokens + excluded.tokens",
                (self._today(), n),
            )

    # --- ranking ------------------------------------------------------------

    def _standings_query(self, extra_where: str = "") -> str:
        return (
            f"SELECT author_id, MAX(author_name) AS author_name, {POINTS} AS points,"
            " SUM(severity > 0) AS hits, COUNT(*) AS judged,"
            " COALESCE(AVG(CASE WHEN severity > 0 THEN severity END), 0) AS avg_severity"
            " FROM messages WHERE guild_id = ? AND severity IS NOT NULL" + extra_where +
            " GROUP BY author_id HAVING hits > 0 ORDER BY points DESC, hits DESC"
        )

    def leaderboard(self, guild_id: int, limit: int = 10) -> list[Standing]:
        rows = self.conn.execute(self._standings_query() + " LIMIT ?", (guild_id, limit)).fetchall()
        return [Standing(**dict(r)) for r in rows]

    def standing(self, guild_id: int, user_id: int) -> tuple[int, Standing] | None:
        """Returns (rank, standing) for one user, rank starting at 1."""
        for rank, row in enumerate(self.conn.execute(self._standings_query(), (guild_id,)), start=1):
            if row["author_id"] == user_id:
                return rank, Standing(**dict(row))
        return None

    def recent_posts(self, guild_id: int, user_id: int, limit: int = 40) -> list[sqlite3.Row]:
        """Someone's latest real posts (not reactions), newest first, flagged or not."""
        return self.conn.execute(
            "SELECT content, severity FROM messages WHERE guild_id = ? AND author_id = ? AND scored = 1"
            " AND (labels IS NULL OR labels NOT LIKE '%\"distress\": true%')"  # never roast someone's distress
            " ORDER BY id DESC LIMIT ?",
            (guild_id, user_id, limit),
        ).fetchall()

    def worst_posts(self, guild_id: int, user_id: int, limit: int = 5) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, channel_id, content, severity, reason, labels FROM messages"
            " WHERE guild_id = ? AND author_id = ? AND severity > 0"
            " ORDER BY severity DESC, id DESC LIMIT ?",
            (guild_id, user_id, limit),
        ).fetchall()

    def api_posts(
        self, guild_id: int, *, min_severity: int = 1, author_id: int | None = None, channel_id: int | None = None,
        behaviour: str | None = None, order: str = "severity", cursor: tuple[int, int] | None = None, limit: int = 50,
    ) -> list[sqlite3.Row]:
        """Flagged posts for the API, keyset-paginated. cursor = (severity, id) of the last row seen."""
        where, args = ["m.guild_id = ?", "m.severity >= ?"], [guild_id, max(1, min_severity)]
        if author_id:
            where.append("m.author_id = ?")
            args.append(author_id)
        if channel_id:
            where.append("m.channel_id = ?")
            args.append(channel_id)
        if behaviour:  # validated by the caller against the known behaviour names
            where.append("m.labels LIKE ?")
            args.append(f'%"{behaviour}"%')
        if cursor and order == "severity":
            where.append("(m.severity < ? OR (m.severity = ? AND m.id < ?))")
            args += [cursor[0], cursor[0], cursor[1]]
        elif cursor:
            where.append("m.id < ?")
            args.append(cursor[1])
        sort = "m.severity DESC, m.id DESC" if order == "severity" else "m.id DESC"
        return self.conn.execute(
            API_SELECT + " WHERE " + " AND ".join(where) + f" ORDER BY {sort} LIMIT ?", (*args, limit)
        ).fetchall()

    def api_post(self, message_id: int) -> sqlite3.Row | None:
        return self.conn.execute(API_SELECT + " WHERE m.id = ?", (message_id,)).fetchone()

    def kimoi_posts(
        self, guild_id: int, user_id: int | None = None, offset: int = 0, limit: int = 10
    ) -> tuple[int, list[sqlite3.Row]]:
        """All flagged posts, most kimoi first (newest first within a score). Returns (total, page)."""
        where = "guild_id = ? AND severity > 0" + (" AND author_id = ?" if user_id else "")
        args = (guild_id, user_id) if user_id else (guild_id,)
        total = self.conn.execute(f"SELECT COUNT(*) FROM messages WHERE {where}", args).fetchone()[0]
        rows = self.conn.execute(
            f"SELECT id, channel_id, author_name, content, severity, reason, deleted, labels FROM messages WHERE {where}"
            " ORDER BY severity DESC, id DESC LIMIT ? OFFSET ?",
            (*args, limit, offset),
        ).fetchall()
        return total, rows

    def author_names(self, guild_id: int, prefix: str, limit: int = 25) -> list[sqlite3.Row]:
        """Everyone on file whose latest name starts with (or contains) prefix, for slash autocomplete."""
        like = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return self.conn.execute(
            "SELECT author_id, author_name FROM messages WHERE id IN"
            " (SELECT MAX(id) FROM messages WHERE guild_id = ? GROUP BY author_id)"
            " AND author_name LIKE ? ESCAPE '\\'"
            " ORDER BY author_name NOT LIKE ? ESCAPE '\\', author_name COLLATE NOCASE LIMIT ?",
            (guild_id, f"%{like}%", f"{like}%", limit),
        ).fetchall()

    def find_author(self, guild_id: int, query: str) -> sqlite3.Row | None:
        """Someone on file by ID, mention or the name they last posted under (for people who left)."""
        digits = query.strip("<@!>")
        if digits.isdigit():
            sql, arg = "author_id = ?", int(digits)
        else:
            sql, arg = "author_name = ? COLLATE NOCASE", query.lstrip("@")
        return self.conn.execute(
            f"SELECT author_id, author_name FROM messages WHERE guild_id = ? AND {sql} ORDER BY id DESC LIMIT 1",
            (guild_id, arg),
        ).fetchone()

    # --- opt-out ------------------------------------------------------------

    def opted_out(self, guild_id: int) -> set[int]:
        rows = self.conn.execute("SELECT user_id FROM optouts WHERE guild_id = ?", (guild_id,))
        return {r["user_id"] for r in rows}

    def opt_out(self, guild_id: int, user_id: int) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO optouts VALUES (?, ?)", (guild_id, user_id))
            self.conn.execute("DELETE FROM messages WHERE guild_id = ? AND author_id = ?", (guild_id, user_id))
            self.conn.execute(
                "UPDATE messages SET reply_author = NULL, reply_text = NULL"
                " WHERE guild_id = ? AND reply_author_id = ?",
                (guild_id, user_id),
            )

    def opt_in(self, guild_id: int, user_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM optouts WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
