"""SQLite storage for scraped messages, DeepSeek verdicts, scan cursors and opt-outs."""

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
    reported    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_unjudged ON messages (guild_id, severity);
CREATE INDEX IF NOT EXISTS idx_messages_author ON messages (guild_id, author_id);

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

CREATE TABLE IF NOT EXISTS optouts (
    guild_id INTEGER NOT NULL,
    user_id  INTEGER NOT NULL,
    PRIMARY KEY (guild_id, user_id)
);
"""

# Kimoi points: severity squared / 10, so one 10/10 post (10 pts) outweighs ten 3/10 posts (9 pts)
# while frequency still accumulates.
POINTS = "SUM(severity * severity) / 10.0"


@dataclass
class Message:
    id: int
    guild_id: int
    channel_id: int
    author_id: int
    author_name: str
    content: str


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
        if "reported" not in cols:  # databases created before live reporting existed
            self.conn.execute("ALTER TABLE messages ADD COLUMN reported INTEGER NOT NULL DEFAULT 0")

    # --- scraping -----------------------------------------------------------

    def get_cursor(self, channel_id: int) -> int | None:
        row = self.conn.execute("SELECT last_id FROM cursors WHERE channel_id = ?", (channel_id,)).fetchone()
        return row["last_id"] if row else None

    def save_batch(self, channel_id: int, last_id: int, messages: list[Message]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO messages (id, guild_id, channel_id, author_id, author_name, content)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(m.id, m.guild_id, m.channel_id, m.author_id, m.author_name, m.content) for m in messages],
            )
            self.conn.execute(
                "INSERT INTO cursors (channel_id, last_id) VALUES (?, ?)"
                " ON CONFLICT (channel_id) DO UPDATE SET last_id = excluded.last_id",
                (channel_id, last_id),
            )

    def save_message(self, m: Message) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO messages (id, guild_id, channel_id, author_id, author_name, content)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (m.id, m.guild_id, m.channel_id, m.author_id, m.author_name, m.content),
            )

    # --- judging ------------------------------------------------------------

    def unjudged(self, guild_id: int, limit: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, author_name, content FROM messages"
            " WHERE guild_id = ? AND severity IS NULL ORDER BY channel_id, id LIMIT ?",
            (guild_id, limit),
        ).fetchall()

    def count_unjudged(self, guild_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE guild_id = ? AND severity IS NULL", (guild_id,)
        ).fetchone()[0]

    def save_verdicts(self, verdicts: list[tuple[int, int, str | None]]) -> None:
        """verdicts: (message_id, severity, reason)."""
        with self.conn:
            self.conn.executemany(
                "UPDATE messages SET severity = ?, reason = ? WHERE id = ?",
                [(sev, reason, mid) for mid, sev, reason in verdicts],
            )

    def unreported(self, guild_id: int, min_severity: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, channel_id, author_id, author_name, content, severity, reason FROM messages"
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

    def worst_posts(self, guild_id: int, user_id: int, limit: int = 5) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, channel_id, content, severity, reason FROM messages"
            " WHERE guild_id = ? AND author_id = ? AND severity > 0"
            " ORDER BY severity DESC, id DESC LIMIT ?",
            (guild_id, user_id, limit),
        ).fetchall()

    # --- opt-out ------------------------------------------------------------

    def opted_out(self, guild_id: int) -> set[int]:
        rows = self.conn.execute("SELECT user_id FROM optouts WHERE guild_id = ?", (guild_id,))
        return {r["user_id"] for r in rows}

    def opt_out(self, guild_id: int, user_id: int) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO optouts VALUES (?, ?)", (guild_id, user_id))
            self.conn.execute("DELETE FROM messages WHERE guild_id = ? AND author_id = ?", (guild_id, user_id))

    def opt_in(self, guild_id: int, user_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM optouts WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
