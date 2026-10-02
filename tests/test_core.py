import json

import pytest

from nsabot.db import DB, Message
from nsabot.judge import parse_verdicts


def msg(i, author, text="hi"):
    return Message(i, 1, 10, author, f"user{author}", text)


def test_ranking_weights_severity_over_frequency():
    db = DB(":memory:")
    db.save_batch(10, 7, [msg(i, author=1 if i <= 5 else 2) for i in range(1, 8)])
    # user1: four 3/10 posts + one clean; user2: one 10/10 post + one clean
    db.save_verdicts([(i, 3, "mild") for i in range(1, 5)] + [(5, 0, None), (6, 10, "fbi"), (7, 0, None)])

    board = db.leaderboard(1)
    assert [s.author_id for s in board] == [2, 1]
    assert board[0].points == 10.0 and board[0].rate == 0.5 and board[1].rate == 0.8
    assert board[1].hits == 4 and board[1].points == pytest.approx(3.6)
    assert db.standing(1, 1)[0] == 2
    assert db.get_cursor(10) == 7
    assert db.count_unjudged(1) == 0


def test_optout_deletes_and_excludes():
    db = DB(":memory:")
    db.save_batch(10, 2, [msg(1, 1), msg(2, 2)])
    db.opt_out(1, 1)
    assert db.opted_out(1) == {1}
    assert [r["id"] for r in db.unjudged(1, 10)] == [2]
    db.opt_in(1, 1)
    assert db.opted_out(1) == set()


def test_parse_verdicts_clamps_and_drops_garbage():
    raw = json.dumps({"flagged": [
        {"i": 0, "severity": 14, "reason": "uooh"},
        {"i": 1, "severity": 0},
        {"i": 9, "severity": 5},
        {"i": "x", "severity": 5},
        {"severity": 5},
    ]})
    assert parse_verdicts(raw, 3) == {0: (10, "uooh")}
    with pytest.raises(json.JSONDecodeError):
        parse_verdicts("not json", 3)
