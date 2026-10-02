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


def test_reporting_watch_and_usage():
    db = DB(":memory:")
    db.save_message(msg(1, 1))
    db.save_message(msg(2, 1))
    db.save_message(msg(2, 1))  # duplicate from live + scan is ignored
    assert db.count_unjudged(1) == 2
    db.save_verdicts([(1, 7, "uooh"), (2, 2, "mild")])
    assert [r["id"] for r in db.unreported(1, 5)] == [1]
    db.mark_reported([1])
    assert db.unreported(1, 5) == []

    db.set_watch(1, 99)
    assert db.watched() == {1: 99}
    db.set_watch(1, None)
    assert db.watched() == {}

    db.add_tokens(100)
    db.add_tokens(50)
    assert db.tokens_today() == 150


def test_judge_records_usage():
    import asyncio
    from types import SimpleNamespace

    from nsabot.judge import Judge

    db = DB(":memory:")
    judge = Judge("key", "model", "http://localhost", db)

    async def fake_create(**kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=600),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"flagged": [{"i": 0, "severity": 6}]}'))],
        )

    judge.client.chat.completions.create = fake_create
    payload = {"channel": {"name": "#x"}, "messages": [{"i": 0, "author": "a", "text": "uooh"}]}
    assert asyncio.run(judge.judge(payload, 1)) == {0: (6, "")}
    asyncio.run(judge.judge(payload, 1))
    assert db.tokens_today() == 1200


def test_possessive_copypasta_is_exact():
    from nsabot.bot import POSSESSIVE

    assert POSSESSIVE == (
        "If my experiences hadn’t included the IRL possessive nature and entitlement of seiyuu and also being spit "
        "on when someone cheered their name, I would think differently but instead, half the content here includes "
        "poorly socialized people with their masturbatory fantasies about seiyuu that are just acting, not actually "
        "interested.\nCreepy and possessive."
    )
