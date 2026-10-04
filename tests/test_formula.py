"""Formula scoring, calibration in DMs, rescoring, /model and /scoring."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

import nsabot.bot as b
from nsabot import scoring
from nsabot.db import DB, Message
from nsabot.judge import Verdict, parse_verdicts

GUILD, ADMIN = 10, 1


# --- the formula --------------------------------------------------------------

@pytest.mark.parametrize("labels,expected", [
    (dict(behaviours=["spending"], target="real", sincerity="sincere"), 6),
    (dict(behaviours=["life_impact"], target="real", intensity="graphic", sincerity="sincere"), 10),
    (dict(behaviours=["bodily_servitude"], target="real", sincerity="bit"), 6),
    (dict(behaviours=["bodily_servitude"], target="real", sincerity="sincere"), 9),
    (dict(behaviours=["unicorn"], target="real", about_someone_else=True), 0),
    (dict(behaviours=["horny"], target="minor", sincerity="ambiguous", intensity="clear"), 10),
    (dict(behaviours=["horny"], target="minor", sincerity="bit", intensity="passing"), 2),  # passing joke: no auto-10
    (dict(behaviours=["life_impact"], target="real", sincerity="sincere", distress=True), 0),
    (dict(behaviours=["stalking_harassment"], target="real", sincerity="sincere"), 9),  # 7 + real 1 + sincere 1
    (dict(behaviours=["worship"], target="character", intensity="passing", sincerity="bit"), 1),  # floor
    (dict(behaviours=["stalking_harassment", "unicorn"], target="real", intensity="graphic",
          sincerity="sincere", doubling_down=True), 10),  # ceiling
    (dict(behaviours=["gachikoi", "spending"], target="real"), 6),  # 4 + real 1 + two behaviours 1
    (dict(behaviours=[]), 0),
    (dict(behaviours=["life_impact"], target="real", intensity="graphic", sincerity="sincere", spiral=True), 8),  # cap
    (dict(behaviours=["life_impact"], target="real", sincerity="bit", spiral=True), 6),  # under the cap: unchanged
])
def test_formula(labels, expected):
    assert scoring.score(labels) == expected


def test_messy_labels_fall_back_to_neutral():
    assert scoring.clean({"behaviours": "UNICORN", "target": "alien", "intensity": 7, "doubling_down": "yes"}) == {
        "behaviours": ["unicorn"], "target": "none", "intensity": "clear", "sincerity": "ambiguous",
        "doubling_down": False, "about_someone_else": False, "distress": False, "spiral": False}
    assert scoring.describe({"behaviours": ["bodily_servitude"], "target": "real", "sincerity": "sincere"}) \
        == "bodily/servitude · real person · sincere"


def test_formula_text_matches_the_tables():
    text = "\n".join(scoring.formula_lines())
    assert f"stalking/harassment {scoring.BASE['stalking_harassment']}" in text
    assert f"obvious bit {scoring.SINCERITY['bit']}" in text and "always **10**" in text
    assert f"at most **{scoring.SPIRAL_CAP}**" in text


def test_parse_labels_into_formula_scores():
    raw = json.dumps({"flagged": [
        {"i": 0, "behaviours": ["unicorn"], "target": "real", "sincerity": "sincere", "reason": "purity police"},
        {"i": 1, "behaviours": ["unicorn"], "target": "real", "about_someone_else": True, "reason": "just asking"},
        {"i": 2, "severity": 9, "behaviours": ["worship"], "intensity": "passing", "sincerity": "bit"},
    ]})
    out = parse_verdicts(raw, 3)
    assert out[0].severity == 8 and out[0].labels["behaviours"] == ["unicorn"]
    assert 1 not in out  # pointing at someone else's kimoi is 0, so not flagged
    assert out[2].severity == 1  # the model's own number is ignored when it gives labels


# --- pipeline -----------------------------------------------------------------

def fake_api(monkeypatch, flagged_for):
    sent = []

    async def create(**kwargs):
        sent.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        flagged = [dict(i=m["i"], **flagged_for(m["text"])) for m in payload["messages"] if "i" in m and flagged_for(m["text"])]
        return NS(usage=None, choices=[NS(finish_reason="stop", message=NS(content=json.dumps({"flagged": flagged})))])

    monkeypatch.setattr(b.judge.client.chat.completions, "create", create)
    return sent


def guild():
    return NS(id=GUILD, get_channel_or_thread=lambda _: NS(name="general", topic=None, is_nsfw=lambda: False),
              get_channel=lambda _: None)


def seed(rows):
    b.db.conn.execute("DELETE FROM messages")
    b.db.conn.execute("DELETE FROM gold")
    b.anchor_cache.clear()
    b.db.save_batch(50, max(r[0] for r in rows), [Message(i, GUILD, 50, 5, "yargas", text) for i, text in rows])


UNICORN = dict(behaviours=["unicorn"], target="real", sincerity="sincere", reason="purity police", evidence="boyfriend")


def test_judging_stores_labels_and_rubric_version(monkeypatch):
    fake_api(monkeypatch, lambda t: UNICORN if "boyfriend" in t else None)
    seed([(1, "she has a boyfriend?? betrayal"), (2, "good morning")])
    asyncio.run(b.judge_backlog(guild()))
    hit, clean = b.db.get_message(1), b.db.get_message(2)
    assert hit["severity"] == 8 and json.loads(hit["labels"])["behaviours"] == ["unicorn"]
    assert hit["rubric_version"] == clean["rubric_version"] == scoring.RUBRIC_VERSION and clean["severity"] == 0
    assert b.tags(hit["labels"]) == "unicorn · real person · sincere"


def test_admin_scores_become_prompt_examples(monkeypatch):
    sent = fake_api(monkeypatch, lambda t: UNICORN if "boyfriend" in t else None)
    seed([(1, "she has a boyfriend?? betrayal"), (2, "good morning"), (3, "another boyfriend post")])
    asyncio.run(b.judge_backlog(guild()))
    b.db.save_gold(GUILD, 1, ADMIN, 8)   # agrees with the formula (8): becomes an example
    b.db.save_gold(GUILD, 2, ADMIN, 0)   # agrees (0): a "not kimoi" example
    b.db.save_gold(GUILD, 3, ADMIN, 2)   # disagrees by 6: left out
    b.anchor_cache.clear()
    anchors = b.anchors_for(GUILD)
    assert '"she has a boyfriend?? betrayal" -> {"behaviours": ["unicorn"]' in anchors and "(admins: 8)" in anchors
    assert '"good morning" -> not kimoi (admins: 0)' in anchors and "another boyfriend" not in anchors
    b.db.save_batch(50, 4, [Message(4, GUILD, 50, 5, "yargas", "new post")])
    asyncio.run(b.judge_backlog(guild()))
    assert sent[-1]["messages"][0]["content"].endswith(anchors)  # appended to the system prompt


def test_recompute_applies_new_weights(monkeypatch):
    db = DB(":memory:")
    db.save_batch(50, 1, [Message(1, GUILD, 50, 5, "u", "x")])
    db.save_verdicts([(1, 8, "r", UNICORN)], scoring.RUBRIC_VERSION)
    assert db.recompute_scores(scoring.score) == 0
    monkeypatch.setitem(scoring.BASE, "unicorn", 3)
    assert db.recompute_scores(scoring.score) == 1 and db.get_message(1)["severity"] == 5


def test_rescore_requeues_old_rubric_without_reposting():
    db = DB(":memory:")
    db.save_batch(50, 3, [Message(i, GUILD, 50, 5, "u", f"p{i}") for i in (1, 2, 3)])
    db.save_verdicts([(1, 7, "old"), (2, 0, None)])                       # v1 (number scores)
    db.save_verdicts([(3, 8, "new", UNICORN)], scoring.RUBRIC_VERSION)    # already current
    assert db.count_outdated(GUILD, scoring.RUBRIC_VERSION) == 2
    assert db.queue_rescore(GUILD, scoring.RUBRIC_VERSION) == 2
    assert db.count_unjudged(GUILD) == 2 and db.get_message(3)["severity"] == 8
    db.save_verdicts([(1, 9, "rescored", UNICORN)], scoring.RUBRIC_VERSION)
    assert [r["id"] for r in db.unreported(GUILD, 5)] == [3]  # 1 is old history: not re-posted


def test_flagged_rescore_only_requeues_flagged_and_distress_posts():
    db = DB(":memory:")
    db.save_batch(50, 5, [Message(i, GUILD, 50, 5, "u", f"p{i}") for i in range(1, 6)])
    db.save_verdicts([(1, 7, "old"), (2, 0, None)])                       # v1: flagged, clean
    db.save_verdicts([(3, 8, "new", UNICORN), (4, 0, None),
                      (5, 0, "distress", {"distress": True})], scoring.RUBRIC_VERSION)
    assert db.count_flagged(GUILD) == 3
    assert db.queue_rescore_flagged(GUILD) == 3
    assert [r["id"] for r in db.unjudged(GUILD)] == [1, 3, 5]  # clean posts keep their verdicts
    db.save_verdicts([(1, 6, "still", UNICORN), (3, 8, "still", UNICORN)], scoring.RUBRIC_VERSION)
    assert db.unreported(GUILD, 5) == []  # already-known posts aren't re-posted


def test_flagged_rescore_batches_scattered_posts_with_a_short_window(monkeypatch):
    sent = fake_api(monkeypatch, lambda t: UNICORN if "boyfriend" in t else None)
    seed([(i, "boyfriend" if i in (1, 400) else "chat") for i in range(1, 401)])
    b.db.save_verdicts([(i, 8 if i in (1, 400) else 0, "r", UNICORN if i in (1, 400) else None)
                        for i in range(1, 401)], scoring.RUBRIC_VERSION)
    b.db.queue_rescore_flagged(GUILD)
    asyncio.run(b.judge_backlog(guild(), None, b.evaluate.MAX_WINDOW))
    assert len(sent) == 2  # two calls with their own context, not one call over all 400 messages
    assert all(len(json.loads(c["messages"][1]["content"])["messages"]) < 30 for c in sent)


# --- calibration in DMs ----------------------------------------------------------

class DM:
    def __init__(self):
        self.sent = []

    async def send(self, content=None, embed=None, view=None):
        self.sent.append(NS(content=content, embed=embed, view=view))


def click(view, label, user=ADMIN, channel=None):
    edits = []

    async def edit_message(**kw):
        edits.append(kw)

    message = NS(embeds=[view_embed(view)])
    it = NS(user=NS(id=user), channel=channel, message=message, response=NS(edit_message=edit_message))
    button = next(c for c in view.children if c.label == label)
    asyncio.run(button.callback(it))
    return edits


_embeds = {}


def view_embed(view):
    return _embeds[id(view)]


def test_calibration_flow(monkeypatch):
    fake_api(monkeypatch, lambda t: UNICORN if "boyfriend" in t else None)
    seed([(1, "chat before"), (2, "she has a boyfriend?? betrayal"), (3, "good morning")])
    asyncio.run(b.judge_backlog(guild()))
    dm = DM()

    async def run():
        await b.send_calibration(dm, ADMIN, GUILD, 0)
        first = dm.sent[-1]
        assert first.embed.title.startswith("Calibration #1") and "betrayal" in first.embed.description  # flagged first
        assert "-# yargas: chat before" in first.embed.description  # a little context
        assert "Bot" not in str(first.embed.fields)  # the bot's score stays hidden until you answer
        _embeds[id(first.view)] = first.embed
        assert not await first.view.interaction_check(NS(user=NS(id=999)))
        return first

    first = asyncio.run(run())
    edits = click(first.view, "6", channel=dm)
    fields = {f.name: f.value for f in edits[0]["embed"].fields}
    assert fields["You"] == "**6**/10" and fields["Bot"].startswith("**8**/10 · unicorn")
    assert all(c.disabled for c in first.view.children)
    assert b.db.gold_rows(GUILD)[0]["admin_score"] == 6
    second = dm.sent[-1]
    assert second.embed.title.startswith("Calibration #2")
    _embeds[id(second.view)] = second.embed
    click(second.view, "Stop", channel=dm)
    assert b.db.calibration_sample(GUILD, flagged=True) is None  # the scored one isn't offered again


def test_calibration_stats():
    db_rows = [(1, "a"), (2, "b"), (3, "c")]
    seed(db_rows)
    b.db.save_verdicts([(1, 8, "r", UNICORN), (2, 0, None), (3, 0, None)], scoring.RUBRIC_VERSION)
    b.db.save_gold(GUILD, 1, ADMIN, 6)   # bot 8: +2
    b.db.save_gold(GUILD, 2, ADMIN, 0)   # bot 0: exact
    b.db.save_gold(GUILD, 3, ADMIN, 3)   # bot 0: -3
    stats = b.calibration_stats(GUILD)
    assert stats["n"] == 3 and stats["mae"] == pytest.approx(5 / 3) and stats["bias"] == pytest.approx(-1 / 3)
    assert stats["within1"] == pytest.approx(1 / 3) and stats["worst"][0][0] == -3


# --- /model and /scoring ------------------------------------------------------------

def run_cmd(cmd, **ctx_extra):
    sent = []

    async def send(content=None, embed=None, ephemeral=False):
        sent.append(embed or content)

    asyncio.run(cmd.callback(NS(guild=NS(id=GUILD), author=NS(id=ADMIN), send=send, **ctx_extra)))
    return sent[0]


def test_model_and_scoring_commands():
    e = run_cmd(b.model)
    assert b.judge.model in e.fields[0].value and "thinking" in e.fields[1].value
    seed([(1, "a")])
    e = run_cmd(b.scoring_)
    assert e.title == f"📐 Kimoi scoring (rubric v{scoring.RUBRIC_VERSION})" and "**Start**" in e.description



# --- evidence quotes and distress --------------------------------------------------

def test_evidence_must_come_from_the_post_itself():
    from nsabot.judge import evidence_found
    assert evidence_found("I can never forgive this betrayal", "she has a boyfriend?? I can NEVER forgive  this betrayal")
    assert evidence_found("推しが結婚…許せない", "え、推しが結婚した？ 許せない")  # CJK, fragments around …
    assert not evidence_found("isn't that grooming", "and in high school")  # someone else's words
    assert not evidence_found("", "anything") and not evidence_found("a", "a")


def test_flags_without_real_evidence_are_dropped():
    raw = json.dumps({"flagged": [
        {"i": 0, "evidence": "and in high school", "behaviours": ["horny"], "target": "minor", "sincerity": "sincere"},
        {"i": 1, "evidence": "isn't that grooming", "behaviours": ["horny"], "target": "minor", "sincerity": "sincere"},
        {"i": 2, "behaviours": ["unicorn"], "target": "real"},  # no quote at all
    ], "distress": [3]})
    texts = ["and in high school", "we dont have to do anything", "she has a bf", "and hopefully die"]
    out = parse_verdicts(raw, 4, texts)
    assert out[0].severity == 10      # quoted correctly (and sincere): the minor rule still applies
    assert 1 not in out and 2 not in out
    assert out[3] == Verdict(0, "distress", {"distress": True})


def test_distress_is_stored_as_zero_and_kept_out_of_roasts(monkeypatch):
    def flagged_for(text):
        return {"behaviours": ["life_impact"], "target": "real", "sincerity": "sincere", "evidence": "die",
                "distress": True} if "die" in text else None
    fake_api(monkeypatch, flagged_for)
    seed([(1, "and hopefully die"), (2, "good morning")])
    judged, flagged = asyncio.run(b.judge_backlog(guild()))
    row = b.db.get_message(1)
    assert (judged, flagged, row["severity"]) == (2, 0, 0) and json.loads(row["labels"]) == {"distress": True}
    assert [r["content"] for r in b.db.recent_posts(GUILD, 5)] == ["good morning"]
