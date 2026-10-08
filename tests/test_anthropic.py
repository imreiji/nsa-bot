"""The Claude backend: request shape for Haiku 5.5, images, refusals."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

import nsabot.bot as b
from nsabot.db import DB, Message
from nsabot.judge import Judge, Refused, Truncated, anthropic_content


def claude_judge(reply="{\"flagged\": []}", stop="end_turn", **kw):
    judge = Judge("key", "claude-haiku-5-5", None, DB(":memory:"), provider="anthropic", **kw)
    sent = []

    class Stream:
        def __init__(self, **kwargs):
            sent.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_final_message(self):
            return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text=reply)], stop_reason=stop,
                      stop_details=NS(category="general_harms") if stop == "refusal" else None,
                      usage=NS(input_tokens=100, output_tokens=20, cache_creation_input_tokens=0,
                               cache_read_input_tokens=3000))

    judge.client.messages.stream = Stream
    return judge, sent


def test_judge_request_fits_haiku_5_5():
    judge, sent = claude_judge(reply='{"flagged": [{"i": 0, "evidence": "my wife", "behaviours": ["worship"], '
                                     '"target": "character"}]}', effort="high")
    result = asyncio.run(judge.judge({"messages": [{"i": 0, "text": "she is my wife"}]}, 1, anchors="\nnotes"))
    req = sent[0]
    assert req["model"] == "claude-haiku-5-5" and "temperature" not in req
    assert req["system"][-1]["cache_control"] == {"type": "ephemeral"} and req["system"][0]["text"].endswith("notes")
    assert [m["role"] for m in req["messages"]] == ["user"]  # never ends on an assistant prefill
    assert req["output_config"] == {"effort": "high"} and req["max_tokens"] == 2000 + 32000
    assert result[0].severity > 0  # the thinking block is skipped, the text block parsed
    assert judge.db.tokens_today() == 3120


def test_thinking_off_means_low_effort_and_default_effort_is_left_to_the_model():
    judge, sent = claude_judge(reply="hi")
    asyncio.run(judge.respond({"messages": []}, thinking=False))
    asyncio.run(judge.roast("x", "stats", []))
    assert sent[0]["output_config"] == {"effort": "low"} and sent[0]["max_tokens"] == 400 + 4000
    assert "output_config" not in sent[1]  # no NSA_EFFORT: Haiku's own default (medium)


def test_images_become_claude_image_blocks():
    content = anthropic_content([{"type": "text", "text": "batch"},
                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                                 {"type": "image_url", "image_url": {"url": "https://cdn/x.jpg"}}])
    assert content == [{"type": "text", "text": "batch"},
                       {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                       {"type": "image", "source": {"type": "url", "url": "https://cdn/x.jpg"}}]


def test_refusals_and_truncation_raise():
    judge, _ = claude_judge(stop="refusal")
    with pytest.raises(Refused, match="general_harms"):
        asyncio.run(judge.judge({"messages": [{"i": 0, "text": "x"}]}, 1))
    judge, _ = claude_judge(stop="max_tokens", reply="")
    with pytest.raises(Truncated):
        asyncio.run(judge.judge({"messages": [{"i": 0, "text": "x"}]}, 1))


def test_a_refused_batch_is_narrowed_down_to_the_post_it_objects_to(monkeypatch):
    b.db.conn.execute("DELETE FROM messages")
    b.db.save_batch(50, 4, [Message(i, 10, 50, 5, "u", "bad" if i == 3 else f"post {i}") for i in range(1, 5)])
    calls = []

    async def judge_with_quip(payload, n, quip, note=None, anchors=None, images=None):
        texts = [m["text"] for m in payload["messages"] if "i" in m]
        calls.append(texts)
        if "bad" in texts:
            raise Refused("declined by the model (general_harms)")
        return {}, None

    monkeypatch.setattr(b.judge, "judge_with_quip", judge_with_quip)
    guild = NS(id=10, get_channel_or_thread=lambda _: NS(name="general", topic=None, is_nsfw=lambda: False),
               get_channel=lambda _: None)
    asyncio.run(b.judge_backlog(guild))
    assert b.db.count_unjudged(10) == 0  # nothing is left to retry forever
    assert b.db.get_message(3)["reason"] == "declined by the model" and b.db.get_message(3)["severity"] == 0
    assert ["bad"] in calls and len(calls) <= 5


def test_evaluation_narrows_refusals_and_reports_dropped_posts(monkeypatch, tmp_path):
    from nsabot import evaluate
    rows = [{"id": str(i), "should": 5, "verdict": "ok", "safety": None, "bot": 5} for i in range(1, 9)]
    path = tmp_path / "review.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(evaluate.load_review_set, "__defaults__", (str(path),))
    b.db.conn.execute("DELETE FROM messages")
    b.db.save_batch(50, 8, [Message(i, 10, 50, 5, "u", "bad" if i == 3 else f"post {i}") for i in range(1, 9)])
    b.db.save_batch(51, 108, [Message(100 + i, 10, 51, 5, "u", "x") for i in range(1, 9)])
    rows += [{"id": str(100 + i), "should": 5, "verdict": "ok", "safety": None, "bot": 5} for i in range(1, 9)]
    path.write_text("\n".join(json.dumps(r) for r in rows))

    async def judge_with_quip(payload, n, quip, note=None, anchors=None, images=None):
        texts = [m["text"] for m in payload["messages"] if "i" in m]
        if "bad" in texts:
            raise Refused("declined by the model (general_harms)")
        if "x" in texts:
            raise RuntimeError("429")
        return {i: b.Verdict(5, "r", {"behaviours": ["spending"]}) for i in range(n)}, None

    monkeypatch.setattr(b.judge, "judge_with_quip", judge_with_quip)
    guild = NS(id=10, get_channel_or_thread=lambda _: NS(name="general", topic=None, is_nsfw=lambda: False))
    report = asyncio.run(b.run_evaluation(guild))
    assert report["asked"] == 16 and report["n"] == 8  # channel 51's batch kept failing
    assert report["failed"] == {"declined by the model (scored 0)": 1, "RuntimeError": 8}
    embed = b.evaluation_embed(10, report)
    assert "8 of 16 posts left out" in embed.description and "RuntimeError ×8" in embed.description


# --- the judge as a Console-built Managed Agent -------------------------------------------------

def agent_judge(events, **kw):
    from nsabot import judge as J
    judge = Judge("key", "claude-haiku-5-5", None, DB(":memory:"), provider="anthropic",
                  agent_id="agent_1", environment_id="env_1", **kw)
    calls = {"create": [], "send": [], "delete": []}

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def __aiter__(self):
            async def gen():
                for e in events:
                    yield e
            return gen()

    async def create(**kwargs):
        calls["create"].append(kwargs)
        return NS(id="sesn_1")

    async def stream(session_id):
        return Stream()

    async def send(session_id, events):
        calls["send"].append(events)

    async def delete(session_id):
        calls["delete"].append(session_id)

    judge.client.beta.sessions = NS(create=create, delete=delete, events=NS(stream=stream, send=send))
    return judge, calls


def usage_event():
    return NS(type="span.model_request_end",
              model_usage=NS(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=100))


def test_agent_judge_runs_one_capped_session_per_batch():
    reply = '{"flagged": [{"i": 0, "evidence": "my wife", "behaviours": ["worship"], "target": "character"}]}'
    judge, calls = agent_judge([NS(type="agent.message", content=[NS(type="text", text=reply)]), usage_event(),
                                NS(type="session.status_idle", stop_reason=NS(type="end_turn"), stop_details=None)])
    result, _ = asyncio.run(judge.judge_with_quip({"messages": [{"i": 0, "text": "she is my wife"}]}, 1, quip=False,
                                                  anchors="\nServer notes: x",
                                                  images=[(0, "data:image/png;base64,AAAA")]))
    assert result[0].severity > 0 and judge.db.tokens_today() == 115
    create = calls["create"][0]
    assert create["agent"] == "agent_1" and create["environment_id"] == "env_1"
    assert create["budget"] == {"type": "limit", "max_list_cost": {"amount": "0.25", "currency": "USD"}}
    content = calls["send"][0][0]["content"]
    assert calls["send"][0][0]["type"] == "user.message"
    assert content[0]["text"].startswith("Context for this server:") and "Server notes" in content[0]["text"]
    assert {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}} in content
    assert calls["delete"] == ["sesn_1"]  # throwaway session cleaned up


def test_agent_refusals_and_failures():
    judge, calls = agent_judge([NS(type="session.status_idle", stop_reason=NS(type="refusal"),
                                   stop_details=NS(category="general_harms"))])
    with pytest.raises(Refused, match="general_harms"):
        asyncio.run(judge.judge({"messages": [{"i": 0, "text": "x"}]}, 1))
    judge, calls = agent_judge([NS(type="session.status_idle", stop_reason=NS(type="budget_reached"), stop_details=None)])
    with pytest.raises(RuntimeError, match="budget_reached"):
        asyncio.run(judge.judge({"messages": [{"i": 0, "text": "x"}]}, 1))
    assert calls["delete"] == []  # failed sessions are kept to inspect in the Console


def test_sync_agent_pushes_the_judge_prompt_only_when_it_changed():
    from nsabot.judge import CLAUDE_JUDGE_PROMPT as JUDGE_PROMPT
    judge, _ = agent_judge([])
    updates = []

    async def retrieve(agent_id):
        return NS(model=NS(id="claude-haiku-5-5"), system="old prompt", tools=[], version=1)

    async def update(agent_id, system):
        updates.append(system)
        return NS(model=NS(id="claude-haiku-5-5"), system=system, tools=[], version=2)

    judge.client.beta.agents = NS(retrieve=retrieve, update=update)
    asyncio.run(judge.sync_agent())
    assert updates == [JUDGE_PROMPT] and judge.agent_model == "claude-haiku-5-5"

    async def retrieve_current(agent_id):
        return NS(model=NS(id="claude-haiku-5-5"), system=JUDGE_PROMPT, tools=[], version=2)

    judge.client.beta.agents = NS(retrieve=retrieve_current, update=update)
    asyncio.run(judge.sync_agent())
    assert len(updates) == 1


def test_claude_gets_its_own_judge_prompt_with_the_same_rules():
    from nsabot.judge import CLAUDE_JUDGE_PROMPT, JUDGE_PROMPT
    judge, sent = claude_judge()
    asyncio.run(judge.judge({"messages": [{"i": 0, "text": "x"}]}, 1))
    assert sent[0]["system"][0]["text"] == CLAUDE_JUDGE_PROMPT
    assert "STEP 1. Things you never flag" in CLAUDE_JUDGE_PROMPT and "<knowledge>" in CLAUDE_JUDGE_PROMPT
    assert JUDGE_PROMPT.endswith(CLAUDE_JUDGE_PROMPT.split("<rules>\n")[1].split("\n</rules>")[0])
    assert Judge("k", "deepseek-flash", "http://x", DB(":memory:")).judge_prompt == JUDGE_PROMPT


def test_prompts_over_haikus_cheap_tier_are_flagged():
    from nsabot.judge import PARSE_STATS
    PARSE_STATS.clear()
    judge, _ = claude_judge()
    judge._check_prompt_size(40_000)
    judge._check_prompt_size(120_000)
    assert PARSE_STATS["over_100k"] == 1
