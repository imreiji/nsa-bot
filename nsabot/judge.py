"""DeepSeek agent that reads message batches and scores how kimoi each post is."""

import json
import logging

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

MAX_CHARS = 800  # per message sent to the model

JUDGE_PROMPT = """You are the NSA (Neckbeard Surveillance Agency), an analyst auditing a Discord server for
"kimoi" (キモい) posts: cringe, creepy or deeply unhinged otaku behaviour.

Kimoi includes, roughly in rising severity:
- casual weeb-speak, unprompted "uwu", "nya", excessive kaomoji
- waifu/husbando declarations, parasocial devotion to vtubers or idols
- "uooooh", "cunny", "correction needed" style posting, horny comments about 2D characters
- describing gacha whale spending, body pillows, shrines, or real-life sacrifices for a character as normal
- in-character roleplay nobody asked for, overly detailed fetish or lore dumps
- genuinely disturbing content involving characters who look like minors (always 9-10)

NOT kimoi: normally discussing anime, games, manga or episodes; sharing art or news; everyday chat.
Judge each message on its own, but use the surrounding messages for context and irony. Be funny
in your reasons but accurate in your scores; most messages in a normal server are 0.

Severity scale: 1-3 mild weeb, 4-6 genuinely kimoi, 7-8 deeply unsettling, 9-10 call the actual NSA.

You receive a JSON list of messages, each with an index "i", "author" and "text".
Reply with a JSON object: {"flagged": [{"i": <index>, "severity": <1-10>, "reason": "<max 15 words>"}]}
Only include messages with severity >= 1. Return {"flagged": []} if nothing is kimoi."""

ROAST_PROMPT = """You are the NSA (Neckbeard Surveillance Agency) writing a short classified dossier on a
Discord user, based on their most kimoi posts and stats. Write 3-5 sentences in a dry, deadpan
intelligence-report voice. Be funny and roast their otaku behaviour, but do not insult appearance,
race, gender, or anything other than what they posted. Plain text, no markdown headers."""


class Judge:
    def __init__(self, api_key: str, model: str, base_url: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def judge(self, messages: list[tuple[str, str]]) -> dict[int, tuple[int, str]]:
        """messages: (author, text) in chronological order.

        Returns {index: (severity, reason)} for flagged messages only. Raises on API errors so the
        caller can leave the batch unjudged and retry later.
        """
        payload = [{"i": i, "author": a, "text": t[:MAX_CHARS]} for i, (a, t) in enumerate(messages)]
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return parse_verdicts(resp.choices[0].message.content or "", len(messages))

    async def roast(self, name: str, stats: str, posts: list[tuple[int, str, str]]) -> str:
        """posts: (severity, text, reason)."""
        evidence = "\n".join(f"- [{sev}/10] {text[:300]!r} (analyst note: {reason})" for sev, text, reason in posts)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ROAST_PROMPT},
                {"role": "user", "content": f"Subject: {name}\nStats: {stats}\nEvidence:\n{evidence}"},
            ],
            temperature=1.0,
            max_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip()


def parse_verdicts(raw: str, n: int) -> dict[int, tuple[int, str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("judge returned invalid JSON: %.200s", raw)
        raise
    out: dict[int, tuple[int, str]] = {}
    for v in data.get("flagged", []) if isinstance(data, dict) else []:
        try:
            i, sev = int(v["i"]), int(v["severity"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < n and sev > 0:
            out[i] = (min(sev, 10), str(v.get("reason", ""))[:200])
    return out
