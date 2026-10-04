"""Formula scoring: DeepSeek labels each post, this module turns the labels into a 0-10 score.

Labelling is more consistent than asking a model for a number, and the weights here can be tuned
(and old posts re-scored from their stored labels) without touching the prompt or the API.
"""

RUBRIC_VERSION = 3  # bump when the labels or their definitions in the prompt change (then /rescore)

# Starting points per behaviour. A post with several takes the highest, plus MULTI_BONUS.
BASE = {
    "worship": 2,
    "spending": 4,
    "gachikoi": 4,
    "horny": 5,
    "unicorn": 6,
    "life_impact": 7,
    "bodily_servitude": 7,
    "stalking_harassment": 7,
}
TARGET = {"real": 1, "character": 0, "minor": 0, "fan": 0, "none": 0}
INTENSITY = {"passing": -1, "clear": 0, "graphic": 1}
SINCERITY = {"bit": -1, "ambiguous": 0, "sincere": 1}
DOUBLING_DOWN = 1
MULTI_BONUS = 1
SEXUAL = {"horny", "bodily_servitude"}  # with a minor target, sincere or ambiguous, more than passing: always 10
SPIRAL_CAP = 8  # an oshi spiral (crying, drinking, not sleeping over a seiyuu) is a server joke, but never a 10

NAMES = {
    "worship": "worship",
    "spending": "spending",
    "gachikoi": "gachikoi",
    "horny": "horny",
    "unicorn": "unicorn",
    "life_impact": "life impact",
    "bodily_servitude": "bodily/servitude",
    "stalking_harassment": "stalking/harassment",
    "real": "real person",
    "character": "2D",
    "minor": "minor character",
    "fan": "another fan",
    "bit": "bit",
    "ambiguous": "ambiguous",
    "sincere": "sincere",
    "spiral": "oshi spiral",
}


def clean(raw: dict) -> dict:
    """Normalise model output: unknown values fall back to neutral defaults."""
    behaviours = raw.get("behaviours") or raw.get("behaviour") or []
    if isinstance(behaviours, str):
        behaviours = [behaviours]
    behaviours = [b for b in dict.fromkeys(str(b).strip().lower() for b in behaviours) if b in BASE]

    def pick(key: str, table: dict, default: str) -> str:
        value = str(raw.get(key, default)).strip().lower()
        return value if value in table else default

    return {
        "behaviours": behaviours,
        "target": pick("target", TARGET, "none"),
        "intensity": pick("intensity", INTENSITY, "clear"),
        "sincerity": pick("sincerity", SINCERITY, "ambiguous"),
        "doubling_down": raw.get("doubling_down") is True,
        "about_someone_else": raw.get("about_someone_else") is True,
        "distress": raw.get("distress") is True,
        "spiral": raw.get("spiral") is True,
    }


def score(labels: dict) -> int:
    """0 = not kimoi, otherwise 1-10."""
    labels = clean(labels)
    behaviours = labels["behaviours"]
    if not behaviours or labels["about_someone_else"] or labels["distress"]:
        return 0
    if (labels["target"] == "minor" and SEXUAL & set(behaviours)
            and labels["sincerity"] != "bit" and labels["intensity"] != "passing"):
        return 10
    total = (
        max(BASE[b] for b in behaviours)
        + TARGET[labels["target"]]
        + INTENSITY[labels["intensity"]]
        + SINCERITY[labels["sincerity"]]
        + (DOUBLING_DOWN if labels["doubling_down"] else 0)
        + (MULTI_BONUS if len(behaviours) > 1 else 0)
    )
    return max(1, min(SPIRAL_CAP if labels["spiral"] else 10, total))


def describe(labels: dict | None) -> str:
    """Short tag line for reports, e.g. 'bodily/servitude · real person · sincere'."""
    if not labels:
        return ""
    labels = clean(labels)
    parts = [" + ".join(NAMES[b] for b in labels["behaviours"])] if labels["behaviours"] else []
    if labels["target"] != "none":
        parts.append(NAMES[labels["target"]])
    if labels["sincerity"] != "ambiguous":
        parts.append(NAMES[labels["sincerity"]])
    if labels["intensity"] == "graphic":
        parts.append("graphic")
    if labels["doubling_down"]:
        parts.append("doubling down")
    if labels["spiral"]:
        parts.append(NAMES["spiral"])
    return " · ".join(parts)


def formula_lines() -> list[str]:
    """The formula as text, for /scoring. Built from the tables above so it can't drift."""
    def signed(n: int) -> str:
        return f"+{n}" if n > 0 else str(n)

    base = ", ".join(f"{NAMES[b]} {v}" for b, v in sorted(BASE.items(), key=lambda kv: kv[1]))
    return [
        f"**Start** (highest behaviour): {base}",
        f"**Target**: real person {signed(TARGET['real'])}, 2D / another fan {signed(TARGET['character'])}",
        "**Intensity**: " + ", ".join(f"{k} {signed(v)}" for k, v in INTENSITY.items()),
        "**Sincerity**: obvious bit " + signed(SINCERITY["bit"]) + ", ambiguous " + signed(SINCERITY["ambiguous"])
        + ", sincere " + signed(SINCERITY["sincere"]),
        f"**Doubling down** {signed(DOUBLING_DOWN)} · **two or more behaviours** {signed(MULTI_BONUS)}",
        f"**Kept between 1 and 10**, and at most **{SPIRAL_CAP}** for an oshi spiral (crying, drinking, not sleeping "
        "over a seiyuu). Sincere, explicit horny or bodily/servitude content about a minor character is always **10**. "
        "Only pointing at someone else's kimoi, and anything about wanting to die or self-harm, is **0**.",
    ]
