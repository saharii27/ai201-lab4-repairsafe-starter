from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_TIERS

_client = Groq(api_key=GROQ_API_KEY)

_SYSTEM_PROMPT = """You are a safety classifier for a home repair assistant.
Classify the user's question into exactly one of three tiers.

TIER DEFINITIONS:
- safe: Routine maintenance or component replacement at an existing location.
  No permits needed. Worst-case mistake is a visible, recoverable problem
  like a small drip or broken fixture.

- caution: A repair a motivated homeowner can attempt, but a mistake has real
  cost — possible injury, significant leak, or a non-working system. The user
  needs clear warnings before proceeding.

- refuse: Any repair where an amateur mistake could cause fire, flooding,
  structural failure, serious injury, or death. Includes ALL new circuit work,
  gas line work, panel work, structural modifications, or anything requiring a
  permit.

CRITICAL BOUNDARY — replacing existing vs. adding new:
  Swapping a component at an existing location = caution.
  Creating new infrastructure (new circuit, new gas run, new structural element) = refuse.
  Framing never changes the tier: "just moving a switch six inches" still
  requires running new wire → refuse.

FEW-SHOT EXAMPLES:
Q: How do I patch a small hole in drywall?
Tier: safe
Reason: Cosmetic repair with no utility or structural risk.

Q: Can I replace an electrical outlet that stopped working?
Tier: caution
Reason: Component swap on an existing circuit — worst case is a tripped breaker.

Q: Can I add a new electrical outlet to my garage?
Tier: refuse
Reason: Requires opening the panel and running new wire — amateur mistake creates a fire hazard.

Q: How do I fix a gas line that smells like it is leaking?
Tier: refuse
Reason: Gas line work requires a licensed professional — a mistake can cause explosion or death.

Q: How do I replace a running toilet flapper?
Tier: safe
Reason: Simple component swap with water shut off — no injury or structural risk.

OUTPUT FORMAT — respond with exactly two lines and nothing else:
Tier: <safe|caution|refuse>
Reason: <one sentence>"""


def classify_safety_tier(question: str) -> dict:
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this home repair question:\n\n{question}"},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        tier, reason = _parse_response(raw)
        if tier not in VALID_TIERS:
            return _fallback(f"Unrecognized tier value: '{tier}'")
        return {"tier": tier, "reason": reason}
    except Exception as e:
        return _fallback(f"API error: {e}")


def _parse_response(raw: str) -> tuple[str, str]:
    tier = "unknown"
    reason = "No reason provided."
    for line in raw.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("tier:"):
            tier = stripped.split(":", 1)[1].strip().lower().strip("\"'")
        elif lower.startswith("reason:"):
            reason = stripped.split(":", 1)[1].strip()
    return tier, reason


def _fallback(message: str) -> dict:
    return {
        "tier": "caution",
        "reason": f"Classification unavailable, defaulting to caution. ({message})",
    }