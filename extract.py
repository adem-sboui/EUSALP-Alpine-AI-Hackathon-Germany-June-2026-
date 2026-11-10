"""Turn a free-text ski-lift risk report into structured fields.

Two modes:
- LLM mode: if ANTHROPIC_API_KEY is set, a language model translates the
  report to English and extracts the fields.
- Fallback mode: simple keyword rules on the English text (no key needed).
"""
import json
import os

RISK_KEYWORDS = {
    "avalanche": ["avalanche"],
    "low snow": ["little natural snow", "lack of snow", "no snow", "low snow"],
    "storm": ["storm", "wind"],
    "heavy rain": ["rain"],
}
MITIGATION_KEYWORDS = {
    "artificial snow": ["artificial snow", "snowmaking"],
    "customer alerts": ["alert", "sms"],
    "summer/alternative offer": ["summer", "alternative"],
    "staff reallocation": ["moved staff", "staff"],
}


def _first_match(text, table, default="other"):
    text = text.lower()
    for label, words in table.items():
        if any(w in text for w in words):
            return label
    return default


def extract_rules(report_en):
    """Fallback: keyword rules on English text."""
    return {
        "english": report_en,
        "risk_type": _first_match(report_en, RISK_KEYWORDS),
        "mitigation": _first_match(report_en, MITIGATION_KEYWORDS),
    }


def extract_llm(report):
    """LLM mode: translate + extract structured JSON."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    prompt = (
        "You process risk reports from ski-lift operators. Translate the "
        "report to English and extract fields. Reply with JSON only, with "
        'keys: "english", "risk_type" (one of: avalanche, low snow, storm, '
        'heavy rain, other), "mitigation" (short phrase).\n\nReport: ' + report
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip().strip("`").removeprefix("json").strip()
    return json.loads(text)


def extract(report, report_en):
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return extract_llm(report)
        except Exception as e:  # fall back instead of crashing the demo
            print("LLM failed, using rules:", e)
    return extract_rules(report_en)
