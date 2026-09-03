"""Prompt builders + the scope-boundary system prompt (v3 add-on Sections 0, 3).

The SCOPE_SYSTEM prompt encodes the HARD scope boundary: this tool is a sensing,
classification, and presentation aid only. The assistant must never select or
recommend a weapon/interceptor, compute engagement/launch/trajectory solutions,
or make any "engage/fire" recommendation - it declines and redirects to
signal-behaviour description instead.
"""

from __future__ import annotations

import json

SCOPE_SYSTEM = (
    "You are the AI Analyst embedded in an Electronic Support (ES) receiver "
    "demonstrator called 'Smart Scan EW'. Your role is strictly SENSING, "
    "CLASSIFICATION, and PRESENTATION support: explain what detected signals' "
    "behaviour patterns look like, explain the scheduler's performance and "
    "metrics, and help an operator prioritise their ATTENTION.\n\n"
    "HARD SCOPE BOUNDARY - you must NEVER:\n"
    "- select, recommend, name, or pair any weapon, interceptor, or munition "
    "(real or fictional) against a detected emitter;\n"
    "- compute or describe missile/interceptor trajectories, guidance, launch "
    "envelopes, or engagement timing;\n"
    "- make any 'engage / fire / intercept-with-X' recommendation of any kind.\n\n"
    "If asked for anything outside scope (e.g. 'which missile should intercept "
    "this?'), politely decline in one sentence, state that weapon/engagement "
    "recommendations are outside this tool's scope, and offer instead to "
    "describe the signal's behaviour pattern or the scheduler's performance.\n\n"
    "Category labels are illustrative behaviour-pattern analogues, NOT a "
    "validated IFF or real platform identification. Be concise, precise, and "
    "operator-friendly. Do not invent numbers not present in the provided data.\n\n"
    "FORMATTING (the UI renders light markdown):\n"
    "- Write in short paragraphs of 1-3 sentences; leave a blank line between them.\n"
    "- When you report a band's attributes, put each on its OWN bullet line "
    "starting with '- ' (e.g. '- Confidence: 0.88'). Never chain many attributes "
    "in one sentence with ' - ' dashes.\n"
    "- Use **bold** only for band labels such as **B08**.\n"
    "- Keep the whole reply scannable and under ~120 words unless asked for a summary."
)


def fallback_narration(detail: dict) -> str:
    """Always-available 1-2 sentence narration from the feature vector.

    Used when Claude is offline so the band popover never shows a blank
    'unavailable' box during a demo. Stays in scope: behaviour + analogue only.
    """
    cls = detail.get("classification", {}) or {}
    feats = detail.get("features", {}) or {}
    label = detail.get("label", f"Band {detail.get('band', '?')}")
    analogue = cls.get("analogue_title") or cls.get("analogue_short") or "unclassified analogue"
    behaviour = cls.get("short") or cls.get("label") or "Unclassified"
    conf = cls.get("confidence")
    duty = feats.get("duty_cycle")
    hop = feats.get("hop_rate")
    per = feats.get("period")
    pstr = feats.get("periodicity_strength")
    ev = feats.get("evidence")
    conf_s = f"{int(round(float(conf) * 100))}%" if conf is not None else "n/a"
    duty_s = f"{float(duty):.2f}" if duty is not None else "n/a"
    hop_s = f"{float(hop):.2f}" if hop is not None else "n/a"
    per_s = f"{float(per):.1f}" if per is not None else "n/a"
    pstr_s = f"{float(pstr):.2f}" if pstr is not None else "n/a"
    return (
        f"{label} matches a {behaviour} pattern, shown as a {analogue} "
        f"for this demo (confidence {conf_s}, {ev or 0} scans of evidence). "
        f"Duty cycle {duty_s}, hop rate {hop_s}, period {per_s} ticks "
        f"(periodicity {pstr_s}). This is an illustrative signal-behaviour "
        f"analogue — not a real platform identification."
    )


def build_narrate_prompt(detail: dict) -> str:
    """1-2 plain-English sentences describing one band's behaviour pattern."""
    cls = detail.get("classification", {})
    feats = detail.get("features", {})
    per = detail.get("periodicity")
    lines = [
        "Describe, in 1-2 plain-English sentences for an operator, the signal "
        "behaviour pattern on this frequency band. Mention the behaviour-pattern "
        "label, why it matched (period/duty/agility), and a rough confidence. "
        "Write plain prose only: no bullet points, no markdown, no bold. "
        "Do NOT mention weapons or engagement.",
        "",
        f"Band label: {detail.get('label', detail.get('band'))}",
        f"Current belief P(active): {detail.get('belief')}",
        f"Behaviour label: {cls.get('label')} (confidence {cls.get('confidence')})",
        f"Illustrative analogue (demo only, not IFF): {cls.get('analogue_title')}",
        f"Matched rule: {cls.get('matched_rule')}",
        f"Duty cycle: {feats.get('duty_cycle')}",
        f"Periodicity strength: {feats.get('periodicity_strength')}, period(ticks): {feats.get('period')}",
        f"Hop rate: {feats.get('hop_rate')}, neighbour bandwidth: {feats.get('bandwidth')}",
        f"Evidence (scans observed): {feats.get('evidence')}",
    ]
    if per:
        lines.append(
            f"Predicted next active tick: {per.get('next_active_tick')} "
            f"(period {per.get('period')})"
        )
    return "\n".join(lines)


def build_chat_prompt(question: str, snapshot: dict) -> str:
    """Operator question + compact scenario snapshot."""
    return (
        "Operator question:\n"
        f"{question}\n\n"
        "Current scenario snapshot (JSON):\n"
        f"{json.dumps(snapshot, indent=2)}\n\n"
        "Answer concisely using only this data. If the question asks for weapon "
        "or engagement recommendations, decline per your scope and offer to "
        "describe signal behaviour or scheduler performance instead."
    )


def build_summary_prompt(
    metrics: dict, classifications: list[dict], periodicity: list[dict]
) -> str:
    """End-of-run written summary for a hackathon report."""
    return (
        "Write a short written summary (2-3 short paragraphs) suitable for a "
        "hackathon report. Frame it as an explainer of WHAT THE SYSTEM OBSERVED "
        "and HOW THE SCHEDULER PERFORMED - not as an operational or engagement "
        "assessment. Compare the Smart Scheduler against the sequential, random, "
        "and greedy baselines using the metrics; note the notable behaviour "
        "patterns detected; and keep behaviour-pattern labels illustrative "
        "(not real platform IDs). Do not mention weapons or engagement.\n\n"
        "Formatting: use 2-3 short paragraphs separated by a blank line. You may "
        "use **bold** for band labels and key figures, and '- ' bullet lines when "
        "listing several bands or metrics. Do not put a heavy bold title at the top; "
        "start directly with the first paragraph.\n\n"
        "Final metrics per strategy (JSON):\n"
        f"{json.dumps(metrics, indent=2)}\n\n"
        "Detected behaviour patterns (top bands):\n"
        f"{json.dumps(classifications, indent=2)}\n\n"
        "Periodicity findings:\n"
        f"{json.dumps(periodicity, indent=2)}"
    )
