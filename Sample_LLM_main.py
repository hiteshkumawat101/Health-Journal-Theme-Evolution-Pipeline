#!/usr/bin/env python3
"""
main.py

Zumlo – Full LLM-based Journal Theme Evolution Pipeline (Groq, llama3-8b-8192)

- Ingests ~15 journal entries (entry_id, date, text, optional tags)
- Uses Groq LLM to:
    - Assign a theme label to EACH entry
    - Rate emotional intensity (1–5) for each entry
    - Generate micro-practices for each temporal phase
    - Generate weekly micro-practices based on dominant weekly theme
    - Generate an overall micro-insight about evolution

- Groups entries into phases by merging consecutive entries with the same theme.
- Produces:
    - Structured JSON (phases, insight, weekly_micro_practices, meta)
    - Matplotlib timeline PNG with:
        - X-axis: date
        - Y-axis: theme
        - Color: emotional intensity (1–5)

Usage:
    python main.py dataset.json --out output.json --plot timeline.png

Dependencies:
    pip install groq pandas matplotlib python-dateutil
"""

import argparse
import json
import os
import re
import time
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# ---------- Configuration ----------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set the GROQ_API_KEY environment variable before running.")

client = Groq(api_key=GROQ_API_KEY)

# Free, fast model on Groq
GROQ_MODEL = "llama-3.1-8b-instant"

# ---------- Groq helper with retry ----------

def groq_retry_chat(messages, model=GROQ_MODEL, max_retries=3, **kwargs):
    """
    Retry wrapper around client.chat.completions.create().
    messages: list of {"role": "...", "content": "..."} dicts.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            return resp
        except Exception as e:
            wait = 1.5 ** attempt
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise


# ---------- LLM logic: annotation, practices, insight ----------

def llm_annotate_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use Groq LLM to annotate each entry with:
        - theme_label (2–5 words, supportive, non-clinical)
        - emotional_intensity (1–5)
    The LLM sees ALL entries together so it can re-use consistent labels.
    Returns a list of dicts:
        { "entry_id": ..., "theme_label": ..., "emotional_intensity": int }
    """
    # Prepare raw JSON for the prompt (string)
    entries_json = json.dumps(
        [
            {"entry_id": e["entry_id"], "date": e["date"], "text": e["text"]}
            for e in entries
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = (
        "You are analyzing a short series of personal journal entries over about three weeks.\n"
        "Your job is to assign a supportive, non-clinical theme and an emotional intensity score to each entry.\n\n"
        "For each entry, provide:\n"
        "- theme_label: 2–5 words, everyday language (e.g., 'Work Stress', 'Meaning & Values Reflection', 'Self-Care & Calm').\n"
        "- emotional_intensity: integer from 1 to 5 (1 = very calm/neutral, 3 = moderate emotion, 5 = very intense feelings).\n\n"
        "Guidelines:\n"
        "- Avoid ANY diagnostic or clinical labels (do not say 'depression', 'anxiety disorder', etc.).\n"
        "- Try to reuse theme labels consistently for similar entries (for example, many entries about stress at work "
        "can all use 'Work Stress').\n"
        "- Themes should support mental wellness reflection (e.g., stress, reflection, self-care, connection, acceptance).\n\n"
        "Here are the journal entries as a JSON array:\n\n"
        f"{entries_json}\n\n"
        "Now return a JSON array where each item has:\n"
        "{\n"
        "  \"entry_id\": string,\n"
        "  \"theme_label\": string,\n"
        "  \"emotional_intensity\": integer (1-5)\n"
        "}\n\n"
        "The order of the output array MUST match the input order.\n"
        "Return ONLY valid JSON (no extra text, no comments)."
    )

    messages = [{"role": "user", "content": prompt}]
    resp = groq_retry_chat(messages, max_tokens=600, temperature=0.4)
    raw = resp.choices[0].message.content.strip()

    # Try to parse JSON robustly
    try:
        annotations = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first JSON array in the response
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            raise ValueError("LLM did not return valid JSON for annotations.")
        annotations = json.loads(m.group(0))

    # Basic validation
    if not isinstance(annotations, list):
        raise ValueError("LLM annotation output is not a JSON array.")
    for item in annotations:
        if "entry_id" not in item or "theme_label" not in item or "emotional_intensity" not in item:
            raise ValueError("Each annotation must contain 'entry_id', 'theme_label', 'emotional_intensity'.")

    # Normalize
    for item in annotations:
        item["theme_label"] = str(item["theme_label"]).strip()
        try:
            item["emotional_intensity"] = int(item["emotional_intensity"])
        except Exception:
            item["emotional_intensity"] = 3

    return annotations


_practice_cache: Dict[str, str] = {}


def llm_micro_practice_for_theme(theme_label: str, example_texts: List[str]) -> str:
    """
    LLM generates a 1–2 sentence micro-practice for this theme.
    Uses example texts from this phase to tailor the suggestion.
    """
    if theme_label in _practice_cache:
        return _practice_cache[theme_label]

    examples_str = "\n".join(f"- {t.strip()}" for t in example_texts[:3])

    prompt = (
        "You are a practical, supportive coach.\n"
        f"The journaling theme is: \"{theme_label}\".\n\n"
        "Here are a few short example snippets from this phase:\n"
        f"{examples_str}\n\n"
        "Create one very short micro-practice (1–2 sentences) that:\n"
        "- Is specific, gentle, and non-judgmental.\n"
        "- Can be done in under 5 minutes.\n"
        "- Supports emotional regulation, clarity, or self-care.\n"
        "- Uses everyday language and avoids any clinical or diagnostic terms.\n\n"
        "Return ONLY the micro-practice as one or two sentences."
    )

    messages = [{"role": "user", "content": prompt}]
    resp = groq_retry_chat(messages, max_tokens=80, temperature=0.6)
    text = resp.choices[0].message.content.strip()
    practice = text.splitlines()[0].strip()
    _practice_cache[theme_label] = practice
    return practice


def llm_overall_insight(phases_output: List[Dict[str, Any]]) -> str:
    """
    LLM micro-insight summarizing overall evolution across phases.
    """
    if not phases_output:
        return "No entries were available to analyze."

    lines = []
    for p in phases_output:
        lines.append(
            f"- Phase '{p['phase_label']}' ran from {p['start_date']} to {p['end_date']} "
            f"and included {len(p['representative_entries'])} representative entries."
        )
    phases_summary = "\n".join(lines)

    prompt = (
        "You are summarizing how someone's journaling themes evolved over a few weeks.\n\n"
        "You will see a list of phases with their labels and date ranges.\n"
        "Write ONE short, supportive insight (1–2 sentences) that captures the overall movement, "
        "for example: 'from overwhelm into more meaning-seeking and steadier self-care'.\n"
        "Avoid any diagnostic or clinical language.\n\n"
        "Phases:\n"
        f"{phases_summary}\n\n"
        "Return ONLY the insight as one or two sentences."
    )

    messages = [{"role": "user", "content": prompt}]
    resp = groq_retry_chat(messages, max_tokens=80, temperature=0.5)
    text = resp.choices[0].message.content.strip()
    return text.splitlines()[0].strip()


# ---------- Phase & weekly logic ----------

def build_phases(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Group consecutive entries with the same theme into phases.
    df must be sorted by date_dt and contain 'entry_id', 'theme_label', 'text'.
    Returns raw phase dicts with datetime objects inside.
    """
    phases_raw: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}

    for _, row in df.iterrows():
        theme = row["theme_label"]
        dt = row["date_dt"]
        eid = row["entry_id"]
        text = row["text"]

        if not current:
            current = {
                "phase_label": theme,
                "start": dt,
                "end": dt,
                "entries": [eid],
                "texts": [text],
            }
        elif theme == current["phase_label"]:
            current["end"] = dt
            current["entries"].append(eid)
            current["texts"].append(text)
        else:
            phases_raw.append(current)
            current = {
                "phase_label": theme,
                "start": dt,
                "end": dt,
                "entries": [eid],
                "texts": [text],
            }

    if current:
        phases_raw.append(current)

    return phases_raw


def compute_weekly_micro_practices(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Group by (year, ISO week), pick dominant theme, generate micro-practice via LLM.
    """
    weekly: List[Dict[str, Any]] = []
    df["year"] = df["date_dt"].dt.isocalendar().year
    df["week"] = df["date_dt"].dt.isocalendar().week

    for (year, week), group in df.groupby(["year", "week"]):
        dominant_theme = group["theme_label"].mode()[0]
        example_texts = group["text"].tolist()
        practice = llm_micro_practice_for_theme(dominant_theme, example_texts)
        weekly.append(
            {
                "year": int(year),
                "week": int(week),
                "start_date": group["date_dt"].min().date().isoformat(),
                "end_date": group["date_dt"].max().date().isoformat(),
                "dominant_theme": dominant_theme,
                "micro_practice": practice,
            }
        )
    return weekly


# ---------- Pipeline ----------

def run_pipeline(input_path: str, output_path: str, plot_path: str):
    # 1) Read input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    entries = json.load(open(input_path, "r", encoding="utf-8"))

    # Normalize into DataFrame
    df = pd.DataFrame(entries)
    required = {"entry_id", "date", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input JSON must contain {required} per entry.")

    df["date_dt"] = pd.to_datetime(df["date"])

    # 2) Ask LLM to annotate each entry (theme + emotional intensity)
    annotations = llm_annotate_entries(entries)

    # Merge annotations back into df by entry_id
    ann_df = pd.DataFrame(annotations)
    df = df.merge(ann_df, on="entry_id", how="left")

    # Safety: clean up column names
    df.rename(columns={"theme_label": "theme_label", "emotional_intensity": "intensity"}, inplace=True)

    # Sort by date
    df = df.sort_values("date_dt").reset_index(drop=True)

    # 3) Build phases (consecutive runs of same theme)
    phases_raw = build_phases(df)

    # 4) Build phases_output for JSON, with LLM micro-practices
    phases_output: List[Dict[str, Any]] = []
    for p in phases_raw:
        theme = p["phase_label"]
        example_texts = p["texts"]
        micro = llm_micro_practice_for_theme(theme, example_texts)
        phases_output.append(
            {
                "phase_label": theme,
                "start_date": p["start"].date().isoformat(),
                "end_date": p["end"].date().isoformat(),
                "representative_entries": p["entries"][:2],
                "micro_practice": micro,
                # simple "confidence" proxy: average intensity in that phase
                "confidence": float(
                    df[df["entry_id"].isin(p["entries"])]["intensity"].mean()
                ),
            }
        )

    # 5) Weekly micro-practices
    weekly_micro = compute_weekly_micro_practices(df)

    # 6) Overall insight
    insight = llm_overall_insight(phases_output)

    # 7) Build output JSON structure
    out_obj: Dict[str, Any] = {
        "phases": phases_output,
        "insight": insight,
        "weekly_micro_practices": weekly_micro,
        "meta": {
            "llm_provider": "groq",
            "llm_model": GROQ_MODEL,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    # 8) Timeline plot (themes over time, colored by intensity)
    unique_themes = list(dict.fromkeys(df["theme_label"].tolist()))
    theme_to_y = {t: i for i, t in enumerate(unique_themes)}
    y_vals = [theme_to_y[t] for t in df["theme_label"]]

    plt.figure(figsize=(14, 5))

    # Scatter points
    sc = plt.scatter(
        df["date_dt"],
        y_vals,
        c=df["intensity"],
        cmap="coolwarm",
        vmin=1,
        vmax=5,
        s=80,
        edgecolor="black",
        alpha=0.9,
    )

    # Horizontal bars for phases
    for p in phases_raw:
        y = theme_to_y[p["phase_label"]]
        plt.hlines(y, xmin=p["start"], xmax=p["end"], linewidth=4, alpha=0.4)

    # Label each theme once, centered
    for theme in unique_themes:
        subset = df[df["theme_label"] == theme]
        mid_date = subset["date_dt"].iloc[len(subset) // 2]
        y = theme_to_y[theme]
        plt.text(
            mid_date,
            y + 0.18,
            theme,
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.yticks(list(theme_to_y.values()), list(theme_to_y.keys()))
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Theme")
    plt.title("Theme Evolution Timeline (color = emotional intensity 1–5)")

    cbar = plt.colorbar(sc)
    cbar.set_label("Emotional Intensity (1–5)")

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    # 9) Log summary
    print("\n✅ Pipeline complete!")
    print(f"- Input:        {input_path}")
    print(f"- Output JSON:  {output_path}")
    print(f"- Timeline PNG: {plot_path}")


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full LLM-based Theme Evolution Pipeline (Groq)")
    parser.add_argument("input", help="Input JSON file with journal entries")
    parser.add_argument("--out", default="output.json", help="Output JSON path")
    parser.add_argument("--plot", default="timeline.png", help="Timeline PNG path")
    args = parser.parse_args()

    run_pipeline(args.input, args.out, args.plot)
