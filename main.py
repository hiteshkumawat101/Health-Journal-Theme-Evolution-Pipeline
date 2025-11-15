#!/usr/bin/env python3
"""
Non-LLM Journal Theme Evolution Pipeline
- Clusters journal entries using TF-IDF + KMeans
- Automatically labels themes (no LLM used)
- Generates micro-practices from predefined templates
- Produces output JSON and a clean timeline plot

Usage:
    python main.py dataset.json --out output.json --plot timeline.png
"""

import argparse
import json
import os
import re
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# -------------------------------------------
# CLEANING
# -------------------------------------------

STOPWORDS = set("""
a an the and or but if while is am are was were be been being
to of in on for from with as by at it this that these those
i me my mine we our ours you your yours he she they them
his her their theirs not no
""".split())

def basic_clean(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def remove_stopwords(s: str) -> str:
    return " ".join([t for t in (s or "").split() if t not in STOPWORDS and len(t) > 1])


# -------------------------------------------
# TF-IDF Helper
# -------------------------------------------

def top_terms_for_cluster(X, labels, feature_names, cluster_id, n=8):
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    centroid = X[idx].mean(axis=0).A1
    order = np.argsort(-centroid)[:n]
    return [feature_names[i] for i in order]


# -------------------------------------------
# THEME LABELING WITHOUT LLM
# -------------------------------------------

THEME_MAP = {
    "stress": ["work", "tired", "pressure", "deadline", "overwhelm", "exhaust", "burnout"],
    "reflection": ["purpose", "meaning", "thinking", "journal", "reflect", "question", "why"],
    "connection": ["friend", "family", "talked", "support", "relationship", "connect"],
    "self-care": ["sleep", "rest", "walk", "exercise", "meditate", "breathe", "calm"],
    "mood": ["happy", "sad", "angry", "upset", "excited", "hopeful"],
}

def guess_theme(top_terms: List[str]) -> str:
    score = {key: 0 for key in THEME_MAP}

    for term in top_terms:
        for theme, keywords in THEME_MAP.items():
            if term in keywords:
                score[theme] += 1

    best_theme = max(score, key=score.get)
    if score[best_theme] == 0:
        # fallback to descriptive label
        return "General Reflection"

    return best_theme.title().replace("-", " & ")


# -------------------------------------------
# MICRO-PRACTICES WITHOUT LLM
# -------------------------------------------

PRACTICES = {
    "Stress": "Take a 3-minute breathing break and relax your shoulders.",
    "Reflection": "Write one question you're exploring today.",
    "Connection": "Send one supportive message to someone you care about.",
    "Self Care": "Pause and take 5 slow breaths or stretch gently.",
    "Mood": "Identify your emotion and name one need you have today.",
    "General Reflection": "Note one small intention for the next few hours.",
}

def micro_practice_for_theme(theme):
    if theme in PRACTICES:
        return PRACTICES[theme]
    return PRACTICES["General Reflection"]


# -------------------------------------------
# MAIN PIPELINE
# -------------------------------------------

def run_pipeline(input_path: str, output_path: str, plot_path: str):

    df = pd.read_json(input_path)
    df["clean"] = df["text"].apply(lambda t: remove_stopwords(basic_clean(t)))

    # Vectorize
    vec = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=1)
    X = vec.fit_transform(df["clean"])
    feature_names = vec.get_feature_names_out()

    # Choose best K = 2–4
    best_k, best_score, best_labels = None, -1, None
    for k in [2, 3, 4]:
        if len(df) < k:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except:
            score = -1
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    df["cluster"] = best_labels

    # Label clusters
    cluster_to_theme = {}
    for c in sorted(set(best_labels)):
        terms = top_terms_for_cluster(X, best_labels, feature_names, c)
        theme = guess_theme(terms)
        cluster_to_theme[c] = theme

    df["theme"] = df["cluster"].map(cluster_to_theme)
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt")

    # Build phases (continuous theme spans)
    phases = []
    current = None

    for _, row in df.iterrows():
        theme = row["theme"]
        if current is None:
            current = {"theme": theme, "start": row["date_dt"], "end": row["date_dt"], "entries": [row["entry_id"]]}
        elif row["theme"] == current["theme"]:
            current["end"] = row["date_dt"]
            current["entries"].append(row["entry_id"])
        else:
            phases.append(current)
            current = {"theme": theme, "start": row["date_dt"], "end": row["date_dt"], "entries": [row["entry_id"]]}

    if current:
        phases.append(current)


    # Macro insight (first vs last third)
    n = len(df)
    first_theme = df.iloc[:max(1, n//3)]["theme"].mode()[0]
    last_theme = df.iloc[-max(1, n//3):]["theme"].mode()[0]

    if first_theme != last_theme:
        insight = f"Your journaling reflects a shift from {first_theme.lower()} to {last_theme.lower()}."
    else:
        insight = f"Your journaling centers around {first_theme.lower()}."


    # Build output JSON
    out = {
        "phases": [],
        "insight": insight,
        "meta": {"best_k": best_k, "silhouette_score": float(best_score)}
    }

    for p in phases:
        out["phases"].append({
            "phase_label": p["theme"],
            "start_date": p["start"].date().isoformat(),
            "end_date": p["end"].date().isoformat(),
            "representative_entries": p["entries"][:2],
            "micro_practice": micro_practice_for_theme(p["theme"]),
            "confidence": float(round(best_score, 3)),
        })

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)


    # -------------------------------------------
    # FIXED TIMELINE PLOT
    # -------------------------------------------

    themes = df["theme"].unique().tolist()
    theme_to_y = {t: i for i, t in enumerate(themes)}
    y_vals = [theme_to_y[t] for t in df["theme"]]

    plt.figure(figsize=(14, 5))

    # scatter points
    plt.scatter(df["date_dt"], y_vals, s=70)

    # horizontal bars for each phase
    for p in phases:
        y = theme_to_y[p["theme"]]
        plt.hlines(y, p["start"], p["end"], colors="blue", linewidth=4, alpha=0.4)

    # label themes
    for theme in themes:
        subset = df[df["theme"] == theme]
        mid = subset["date_dt"].iloc[len(subset)//2]
        y = theme_to_y[theme]
        plt.text(mid, y + 0.1, theme, ha="center", fontsize=10, fontweight="bold")

    plt.yticks(list(theme_to_y.values()), list(theme_to_y.keys()))
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Theme")
    plt.title("Theme Evolution Timeline")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    print("\n✔ Pipeline finished!")
    print("JSON:", output_path)
    print("Plot:", plot_path)
    print("Best K:", best_k, "| Silhouette:", best_score)


# -------------------------------------------
# CLI
# -------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="output.json")
    ap.add_argument("--plot", default="timeline.png")
    args = ap.parse_args()
    run_pipeline(args.input, args.out, args.plot)
