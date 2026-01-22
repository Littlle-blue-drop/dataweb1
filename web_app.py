from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import logging
import re
import tempfile

from flask import Flask, render_template, request


def tool_read_pdf_text(path: str) -> str:
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    from pdfminer.high_level import extract_text
    return extract_text(path)


def tool_clean_text(text: str) -> str:
    # Remove URLs and common header/footer noise
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}\b", " ", text)
    text = re.sub(r"\s+\d+/\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tool_extract_key_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    keywords = ["projected", "expected", "CAGR", "inference", "training", "percent", "GW", "kW", "MW"]
    return [s.strip() for s in sentences if any(k in s for k in keywords)]


def tool_summarize(title: str, date: str, key_sents: List[str], limit: int = 6) -> str:
    lines = [
        f"Title: {title or 'N/A'}",
        f"Date: {date or 'N/A'}",
        "Summary:",
    ]
    lines += [f"- {s}" for s in key_sents[:limit]]
    return "\n".join(lines)


def tool_refine_summary(summary: str) -> str:
    # Dedupe + trim overly long bullets
    lines = summary.splitlines()
    seen = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            if line.startswith("- ") and len(line) > 220:
                line = line[:217] + "..."
            out.append(line)
    return "\n".join(out)


def score_summary(summary: str, key_sents: List[str]) -> float:
    # 1) Structure
    structure = 0.0
    structure += 0.2 if "Title:" in summary else 0.0
    structure += 0.2 if "Date:" in summary else 0.0
    structure += 0.1 if "Summary:" in summary else 0.0

    # 2) Coverage of key sentences
    def content_tokens(s: str) -> set:
        return set(re.findall(r"[A-Za-z0-9]+", s.lower()))

    summary_tokens = content_tokens(summary)
    covered = 0
    for s in key_sents[:10]:
        toks = content_tokens(s)
        if toks and len(toks & summary_tokens) / max(1, len(toks)) >= 0.35:
            covered += 1
    coverage_rate = covered / max(1, min(10, len(key_sents)))

    # 3) Repetition (lower is better)
    lines = [l.strip() for l in summary.splitlines() if l.strip().startswith("- ")]
    norm_lines = [" ".join(re.findall(r"[A-Za-z0-9]+", l.lower())) for l in lines]
    unique_ratio = len(set(norm_lines)) / max(1, len(norm_lines))
    repetition_penalty = 1.0 - unique_ratio

    # 4) Fact density (numbers/units)
    facts = re.findall(r"\b(\d+(\.\d+)?%?|\d+\s*(GW|MW|kW|ms|months|years|percent))\b", summary)
    fact_density = min(1.0, len(facts) / 5.0)

    # 5) Length sanity
    length_penalty = 0.0
    if len(summary) < 300:
        length_penalty = 0.1
    if len(summary) > 2000:
        length_penalty = 0.1

    score = (
        structure * 0.4 +
        coverage_rate * 0.25 +
        fact_density * 0.25 +
        (1.0 - repetition_penalty) * 0.1
    )
    score = score - length_penalty
    return max(0.0, min(score, 1.0))


@dataclass
class AgentState:
    path: str
    raw_text: str = ""
    title: str = ""
    date: str = ""
    key_sents: List[str] = field(default_factory=list)
    best_summary: str = ""
    best_score: float = 0.0


def run_agent_on_pdf(path: str) -> Tuple[str, float]:
    state = AgentState(path=path)
    state.raw_text = tool_clean_text(tool_read_pdf_text(state.path))

    t = re.search(r"The next big shifts in AI workloads and\s*hyperscaler strategies",
                  state.raw_text, re.I)
    d = re.search(r"December\s+\d{1,2},\s+\d{4}", state.raw_text)
    state.title = t.group(0) if t else ""
    state.date = d.group(0) if d else ""
    state.key_sents = tool_extract_key_sentences(state.raw_text)

    summary = tool_summarize(state.title, state.date, state.key_sents)
    score = score_summary(summary, state.key_sents)
    state.best_summary, state.best_score = summary, score

    refined = tool_refine_summary(summary)
    refined_score = score_summary(refined, state.key_sents)
    if refined_score >= score:
        state.best_summary, state.best_score = refined, refined_score

    return state.best_summary, state.best_score


app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", summary="", score=None, error=None)


@app.post("/summarize")
def summarize():
    file = request.files.get("pdf")
    if not file or not file.filename.lower().endswith(".pdf"):
        return render_template("index.html", summary="", score=None, error="Please upload a PDF file.")

    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
        file.save(tmp.name)
        summary, score = run_agent_on_pdf(tmp.name)

    return render_template("index.html", summary=summary, score=score, error=None)


if __name__ == "__main__":
    app.run(debug=True)
