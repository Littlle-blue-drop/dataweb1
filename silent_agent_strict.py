from dataclasses import dataclass, field
from typing import List
import logging
import re

# ---- Tools ----
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

# ---- Strict Scoring ----
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

# ---- Agent State ----
@dataclass
class AgentState:
    goal: str
    path: str
    raw_text: str = ""
    title: str = ""
    date: str = ""
    key_sents: List[str] = field(default_factory=list)
    summary: str = ""
    best_summary: str = ""
    best_score: float = 0.0

# ---- Agent ----
class SilentMultiRoundAgent:
    def __init__(self, state: AgentState, max_rounds: int = 3, min_confidence: float = 0.7):
        self.state = state
        self.max_rounds = max_rounds
        self.min_confidence = min_confidence

    def run(self):
        # Round 1: read + extract
        self.state.raw_text = tool_read_pdf_text(self.state.path)
        self.state.raw_text = tool_clean_text(self.state.raw_text)
        t = re.search(r"The next big shifts in AI workloads and\s*hyperscaler strategies",
                      self.state.raw_text, re.I)
        d = re.search(r"December\s+\d{1,2},\s+\d{4}", self.state.raw_text)
        self.state.title = t.group(0) if t else ""
        self.state.date = d.group(0) if d else ""
        self.state.key_sents = tool_extract_key_sentences(self.state.raw_text)

        # Round 2+: draft -> score -> refine -> rollback if needed
        summary = tool_summarize(self.state.title, self.state.date, self.state.key_sents)
        score = score_summary(summary, self.state.key_sents)
        self._update_best(summary, score)

        for _ in range(self.max_rounds - 1):
            if score >= self.min_confidence:
                break
            refined = tool_refine_summary(summary)
            refined_score = score_summary(refined, self.state.key_sents)

            if refined_score < score:
                summary = self.state.best_summary
                score = self.state.best_score
                break
            else:
                summary, score = refined, refined_score
                self._update_best(summary, score)

        print(self.state.best_summary)

    def _update_best(self, summary: str, score: float):
        if score >= self.state.best_score:
            self.state.best_score = score
            self.state.best_summary = summary

if __name__ == "__main__":
    state = AgentState(
        goal="Summarize PDF",
        path="workloads.pdf",
    )
    agent = SilentMultiRoundAgent(state, max_rounds=3, min_confidence=0.7)
    agent.run()
