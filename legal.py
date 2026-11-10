"""Small retrieval-augmented (RAG) legal helper.

Retrieval: bag-of-words cosine similarity over illustrative regulation
snippets (invented, NOT real law). Optional: if ANTHROPIC_API_KEY is set,
a language model answers using only the retrieved snippets.
"""
import math
import os
import re
from collections import Counter

STOP = {"the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "is",
        "are", "be", "may", "must", "can", "we", "our", "do", "i", "with"}


def _tokens(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP]


def _cos(a, b):
    num = sum(a[w] * b[w] for w in a if w in b)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(
        sum(v * v for v in b.values()))
    return num / den if den else 0.0


def retrieve(df, question, k=2):
    q = Counter(_tokens(question))
    scored = []
    for _, r in df.iterrows():
        doc = Counter(_tokens(r["title"] + " " + r["text"]))
        scored.append((_cos(q, doc), r))
    scored.sort(key=lambda x: -x[0])
    return [(round(s, 2), r) for s, r in scored[:k] if s > 0]


def answer(df, question):
    hits = retrieve(df, question)
    if not hits:
        return "No relevant passage found.", []
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            context = "\n".join(f'[{r["id"]}] {r["text"]}' for _, r in hits)
            msg = anthropic.Anthropic().messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content":
                           "Answer using ONLY these illustrative passages. "
                           "Cite the passage id. If unsure, say so.\n\n"
                           f"{context}\n\nQuestion: {question}"}],
            )
            return msg.content[0].text, hits
        except Exception as e:
            print("LLM failed, showing passages only:", e)
    return "Most relevant passages (no language model configured):", hits
