# Skisilience-lite

A small prototype inspired by **Skisilience**, my team's project at the
EUSALP AI Hackathon (Kempten UAS, Germany), where we placed 3rd.

**All data is synthetic (invented). The risk score, advice and legal
snippets are demonstrations, not real predictions or legal guidance.**

## The Skisilience concept (team pitch)
Ski-lift operators across the Alps share anonymized risk data and adaptation
strategies, with AI helping across languages and regions.



## What this prototype implements
Simplified versions of the pitch ideas, built afterwards by me with AI
assistance to learn how they work:

| Feature | How it works here |
|---|---|
| Report translation and structuring | LLM if API key set, otherwise keyword rules (`extract.py`) |
| RI score | Transparent weighted formula with what-if sliders (`risk.py`) |
| Strategy agent | Finds similar stations and suggests measures that worked (`network.py`) |
| Legal agent | Small RAG demo: retrieval over invented snippets, optional LLM answer (`legal.py`) |
| Cooperative network | Graph linking stations with similar risk profiles (`network.py`) |

## Not implemented
Satellite data ingestion, a trained predictive model, real regulations and
real operator data. The RI score is not validated.

## Run it
    pip install -r requirements.txt
    streamlit run app.py

Optional: set `ANTHROPIC_API_KEY` for LLM translation and legal answers.
Everything also works without a key.

## Files
`app.py` dashboard, `extract.py`, `risk.py`, `network.py`, `legal.py`,
and `data/` with synthetic CSVs.
