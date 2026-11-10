import pandas as pd
import streamlit as st

import legal
import network
import risk
from extract import extract

st.set_page_config(page_title="Skisilience-lite", layout="wide")
st.title("Skisilience-lite")
st.caption("Simplified prototype inspired by the Skisilience hackathon pitch. "
           "All data is synthetic. Scores and advice are demos, not real "
           "predictions or legal guidance.")

reports = pd.read_csv("data/sample_reports.csv")
stations = pd.read_csv("data/stations.csv")
regs = pd.read_csv("data/regulations.csv")
stations["ri_score"] = stations.apply(risk.score_row, axis=1)


@st.cache_data
def process(df):
    rows = []
    for _, r in df.iterrows():
        out = extract(r["report"], r["report_en"])
        rows.append({"region": r["region"], "original": r["report"], **out})
    return pd.DataFrame(rows)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Reports", "RI score", "Strategy agent", "Legal agent", "Network"])

with tab1:
    st.subheader("Translated and structured reports")
    data = process(reports)
    st.dataframe(data[["region", "english", "risk_type", "mitigation"]])
    c1, c2 = st.columns(2)
    c1.bar_chart(data["risk_type"].value_counts())
    c2.bar_chart(data["mitigation"].value_counts())
    text = st.text_area("Try your own report (any language)")
    if st.button("Analyse report") and text:
        st.json(extract(text, text))

with tab2:
    st.subheader("RI score (demo risk index, 0 = low risk, 100 = high risk)")
    st.caption("Transparent weighted formula on synthetic data. "
               "Not a trained or validated model.")
    st.bar_chart(stations.set_index("station")["ri_score"])
    st.markdown("**What-if:** change the inputs for one station")
    name = st.selectbox("Station", stations.station, key="ri")
    r = stations[stations.station == name].iloc[0]
    snow = st.slider("Snow trend (reliable days per decade)", -20, 0,
                     int(r["snow_trend_days_per_decade"]))
    energy = st.slider("Energy cost index", 0, 100, int(r["energy_cost_index"]))
    closures = st.slider("Closure days last season", 0, 40,
                         int(r["closure_days_last_season"]))
    st.metric("RI score", risk.ri_score(snow, energy, closures, r["altitude_m"]))
    comps = risk.components(snow, energy, closures, r["altitude_m"])
    st.bar_chart(pd.Series(comps, name="component (0-1)"))

with tab3:
    st.subheader("Strategy agent")
    st.caption("Suggests diversification measures that worked for similar "
               "stations in other places (simple similarity, not an LLM).")
    name = st.selectbox("Your station", stations.station, key="strat")
    sugg = network.strategy_suggestions(stations, name)
    if sugg:
        st.dataframe(pd.DataFrame(sugg))
    else:
        st.info("No successful similar cases in the sample data.")

with tab4:
    st.subheader("Legal agent (RAG demo)")
    st.caption("Retrieves from invented, illustrative regulation snippets. "
               "Not real legal text or advice.")
    q = st.text_input("Ask a question",
                      "Can we open a chairlift in summer for hikers?")
    if st.button("Ask") and q:
        reply, hits = legal.answer(regs, q)
        st.write(reply)
        for score, row in hits:
            st.markdown(f"**[{row['id']}] {row['title']}** "
                        f"({row['scope']}, match {score})")
            st.write(row["text"])

with tab5:
    st.subheader("Cooperative network")
    st.caption("Stations from different regions with similar risk profiles "
               "are linked: they could learn from each other.")
    thr = st.slider("Similarity threshold", 0.5, 0.95, 0.8, 0.01)
    st.graphviz_chart(network.to_dot(stations, thr))
    st.dataframe(pd.DataFrame(network.edges(stations, thr),
                              columns=["station A", "station B", "similarity"]))
