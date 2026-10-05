"""Streamlit dashboard -> run with:  streamlit run app.py
Stage 9 of the ML lifecycle (deployment). Model training is cached; sliders only re-run the fast forecast step."""
import pandas as pd
import streamlit as st
import election_forecast as ef

st.set_page_config(page_title="eThekwini 2026 Forecast", layout="wide")


@st.cache_resource(show_spinner="Training models on 2016 -> 2021 ...")
def get_models():
    return ef.train_all()


bundles, geo, audit = get_models()
hist = ef.history(bundles)

# ---------------- sidebar: scenario inputs (assumptions the data cannot identify) ----------------
st.sidebar.header("Scenario assumptions")
p = dict(
    persistence=st.sidebar.slider("Swing persistence (0 = frozen at 2021, 1 = full 2016-21 trend)", 0.0, 1.0, 0.5, 0.05),
    turnout_reversion=st.sidebar.slider("Turnout rebound toward 2016", 0.0, 1.0, 0.5, 0.05),
    reg_growth=st.sidebar.slider("Registered-voter growth vs 2021 (%)", -5.0, 15.0, 0.0, 0.5) / 100,
    actionsa_mult=st.sidebar.slider("ActionSA strength vs 2021", 0.0, 2.0, 1.0, 0.1),
    emerging_share=st.sidebar.slider("Party with no history in the data (e.g. MK), % of vote", 0.0, 40.0, 0.0, 1.0) / 100,
    n_sims=st.sidebar.select_slider("Monte-Carlo simulations", [200, 500, 1000, 2000], 1000))
R = ef.run_all(bundles, geo, p)
o1, o3, o4 = R["o1"], R["o3"], R["o4"]

st.title("eThekwini Metro - 04 Nov 2026 Local Government Election Forecast")
st.caption("Voting-district model trained on 2016 and 2021 results; uncertainty from Monte-Carlo simulation.")
t1, t2, t3, t4, t5 = st.tabs(["1 Largest party & coalitions", "2 Ward leaders", "3 Vote share", "4 Turnout", "Data & validation"])

with t1:
    c = st.columns(4)
    c[0].metric("Largest party", o1["largest"])
    c[1].metric("Seats won", int(o1["seats"][o1["largest"]]), f"majority = {o1['majority']} of {o1['n_seats']}")
    c[2].metric("Single-party majority?", "Yes" if o1["single_majority"] else "No")
    c[3].metric("Chance of hung council", f"{o1['p_hung']*100:.0f}%")
    a, b = st.columns(2)
    a.subheader("Projected council seats"); a.bar_chart(o1["seats"])
    b.subheader("Probability of being largest party"); b.bar_chart(o1["p_largest"] * 100)
    st.subheader("Coalition outlook (minimal winning coalitions)")
    if o1["single_majority"]:
        st.success("A single party is projected to hold a majority - no coalition needed.")
    else:
        st.dataframe(o1["coalitions"], hide_index=True, width="stretch")
        st.caption("Arithmetic only: the data contain no information on which parties will actually co-operate. "
                   "'OTHER' is pooled small parties and independents and is excluded from coalitions.")

with t2:
    st.subheader("Projected leader in the three selected wards")
    all_w = list(R["ward_all"].index)
    chosen = st.multiselect("Choose exactly 3 wards (defaults are auto-picked contrasts)", all_w, R["chosen"], max_selections=3)
    if len(chosen) == 3:
        st.dataframe(R["ward_all"].loc[chosen].rename(columns={"Rural": "Rural VD share"}), width="stretch")
        st.bar_chart(R["shares_by_ward"].loc[chosen].T)
    st.caption("Leader = largest of ANC/DA/EFF/IFP/ActionSA on the ward ballot. Independents are pooled in OTHER, "
               "so a strong independent is not identified. 'P(leader wins)' comes from the simulations.")

with t3:
    st.subheader("Projected PR vote share by party")
    st.dataframe(o3, width="stretch")
    df = o3[["Projected share (%)"]].copy()
    df["2021 actual (%)"] = hist["2021"]["share"].reindex(df.index)
    st.bar_chart(df)

with t4:
    c = st.columns(3)
    c[0].metric("Projected voters (ballots cast)", f"{o4['ballots']:,.0f}")
    c[1].metric("Turnout of registered", f"{o4['turnout']*100:.1f}%")
    c[2].metric("80% range", f"{o4['lo']:,.0f} - {o4['hi']:,.0f}")
    st.write(f"Registered voters assumed: **{o4['registered']:,.0f}**  |  2021: {hist['2021']['ballots']:,} voters "
             f"({hist['2021']['turnout']*100:.1f}%)  |  2016: {hist['2016']['ballots']:,} ({hist['2016']['turnout']*100:.1f}%)")
    st.caption("2026 registration figures are not in the datasets - set the growth slider once the IEC publishes them.")

with t5:
    st.subheader("Dataset audit"); st.dataframe(audit, hide_index=True, width="stretch")
    st.subheader("Backtest: predicting 2021 from 2016 (grouped by ward)")
    st.dataframe(bundles["PR"]["cv"], hide_index=True, width="stretch")
    st.caption("Metro-level error is optimistic: the 2016->21 metro swing is estimated from the same election. "
               "Whether that swing continues to 2026 is the 'persistence' assumption and cannot be validated.")
