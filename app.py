import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from deltiet_engine import analyze_curves, extract_nasa_mat

st.set_page_config(page_title="Delt-IEt Battery Lab", page_icon="🔋", layout="wide")
st.title("🔋 Delt-IEt Battery Lab")
st.caption("Experimental environment for battery discharge curve analysis")

with st.sidebar:
    st.header("Engine settings")
    st.markdown("**Mode:** compatibility with original B0005 notebook")
    window_norm = st.number_input("Normalization window (h)", min_value=2, max_value=100, value=10)
    percentile = st.number_input("Threshold percentile (Θ)", min_value=50, max_value=99, value=95)
    st.divider()
    st.caption("Parameters affect results. Record them when exporting to compare analyses.")

st.markdown("""
Upload discharge data as **NASA `.mat`**, **CSV** or **Excel**. The tool computes
Shannon entropy for each voltage curve and applies the Delt-IEt engine.
""")
st.warning("Version 1.0 experimental. The result is a mathematical signal for investigation; it is not a certified diagnostic or a guaranteed failure prediction.")

uploaded = st.file_uploader("Choose a data file", type=["mat", "csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Start by uploading a file. For CSV/Excel, use one row per voltage sample with a column identifying the cycle (long format), or one row per cycle with multiple voltage columns (wide format).")

    st.subheader("🚀 Don't have battery data? Use the NASA benchmark dataset")
    st.markdown("""
The **NASA PCoE Li-ion Battery Aging Dataset** (cells B0005, B0006, B0007) is the standard
benchmark for battery degradation research. It is **free and public domain**.

| Step | Action |
|------|--------|
| 1 | Go to [**data.nasa.gov — Li-ion Battery Aging Datasets**](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets) |
| 2 | Click the **landing page** link → Dashlink Resource 133 |
| 3 | Download **B0005.mat** (≈ 16 MB) |
| 4 | Upload it here — the tool recognises NASA `.mat` files automatically |

> *Reference: B. Saha and K. Goebel (2007). Battery Data Set, NASA Prognostics Data Repository, NASA Ames Research Center.*

Alternatively, pre-converted CSV versions are available on
[IEEE DataPort (DOI 10.21227/c88e-y548)](https://ieee-dataport.org/documents/nasa-ames-pcoe-ehf-calce-and-oxford) —
download `B0005_discharge.csv` and upload it directly.
""")

    st.subheader("Recommended CSV format — long format")
    st.code("cycle,voltage,capacity_Ah\\n1,4.18,2.05\\n1,4.16,2.05\\n1,4.12,2.05\\n2,4.17,2.03\\n2,4.14,2.03", language="csv")
    st.stop()

try:
    if uploaded.name.lower().endswith(".mat"):
        ids, curves, caps, battery_name = extract_nasa_mat(io.BytesIO(uploaded.getvalue()))
        st.success(f"NASA file recognized: {battery_name} — {len(curves)} discharge cycles.")
    else:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)
        if raw.empty:
            st.error("The file is empty.")
            st.stop()

        st.subheader("1. Configure columns")
        mode = st.radio("How is your data organized?", ["Long format: one row per sample", "Wide format: one row per cycle"], horizontal=True)
        cols = list(raw.columns)
        if mode.startswith("Long"):
            c1, c2, c3 = st.columns(3)
            cycle_col = c1.selectbox("Cycle column", cols, index=cols.index("cycle") if "cycle" in cols else 0)
            voltage_candidates = [c for c in cols if "volt" in str(c).lower()]
            voltage_col = c2.selectbox("Voltage column", cols, index=cols.index(voltage_candidates[0]) if voltage_candidates else min(1, len(cols)-1))
            cap_candidates = [c for c in cols if "capacity" in str(c).lower()]
            cap_options = ["(no capacity)"] + cols
            cap_col = c3.selectbox("Capacity column (optional)", cap_options, index=cap_options.index(cap_candidates[0]) if cap_candidates else 0)
            sort_candidates = [c for c in cols if any(k in str(c).lower() for k in ["time", "step", "sample"])]
            sort_col = st.selectbox("Sort samples within cycle by (optional)", ["(file order)"] + cols, index=(["(file order)"] + cols).index(sort_candidates[0]) if sort_candidates else 0)
            work = raw.copy()
            work["_cycle_sort"] = work[cycle_col]
            if sort_col != "(file order)":
                work = work.sort_values([cycle_col, sort_col], kind="stable")
            else:
                work = work.sort_values([cycle_col], kind="stable")
            ids, curves, caps = [], [], []
            for cid, group in work.groupby(cycle_col, sort=True):
                vals = pd.to_numeric(group[voltage_col], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals) < 3:
                    continue
                ids.append(cid)
                curves.append(vals)
                if cap_col != "(no capacity)":
                    cv = pd.to_numeric(group[cap_col], errors="coerce").dropna()
                    caps.append(float(cv.iloc[0]) if len(cv) else np.nan)
                else:
                    caps.append(np.nan)
        else:
            st.caption("Select columns containing voltage curve samples. Each row must represent one cycle.")
            cycle_col = st.selectbox("Cycle identifier", ["(use row number)"] + cols)
            voltage_candidates = [c for c in cols if "volt" in str(c).lower()]
            default_voltage = voltage_candidates
            voltage_cols = st.multiselect("Voltage sample columns", cols, default=default_voltage)
            cap_candidates = [c for c in cols if "capacity" in str(c).lower()]
            cap_options = ["(no capacity)"] + cols
            cap_col = st.selectbox("Capacity column (optional)", cap_options, index=cap_options.index(cap_candidates[0]) if cap_candidates else 0)
            if not voltage_cols:
                st.error("Select at least one voltage column.")
                st.stop()
            ids, curves, caps = [], [], []
            for i, row in raw.iterrows():
                vals = pd.to_numeric(row[voltage_cols], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals) < 3:
                    continue
                ids.append(i + 1 if cycle_col == "(use row number)" else row[cycle_col])
                curves.append(vals)
                caps.append(np.nan if cap_col == "(no capacity)" else pd.to_numeric(pd.Series([row[cap_col]]), errors="coerce").iloc[0])
    if len(curves) < 3:
        st.error("At least three valid curves are required. Check your columns and file format.")
        st.stop()

    st.subheader("2. Data identified")
    st.write(f"**Valid cycles:** {len(curves)}  |  **First cycle:** {ids[0]}  |  **Last cycle:** {ids[-1]}")
    with st.expander("Preview curve sample"):
        st.dataframe(pd.DataFrame({"cycle": ids, "voltage_samples": [len(x) for x in curves], "capacity_Ah": caps}).head(20), use_container_width=True)

    if st.button("▶ Run Delt-IEt", type="primary", use_container_width=True):
        result = analyze_curves(curves, ids, caps, int(window_norm), int(percentile))
        st.session_state["deltiet_result"] = result
        st.session_state["deltiet_source"] = uploaded.name
        st.session_state["deltiet_params"] = (int(window_norm), int(percentile))

except Exception as exc:
    st.error(f"Could not parse the file: {exc}")
    st.stop()

if "deltiet_result" in st.session_state:
    df = st.session_state["deltiet_result"]
    theta = float(df["theta"].iloc[0])
    events = df[df["event"] == 1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cycles analyzed", f"{len(df)}")
    m2.metric("Events detected", f"{len(events)}")
    m3.metric("Threshold Θ", f"{theta:.6g}")
    m4.metric("First event", str(events["cycle"].iloc[0]) if len(events) else "None")

    st.subheader("3. Results")
    tab1, tab2, tab3 = st.tabs(["Delt-IEt Index", "Capacity", "Table"])
    with tab1:
        fig = px.line(df, x="cycle", y="abs_DeltIEt", title="Index magnitude |Delt-IEt|")
        fig.add_hline(y=theta, line_dash="dash", annotation_text=f"Θ = {theta:.6g}")
        if len(events):
            fig.add_scatter(x=events["cycle"], y=events["abs_DeltIEt"], mode="markers", name="Events")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        if df["capacity_Ah"].notna().any():
            fig2 = px.line(df, x="cycle", y="capacity_Ah", title="Capacity per cycle (when provided)")
            if len(events):
                fig2.add_scatter(x=events["cycle"], y=events["capacity_Ah"], mode="markers", name="Cycles with event")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("The file contains no capacity column. The index calculation does not depend on it.")
    with tab3:
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button("⬇ Download results CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="deltiet_battery_lab_results.csv", mime="text/csv")
    st.download_button("⬇ Download parameter report", (
        f"Delt-IEt Battery Lab\\nFile: {st.session_state.get('deltiet_source','')}\\n"
        f"Engine: legacy compatible with B0005 notebook\\nh={st.session_state['deltiet_params'][0]}\\n"
        f"Percentile={st.session_state['deltiet_params'][1]}\\nTheta={theta:.12g}\\n"
        f"Cycles={len(df)}\\nEvents={events['cycle'].tolist()}\\n"
        "Threshold computed retrospectively over the full series; does not represent online operation.\\n"
    ).encode("utf-8"), file_name="deltiet_battery_lab_report.txt", mime="text/plain")

    st.subheader("4. How to interpret")
    st.markdown("""
- **Event** means `|Delt-IEt|` exceeded the configured percentile for this series.
- The algorithm alone does not identify the electrochemical cause of the event.
- Compare events against voltage, capacity, temperature, current, charge/discharge protocol and independent data.
- For scientific comparison, keep the same preprocessing and record the parameters.
""")
    st.caption("Important limitation: this version preserves the retrospective threshold from the original notebook (percentile computed over the full series). It should not be presented as an online/real-time alert.")

                  
