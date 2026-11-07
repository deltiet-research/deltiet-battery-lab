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

uploaded = st.file_uploader("Choose a data file", type=["mat", "h5", "hdf5", "csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Start by uploading a file. For CSV/Excel, use one row per voltage sample with a column identifying the cycle (long format), or one row per cycle with multiple voltage columns (wide format).")

    st.subheader("🚀 Don't have battery data? Use one of these free datasets")
    st.markdown("""
#### 1 · NASA PCoE — constant-current protocol (cells B0005, B0006, B0007, B0018)

**👉 [Direct download — Battery Data Set ZIP (≈ 60 MB)](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip)**

`.mat` files. Extract and upload any — the tool recognises NASA format automatically.

> *B. Saha and K. Goebel (2007). Battery Data Set, NASA Ames PCoE.*

---

#### 2 · Galijatovic 2023 — dynamic load profiles, 116 cells, 37 protocols (Panasonic NCR18650B)

**👉 [Download individual cells from Zenodo](https://zenodo.org/records/7896429)**

`.h5` files (~36–520 MB each). Upload any `2_XXX.h5` — the tool reads the format automatically.

> *Galijatovic et al. (2023). Ageing Data of Lithium-ion Panasonic NCR18650B Battery Cells. Zenodo.*

---

#### 3 · RWTH Aachen 2021 — NMC/graphite, 48 cells (Sanyo/Panasonic UR18650E), Germany

**👉 [Download Rawdata.zip (~80 MB)](https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip)**

ZIP contains individual `.csv` files, one per cell. Extract any single CSV and upload — the tool auto-detects RWTH format.

> *Li et al. (2021). Time-series cyclic aging data on 48 commercial NMC/graphite cells. RWTH Aachen.*

---

#### 4 · Tongji University / KIT 2022 — NCA & NCM, 18650 cells, China + Germany

**👉 [Download NCM+NCA ZIP (356 MB)](https://zenodo.org/records/6405084/files/Dataset_3_NCM_NCA_battery.zip?download=1)**

ZIP contains individual `.csv` files per cell (`CYX-Y_Z-#N.csv`). Extract any CSV and upload — the tool auto-detects Tongji format.

> *Che et al. (2022). Data-driven capacity estimation of commercial lithium-ion batteries. Nature Communications.*
""")

    st.subheader("Recommended CSV format — long format")
    st.code("cycle,voltage,capacity_Ah\\n1,4.18,2.05\\n1,4.16,2.05\\n1,4.12,2.05\\n2,4.17,2.03\\n2,4.14,2.03", language="csv")
    st.stop()

def _try_auto_csv(raw_df):
    """
    Try to auto-parse known raw CSV formats (no manual column mapping needed).
    Returns (ids, curves, caps, battery_name, description) or None if unrecognized.
    """
    cols_lower = [str(c).lower().strip() for c in raw_df.columns]
    col_map = {c_low: c_orig for c_low, c_orig in zip(cols_lower, raw_df.columns)}

    # --- Tongji / KIT format ---
    # Columns: time/s, control/V/mA, Ecell/V, <I>/mA, Q discharge/mA.h, Q charge/mA.h
    tongji_v = next((col_map[c] for c in cols_lower if "ecell" in c), None)
    tongji_q = next((col_map[c] for c in cols_lower if "q discharge" in c or "qdischarge" in c), None)
    if tongji_v is not None and tongji_q is not None:
        v_arr = pd.to_numeric(raw_df[tongji_v], errors="coerce").to_numpy(dtype=float)
        q_arr = pd.to_numeric(raw_df[tongji_q], errors="coerce").fillna(0).to_numpy(dtype=float)
        # Cycle boundaries: Q_discharge resets (drops from >0.5 to <0.1 mAh)
        resets = np.where((q_arr[1:] < 0.1) & (q_arr[:-1] > 0.5))[0] + 1
        boundaries = np.concatenate([[0], resets, [len(q_arr)]])
        ids, curves, caps = [], [], []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            q_seg = q_arr[s:e]
            discharge_idx = np.where(q_seg > 0.1)[0]
            if len(discharge_idx) < 10:
                continue
            v_seg = v_arr[s + discharge_idx[0]: s + discharge_idx[-1]]
            v_seg = v_seg[~np.isnan(v_seg)]
            if len(v_seg) < 10:
                continue
            ids.append(len(ids) + 1)
            curves.append(v_seg)
            caps.append(float(q_seg.max()) / 1000.0)  # mAh → Ah
        if len(curves) >= 3:
            return ids, curves, caps, "Tongji/KIT cell", f"Tongji/KIT CSV — {len(curves)} discharge cycles (NCA/NCM)."

    # --- RWTH Aachen / TEMPEST format ---
    # Columns contain 'volt' and 'curr' (case-insensitive)
    # TEMPEST: Voltage_V, Current_A | RWTH: Voltage, Current
    rwth_v = next((col_map[c] for c in cols_lower if "volt" in c), None)
    rwth_i = next((col_map[c] for c in cols_lower if "curr" in c or c == "i" or c == "i/a" or c == "current[a]"), None)
    if rwth_v is not None and rwth_i is not None:
        v_arr = pd.to_numeric(raw_df[rwth_v], errors="coerce").to_numpy(dtype=float)
        i_arr = pd.to_numeric(raw_df[rwth_i], errors="coerce").fillna(0).to_numpy(dtype=float)
        # Discharge: current < -0.01 A. Find contiguous discharge blocks.
        is_discharge = i_arr < -0.01
        # Detect transitions into discharge (False→True)
        starts = np.where(~is_discharge[:-1] & is_discharge[1:])[0] + 1
        ends   = np.where(is_discharge[:-1] & ~is_discharge[1:])[0] + 1
        if len(starts) == 0:
            return None
        # Align starts and ends
        if len(ends) == 0 or ends[0] < starts[0]:
            ends = np.append(ends, len(i_arr))
        if ends[-1] < starts[-1]:
            ends = np.append(ends, len(i_arr))
        ids, curves, caps = [], [], []
        for s, e in zip(starts, ends):
            v_seg = v_arr[s:e]
            v_seg = v_seg[~np.isnan(v_seg)]
            if len(v_seg) < 10:
                continue
            ids.append(len(ids) + 1)
            curves.append(v_seg)
            # Capacity: integrate current over segment (rough Ah estimate)
            dt = 1.0  # assume 1s sampling if unknown
            caps.append(abs(float(np.trapz(i_arr[s:e]))) * dt / 3600.0)
        if len(curves) >= 3:
            return ids, curves, caps, "RWTH cell", f"RWTH Aachen CSV — {len(curves)} discharge cycles (NMC)."

    # --- Real-world multi-cell pack format ---
    # Columns: vol_1..vol_8, temp_1..temp_8, cur
    # Wide time-series: each row = one time sample, 8 cells measured simultaneously
    # Sign convention: cur > 0 = discharge, cur < 0 = charge
    vol_cols = [col_map[c] for c in cols_lower if c.startswith("vol_") and c[4:].isdigit()]
    cur_col  = col_map.get("cur", None)
    if len(vol_cols) >= 1 and cur_col is not None:
        # Use vol_1 (first cell) as representative voltage
        v_arr = pd.to_numeric(raw_df[vol_cols[0]], errors="coerce").to_numpy(dtype=float)
        i_arr = pd.to_numeric(raw_df[cur_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        # Discharge: cur > threshold (positive current = discharging in this convention)
        dis_thresh = max(1.0, np.percentile(np.abs(i_arr[np.abs(i_arr) > 0.5]), 10)) if np.any(np.abs(i_arr) > 0.5) else 1.0
        is_discharge = i_arr > dis_thresh
        starts = np.where(~is_discharge[:-1] & is_discharge[1:])[0] + 1
        ends   = np.where(is_discharge[:-1] & ~is_discharge[1:])[0] + 1
        if len(starts) == 0:
            return None
        if len(ends) == 0 or ends[0] < starts[0]:
            ends = np.append(ends, len(i_arr))
        if ends[-1] < starts[-1]:
            ends = np.append(ends, len(i_arr))
        ids, curves, caps = [], [], []
        for s, e in zip(starts, ends):
            v_seg = v_arr[s:e]
            v_seg = v_seg[~np.isnan(v_seg)]
            if len(v_seg) < 10:
                continue
            ids.append(len(ids) + 1)
            curves.append(v_seg)
            caps.append(abs(float(np.trapz(i_arr[s:e]))) / 3600.0)
        if len(curves) >= 3:
            n_cells = len(vol_cols)
            return ids, curves, caps, f"Pack ({n_cells} cells)", \
                   f"Multi-cell pack CSV — {len(curves)} discharge cycles, {n_cells} cells (LFP/real-world)."

    # --- CALCE Arbin format ---
    # Columns: Cycle_Index, Voltage(V), Current(A), Discharge_Capacity(Ah)
    # (from University of Maryland CALCE battery dataset)
    arbin_cycle = next((col_map[c] for c in cols_lower if "cycle_index" in c or c == "cycle index"), None)
    arbin_v     = next((col_map[c] for c in cols_lower if "voltage(v)" in c or c == "voltage(v)" or c == "voltage [v]"), None)
    arbin_cap   = next((col_map[c] for c in cols_lower if "discharge_capacity" in c or "discharge capacity" in c), None)
    if arbin_cycle is None:
        arbin_cycle = next((col_map[c] for c in cols_lower if c.startswith("cycle") and "index" in c), None)
    if arbin_v is None:
        arbin_v = next((col_map[c] for c in cols_lower if "volt" in c), None)
    if arbin_cycle is not None and arbin_v is not None:
        cycle_arr = pd.to_numeric(raw_df[arbin_cycle], errors="coerce").ffill().to_numpy()
        v_arr     = pd.to_numeric(raw_df[arbin_v], errors="coerce").to_numpy(dtype=float)
        cap_arr   = pd.to_numeric(raw_df[arbin_cap], errors="coerce").fillna(0).to_numpy(dtype=float) if arbin_cap else None
        ids, curves, caps = [], [], []
        for cyc_id in sorted(set(cycle_arr[~np.isnan(cycle_arr)])):
            mask = cycle_arr == cyc_id
            v_seg = v_arr[mask]
            # Keep only the discharge portion: voltage decreasing below 4.3V
            # Arbin marks discharge by negative current; use voltage drop instead
            v_valid = v_seg[(~np.isnan(v_seg)) & (v_seg > 2.0) & (v_seg < 4.4)]
            if len(v_valid) < 20:
                continue
            # Find the discharge segment: last monotonically-ish decreasing region
            # Simpler: take the longest contiguous sub-array where dV < 0 on average
            # Use all points where voltage is declining — just use full valid segment,
            # the DeltIEt engine handles mixed charge/discharge via entropy anyway
            ids.append(int(cyc_id))
            curves.append(v_valid)
            if cap_arr is not None:
                caps.append(float(np.nanmax(cap_arr[mask])))
            else:
                caps.append(0.0)
        if len(curves) >= 3:
            return ids, curves, caps, "CALCE cell", f"CALCE Arbin Excel — {len(curves)} cycles detected (LCO)."

    return None


def extract_mendeley_mat(file_bytes, cell_index=0):
    """Parse Mendeley Dataset5 (MATLAB v7.3 / HDF5).

    Structure: Dataset5/cycles[cell] → Partial_dQ refs (1 per cycle, 121-pt dQ/dV)
               Dataset5/Capacity[cell] → capacity array (Ah per cycle)
    """
    import h5py, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        with h5py.File(tmp_path, 'r') as f:
            root_key = next(k for k in f.keys() if not k.startswith('#'))
            ds = f[root_key]
            n_cells = ds['cycles'].shape[0]
            ci = min(cell_index, n_cells - 1)

            cyc_ref  = ds['cycles'][ci, 0]
            cap_ref  = ds['Capacity'][ci, 0]
            c_grp    = f[cyc_ref]
            cap_arr  = f[cap_ref][0, :].astype(float)

            pdq_refs = c_grp['Partial_dQ'][:, 0]
            ids, curves, caps = [], [], []
            for i, ref in enumerate(pdq_refs):
                try:
                    dq = f[ref][0, :].astype(float)
                    dq = dq[np.isfinite(dq)]
                    if len(dq) < 20:
                        continue
                    ids.append(i + 1)
                    curves.append(dq)
                    caps.append(float(cap_arr[i]) if i < len(cap_arr) else np.nan)
                except Exception:
                    continue
        return ids, curves, caps, f"Mendeley-cell{ci+1}", n_cells
    finally:
        os.unlink(tmp_path)


def extract_galijatovic_h5(file_bytes):
    """Extract discharge cycles from Galijatovic 2023 HDF5 format.

    Strategy: each test block may contain multiple full charge-discharge cycles
    identified by resets in the 'dischargecap' column.  Dynamic-protocol blocks
    have many small current pulses; we only count a reset as a new cycle boundary
    when the preceding segment discharged at least MIN_CAP_AH ampere-hours —
    this filters out micro-pulse artefacts that previously inflated the count
    to tens-of-thousands of false cycles.
    """
    import h5py, tempfile, os
    MIN_CAP_AH = 0.3   # Panasonic NCR18650B nominal ~3.4 Ah; require ≥30% per cycle
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        ids, curves, caps = [], [], []
        with h5py.File(tmp_path, 'r') as f:
            # Read cell identifier from metadata
            try:
                info = f['info'][:]
                cell_id = info['idcell'][0].decode('utf-8') if len(info) > 0 else "unknown"
            except Exception:
                cell_id = "unknown"

            tests = sorted(f['tests'].keys())
            cycle_num = 0

            for t in tests:
                grp = f['tests'][t]
                n = grp['voltage'].shape[0]
                if n < 100:          # skip tiny/corrupt blocks
                    continue

                dc = grp['dischargecap'][:]
                v  = grp['voltage'][:]

                # Detect cycle boundaries: dischargecap drops from ≥MIN_CAP_AH to near 0
                # (a "real" full-cycle reset, not a micropulse artefact)
                prev_large = dc[:-1] >= MIN_CAP_AH
                next_small = dc[1:]  < 0.05
                reset_idx  = np.where(prev_large & next_small)[0] + 1
                boundaries = np.concatenate([[0], reset_idx, [n]])

                for i in range(len(boundaries) - 1):
                    s, e = int(boundaries[i]), int(boundaries[i+1])
                    seg_dc = dc[s:e]
                    max_cap = float(seg_dc.max())

                    if max_cap < MIN_CAP_AH:   # partial / rest segment — skip
                        continue

                    # Isolate the discharge portion (dischargecap actively increasing)
                    increasing = np.where(np.diff(seg_dc) > 1e-5)[0]
                    if len(increasing) < 20:
                        continue
                    ds = int(increasing[0])  + s
                    de = int(increasing[-1]) + s + 1
                    v_curve = v[ds:de].astype(float)
                    if len(v_curve) < 20:
                        continue

                    cycle_num += 1
                    ids.append(cycle_num)
                    curves.append(v_curve)
                    caps.append(max_cap)

        return ids, curves, caps, cell_id
    finally:
        os.unlink(tmp_path)

try:
    fname = uploaded.name.lower()
    if fname.endswith(".mat"):
        try:
            ids, curves, caps, battery_name = extract_nasa_mat(io.BytesIO(uploaded.getvalue()))
            st.success(f"NASA file recognized: {battery_name} — {len(curves)} discharge cycles.")
        except Exception as _mat_err:
            # MATLAB v7.3 files are HDF5 — try h5py parsers in order
            if any(k in str(_mat_err) for k in ("v7.3", "HDF", "h5py", "Please use HDF")):
                _hdf_ok = False
                # 1st try: Galijatovic / generic HDF5 layout (f['tests'])
                try:
                    ids, curves, caps, battery_name = extract_galijatovic_h5(uploaded.getvalue())
                    st.success(f"MATLAB v7.3 (HDF5): cell {battery_name} — {len(curves)} discharge cycles.")
                    _hdf_ok = True
                except Exception:
                    pass
                # 2nd try: Mendeley Dataset5 layout (f['Dataset5'])
                if not _hdf_ok:
                    try:
                        ids, curves, caps, battery_name, _n_cells = extract_mendeley_mat(uploaded.getvalue())
                        st.success(
                            f"Mendeley Dataset5 (dQ/dV): {battery_name} — {len(curves)} cycles "
                            f"({_n_cells} cells in file, showing cell 1)."
                        )
                        _hdf_ok = True
                    except Exception as _mend_err:
                        pass
                if not _hdf_ok:
                    raise ValueError(
                        "Could not parse MATLAB v7.3 file. "
                        "Supported HDF5 layouts: Galijatovic (f['tests']) and Mendeley Dataset5 (f['Dataset5'])."
                    )
            else:
                raise
    elif fname.endswith(".h5") or fname.endswith(".hdf5"):
        ids, curves, caps, battery_name = extract_galijatovic_h5(uploaded.getvalue())
        st.success(f"HDF5 file recognized: cell {battery_name} — {len(curves)} discharge cycles (dynamic protocol).")
    else:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            # CALCE Arbin Excel files have metadata rows before the real header.
            # Try header rows 0-5; pick the first one that gives named columns
            # containing battery-relevant keywords.
            raw = None
            _keywords = {"cycle", "voltage", "current", "capacity", "time", "step"}
            for _hrow in range(6):
                try:
                    _candidate = pd.read_excel(uploaded, header=_hrow)
                    _cols_lower = {str(c).lower() for c in _candidate.columns}
                    if _keywords & _cols_lower:  # at least one keyword found
                        raw = _candidate
                        break
                except Exception:
                    pass
            if raw is None:
                raw = pd.read_excel(uploaded, header=0)  # fallback
        if raw.empty:
            st.error("The file is empty.")
            st.stop()

        # Try auto-detect known formats first
        auto = _try_auto_csv(raw)
        if auto is not None:
            ids, curves, caps, battery_name, _desc = auto
            st.success(f"Format auto-recognized: {_desc}")
            if len(curves) < 3:
                st.error("At least three valid discharge cycles required.")
                st.stop()
            # Jump straight to results, skip manual mapping UI
            if st.button("▶ Run Delt-IEt", key="auto_run", type="primary", use_container_width=True):
                result = analyze_curves(curves, ids, caps, int(window_norm), int(percentile))
                st.session_state["deltiet_result"] = result
                st.session_state["deltiet_source"] = uploaded.name
                st.session_state["deltiet_params"] = (int(window_norm), int(percentile))
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

       
