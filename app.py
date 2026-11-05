import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from deltiet_engine import analyze_curves, extract_nasa_mat

st.set_page_config(page_title="Delt-IEt Battery Lab", page_icon="🔋", layout="wide")
st.title("🔋 Delt-IEt Battery Lab")
st.caption("Ambiente experimental para análise de curvas de descarga de baterias")

with st.sidebar:
    st.header("Configuração do motor")
    st.markdown("**Modo:** compatibilidade com o notebook B0005 original")
    window_norm = st.number_input("Janela de normalização (h)", min_value=2, max_value=100, value=10)
    percentile = st.number_input("Percentil do limiar (Θ)", min_value=50, max_value=99, value=95)
    st.divider()
    st.caption("Os parâmetros alteram o resultado. Registre-os ao exportar e comparar análises.")

st.markdown("""
Envie dados de descarga em **NASA `.mat`**, **CSV** ou **Excel**. O programa calcula
uma entropia de Shannon para cada curva de tensão e aplica o motor Delt-IEt.
""")
st.warning("Versão 1.0 experimental. O resultado é um sinal matemático para investigação; não é diagnóstico certificado nem previsão garantida de falha.")

uploaded = st.file_uploader("Escolha um arquivo de dados", type=["mat", "csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Comece enviando um arquivo. Para CSV/Excel, use uma linha por amostra de tensão e uma coluna que identifique o ciclo (formato longo), ou uma linha por ciclo com várias colunas de tensão (formato largo).")
    st.subheader("Formato CSV recomendado — formato longo")
    st.code("cycle,voltage,capacity_Ah\\n1,4.18,2.05\\n1,4.16,2.05\\n1,4.12,2.05\\n2,4.17,2.03\\n2,4.14,2.03", language="csv")
    st.stop()

try:
    if uploaded.name.lower().endswith(".mat"):
        ids, curves, caps, battery_name = extract_nasa_mat(io.BytesIO(uploaded.getvalue()))
        st.success(f"Arquivo NASA reconhecido: {battery_name} — {len(curves)} ciclos de descarga.")
    else:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)
        if raw.empty:
            st.error("O arquivo está vazio.")
            st.stop()

        st.subheader("1. Configure as colunas")
        mode = st.radio("Como seus dados estão organizados?", ["Formato longo: uma linha por amostra", "Formato largo: uma linha por ciclo"], horizontal=True)
        cols = list(raw.columns)
        if mode.startswith("Formato longo"):
            c1, c2, c3 = st.columns(3)
            cycle_col = c1.selectbox("Coluna do ciclo", cols, index=cols.index("cycle") if "cycle" in cols else 0)
            voltage_candidates = [c for c in cols if "volt" in str(c).lower()]
            voltage_col = c2.selectbox("Coluna da tensão", cols, index=cols.index(voltage_candidates[0]) if voltage_candidates else min(1, len(cols)-1))
            cap_candidates = [c for c in cols if "capacity" in str(c).lower() or "capacidade" in str(c).lower()]
            cap_options = ["(sem capacidade)"] + cols
            cap_col = c3.selectbox("Coluna de capacidade (opcional)", cap_options, index=cap_options.index(cap_candidates[0]) if cap_candidates else 0)
            sort_candidates = [c for c in cols if any(k in str(c).lower() for k in ["time", "tempo", "step", "sample", "amostra"])]
            sort_col = st.selectbox("Ordenar amostras dentro do ciclo por (opcional)", ["(ordem do arquivo)"] + cols, index=(["(ordem do arquivo)"] + cols).index(sort_candidates[0]) if sort_candidates else 0)
            work = raw.copy()
            work["_cycle_sort"] = work[cycle_col]
            if sort_col != "(ordem do arquivo)":
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
                if cap_col != "(sem capacidade)":
                    cv = pd.to_numeric(group[cap_col], errors="coerce").dropna()
                    caps.append(float(cv.iloc[0]) if len(cv) else np.nan)
                else:
                    caps.append(np.nan)
        else:
            st.caption("Selecione as colunas que contêm amostras da curva de tensão. Cada linha deve representar um ciclo.")
            cycle_col = st.selectbox("Identificador do ciclo", ["(usar número da linha)"] + cols)
            voltage_candidates = [c for c in cols if "volt" in str(c).lower()]
            default_voltage = voltage_candidates
            voltage_cols = st.multiselect("Colunas de amostras de tensão", cols, default=default_voltage)
            cap_candidates = [c for c in cols if "capacity" in str(c).lower() or "capacidade" in str(c).lower()]
            cap_options = ["(sem capacidade)"] + cols
            cap_col = st.selectbox("Coluna de capacidade (opcional)", cap_options, index=cap_options.index(cap_candidates[0]) if cap_candidates else 0)
            if not voltage_cols:
                st.error("Selecione pelo menos uma coluna de tensão.")
                st.stop()
            ids, curves, caps = [], [], []
            for i, row in raw.iterrows():
                vals = pd.to_numeric(row[voltage_cols], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals) < 3:
                    continue
                ids.append(i + 1 if cycle_col == "(usar número da linha)" else row[cycle_col])
                curves.append(vals)
                caps.append(np.nan if cap_col == "(sem capacidade)" else pd.to_numeric(pd.Series([row[cap_col]]), errors="coerce").iloc[0])
    if len(curves) < 3:
        st.error("Não há pelo menos três curvas válidas. Verifique as colunas e o formato do arquivo.")
        st.stop()

    st.subheader("2. Dados identificados")
    st.write(f"**Ciclos válidos:** {len(curves)}  |  **Primeiro ciclo:** {ids[0]}  |  **Último ciclo:** {ids[-1]}")
    with st.expander("Visualizar amostra das curvas"):
        st.dataframe(pd.DataFrame({"cycle": ids, "n_amostras_tensao": [len(x) for x in curves], "capacity_Ah": caps}).head(20), use_container_width=True)

    if st.button("▶ Executar Delt-IEt", type="primary", use_container_width=True):
        result = analyze_curves(curves, ids, caps, int(window_norm), int(percentile))
        st.session_state["deltiet_result"] = result
        st.session_state["deltiet_source"] = uploaded.name
        st.session_state["deltiet_params"] = (int(window_norm), int(percentile))

except Exception as exc:
    st.error(f"Não foi possível interpretar o arquivo: {exc}")
    st.stop()

if "deltiet_result" in st.session_state:
    df = st.session_state["deltiet_result"]
    theta = float(df["theta"].iloc[0])
    events = df[df["event"] == 1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ciclos analisados", f"{len(df)}")
    m2.metric("Eventos detectados", f"{len(events)}")
    m3.metric("Limiar Θ", f"{theta:.6g}")
    m4.metric("Primeiro evento", str(events["cycle"].iloc[0]) if len(events) else "Nenhum")

    st.subheader("3. Resultado")
    tab1, tab2, tab3 = st.tabs(["Índice Delt-IEt", "Capacidade", "Tabela"])
    with tab1:
        fig = px.line(df, x="cycle", y="abs_DeltIEt", title="Magnitude do índice |Delt-IEt|")
        fig.add_hline(y=theta, line_dash="dash", annotation_text=f"Θ = {theta:.6g}")
        if len(events):
            fig.add_scatter(x=events["cycle"], y=events["abs_DeltIEt"], mode="markers", name="Eventos")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        if df["capacity_Ah"].notna().any():
            fig2 = px.line(df, x="cycle", y="capacity_Ah", title="Capacidade por ciclo (quando fornecida)")
            if len(events):
                fig2.add_scatter(x=events["cycle"], y=events["capacity_Ah"], mode="markers", name="Ciclos com evento")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("O arquivo não contém coluna de capacidade. O cálculo do índice não depende dela.")
    with tab3:
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button("⬇ Baixar resultados CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="deltiet_battery_lab_resultados.csv", mime="text/csv")
    st.download_button("⬇ Baixar relatório de parâmetros", (
        f"Delt-IEt Battery Lab\\nArquivo: {st.session_state.get('deltiet_source','')}\\n"
        f"Motor: legado compatível com notebook B0005\\nh={st.session_state['deltiet_params'][0]}\\n"
        f"Percentil={st.session_state['deltiet_params'][1]}\\nTheta={theta:.12g}\\n"
        f"Ciclos={len(df)}\\nEventos={events['cycle'].tolist()}\\n"
        "Limiar calculado retrospectivamente sobre a série completa; não representa operação online.\\n"
    ).encode("utf-8"), file_name="deltiet_battery_lab_relatorio.txt", mime="text/plain")

    st.subheader("4. Como interpretar")
    st.markdown("""
- **Evento** significa que `|Delt-IEt|` ultrapassou o percentil configurado nesta série.
- O algoritmo não informa sozinho a causa eletroquímica do evento.
- Compare eventos com tensão, capacidade, temperatura, corrente, protocolo de carga/descarga e dados independentes.
- Para comparação científica, mantenha o mesmo pré-processamento e registre os parâmetros.
""")
    st.caption("Limitação importante: esta versão preserva o limiar retrospectivo do notebook original (percentil calculado sobre toda a série). Portanto, ela não deve ser apresentada como alerta online/tempo real.")
