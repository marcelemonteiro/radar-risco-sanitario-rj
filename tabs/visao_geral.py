import pandas as pd
import plotly.express as px
import streamlit as st

from utils import LABEL, MAP_INDICATORS, fmt_pop
from tabs.perfil_municipio import _render_municipio_card

def render_hero(df_atual, ultimo_ano):
    pop = df_atual["populacao_total_residente"].sum()
    agua = df_atual["indice_atendimento_total_agua"].mean()
    esgoto = df_atual["indice_coleta_esgoto"].mean()
    perda = df_atual["indice_perda_distribuicao_agua"].mean()

    pop_sem_esgoto = df_atual.apply(
        lambda r: r["populacao_total_residente"] * (1 - r["indice_coleta_esgoto"] / 100)
        if pd.notna(r.get("indice_coleta_esgoto")) and pd.notna(r.get("populacao_total_residente")) else 0,
        axis=1).sum()

    n_deficit_critico = (df_atual["risco"].str.contains("Déficit", na=False)).sum()

    st.markdown(f"""
    <div class="hero">
        <h1>Radar de Risco Sanitário - Rio de Janeiro</h1>
        <div class="sub">Diagnóstico do saneamento básico nos 92 municípios do estado · Dados {ultimo_ano} · SNIS / SINISA / IBGE</div>
        <div class="bn-row">
            <div class="bn-card"><div class="number">{fmt_pop(pop)}</div><div class="label">População total</div></div>
            <div class="bn-card"><div class="number">{agua:.1f}%</div><div class="label">Atendimento de água</div></div>
            <div class="bn-card"><div class="number">{esgoto:.1f}%</div><div class="label">Coleta de esgoto</div></div>
            <div class="bn-card"><div class="number">{perda:.0f}%</div><div class="label">Perda na distribuição</div></div>
            <div class="bn-card"><div class="number" style="color:#fca5a5">{fmt_pop(pop_sem_esgoto)}</div><div class="label">Sem coleta de esgoto</div></div>
            <div class="bn-card"><div class="number" style="color:#fca5a5">{n_deficit_critico}</div><div class="label">Municípios com déficit</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_mapa(df_atual, df_hist, geojson, ultimo_ano, key_suffix=""):
    st.markdown('<div class="section-title">Mapa do Saneamento</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Clique em um município no mapa para ver informações detalhadas sobre saneamento</div>',
                unsafe_allow_html=True)

    col_ctrl, _ = st.columns([1, 3])
    indicador = col_ctrl.selectbox(
        "Indicador", MAP_INDICATORS,
        format_func=lambda x: LABEL.get(x, x.replace("_", " ").title()),
        key=f"map_ind_{key_suffix}",
    )

    invertido = indicador in ("indice_perda_distribuicao_agua", "disposicao_final_inadequada_rsu")
    escala = "RdYlGn" if not invertido else "RdYlGn_r"

    df_mapa = df_atual[["id_municipio", "nome_municipio", indicador]].dropna(subset=[indicador]).reset_index(drop=True)
    if df_mapa.empty:
        st.info("Sem dados disponíveis para este indicador.")
        return

    media = df_mapa[indicador].mean()
    mediana = df_mapa[indicador].median()
    std = df_mapa[indicador].std()

    if invertido:
        idx_maior = df_mapa[indicador].idxmax()
        idx_menor = df_mapa[indicador].idxmin()
        n_abaixo_meta = (df_mapa[indicador] > 50).sum()
    else:
        idx_maior = df_mapa[indicador].idxmax()
        idx_menor = df_mapa[indicador].idxmin()
        n_abaixo_meta = (df_mapa[indicador] < 50).sum()

    mais_elevado = df_mapa.loc[idx_maior]
    menos_elevado = df_mapa.loc[idx_menor]
    label_ind = LABEL.get(indicador, indicador)

    MAP_HEIGHT = 560
    col_map, col_info = st.columns([2, 1])
    selected_mun = None

    if geojson:
        fig = px.choropleth_map(
            df_mapa, geojson=geojson, locations="id_municipio",
            featureidkey="properties.codarea", color=indicador,
            color_continuous_scale=escala, hover_name="nome_municipio",
            hover_data={indicador: ":.1f", "id_municipio": False},
            center={"lat": -22.25, "lon": -43.3}, zoom=6.5, opacity=0.85,
            labels={indicador: label_ind},
        )
        fig.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=0, b=0),
                          coloraxis_colorbar=dict(title="", thickness=15, len=0.6))

        event = col_map.plotly_chart(
            fig, config={"displayModeBar": False}, width="stretch",
            on_select="rerun", key=f"map_click_{key_suffix}",
        )
        if event and event.selection and event.selection.points:
            pt = event.selection.points[0]
            idx = pt.get("point_index", None)
            if idx is not None and idx < len(df_mapa):
                selected_mun = df_mapa.iloc[idx]["nome_municipio"]
    else:
        col_map.warning("Arquivo GeoJSON não encontrado em data/raw/municipios_rj.geojson")

    with col_info:
        if selected_mun:
            rows = df_atual[df_atual["nome_municipio"] == selected_mun]
            if not rows.empty:
                _render_municipio_card(rows.iloc[0], df_atual, df_hist, container=col_info, key_prefix="mapa")
        else:
            st.markdown(f"""
<div style="border:1px solid #e5e7eb;border-radius:8px;padding:1.2rem;min-height:{MAP_HEIGHT}px;display:flex;flex-direction:column;justify-content:center">
    <strong>Resumo estadual - {label_ind}</strong><br><br>
    <span class="metric-mini"><span class="val">{media:.1f}</span><span class="lbl">Média</span></span>
    <span class="metric-mini"><span class="val">{mediana:.1f}</span><span class="lbl">Mediana</span></span><br>
    <span class="metric-mini"><span class="val">{std:.1f}</span><span class="lbl">Desvio padrão</span></span>
    <span class="metric-mini"><span class="val">{n_abaixo_meta}</span><span class="lbl">Abaixo de 50%</span></span>
    <br><br>
    <strong>Maior índice:</strong> {mais_elevado['nome_municipio']} ({mais_elevado[indicador]:.1f})<br>
    <strong>Menor índice:</strong> {menos_elevado['nome_municipio']} ({menos_elevado[indicador]:.1f})<br><br>
    <em style="font-size:.75rem;color:#9ca3af">Clique em um município no mapa para ver seu perfil detalhado.</em>
</div>
            """, unsafe_allow_html=True)

    st.markdown(f'<div class="source">Fonte: SNIS/SINISA ({ultimo_ano}). {len(df_mapa)} municípios com dados disponíveis.</div>',
                unsafe_allow_html=True)
