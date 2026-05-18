import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import COR_RISCO, LABEL, META_AGUA, META_ESGOTO, fmt_area, fmt_pop

def _render_municipio_card(row, df_atual, df_hist=None, container=None, key_prefix=""):
    t = container or st
    mun = row.get("nome_municipio", "")
    score = row.get("score", np.nan)
    risco = row.get("risco", "Sem dados")
    cor = COR_RISCO.get(risco, "#cbd5e1")
    porte = row.get("porte", "")

    pop = row.get("populacao_total_residente", np.nan)
    pop_urb = row.get("populacao_urbana_residente", np.nan)
    pop_rur = row.get("populacao_rural_residente", np.nan)
    area = row.get("area_municipio", np.nan)
    dens = row.get("densidade_demografica", np.nan)
    agua = row.get("indice_atendimento_total_agua", np.nan)
    esgoto = row.get("indice_coleta_esgoto", np.nan)
    trat = row.get("indice_tratamento_esgoto", np.nan)
    perda = row.get("indice_perda_distribuicao_agua", np.nan)
    residuos = row.get("cobertura_residuos_solidos", np.nan)
    invest = row.get("investimento_per_capita", np.nan)

    def _v(val, fmt=".1f", suf="%"):
        return f"{val:{fmt}}{suf}" if pd.notna(val) else "N/D"

    t.markdown(f"""
    <div class="profile-header" style="padding:1.5rem 1.5rem 1rem">
        <h2 style="font-size:1.4rem;margin-bottom:.3rem">{mun}</h2>
        <span class="badge" style="background:{cor};color:#fff">{risco}</span>
        <span class="badge" style="background:rgba(255,255,255,.15);color:#fff;margin-left:.3rem">{porte}</span>
        <span class="badge" style="background:rgba(255,255,255,.1);color:rgba(255,255,255,.7);margin-left:.3rem">Índice {_v(score, '.0f', '')}</span>
        <div style="margin-top:1rem;padding:.8rem 1rem;background:rgba(255,255,255,.05);border-radius:10px;border:1px solid rgba(255,255,255,.08)">
            <div style="font-size:2rem;font-weight:900">{fmt_pop(pop, abrev=False)}</div>
            <div style="font-size:.7rem;opacity:.6;text-transform:uppercase;letter-spacing:.3px">Habitantes</div>
        </div>
        <div class="profile-grid" style="margin-top:.8rem">
            <div class="profile-metric"><div class="val">{fmt_pop(pop_urb, abrev=False)}</div><div class="lbl">Pop. urbana</div></div>
            <div class="profile-metric"><div class="val">{fmt_pop(pop_rur, abrev=False)}</div><div class="lbl">Pop. rural</div></div>
            <div class="profile-metric"><div class="val">{fmt_area(area)}</div><div class="lbl">Área</div></div>
            <div class="profile-metric"><div class="val">{_v(dens, '.0f', ' hab/km²')}</div><div class="lbl">Densidade</div></div>
            <div class="profile-metric"><div class="val">{_v(agua)}</div><div class="lbl">Água</div></div>
            <div class="profile-metric"><div class="val">{_v(esgoto)}</div><div class="lbl">Esgoto</div></div>
            <div class="profile-metric"><div class="val">{_v(trat)}</div><div class="lbl">Tratamento</div></div>
            <div class="profile-metric"><div class="val">{_v(perda, '.0f')}</div><div class="lbl">Perda</div></div>
            <div class="profile-metric"><div class="val">{_v(residuos)}</div><div class="lbl">Resíduos</div></div>
            <div class="profile-metric"><div class="val">{_v(invest, '.1f', '')}</div><div class="lbl">R$/hab investido</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    media_agua = df_atual["indice_atendimento_total_agua"].mean()
    media_esgoto = df_atual["indice_coleta_esgoto"].mean()

    linhas = []
    if pd.notna(agua):
        cmp_agua = "acima" if agua >= media_agua else "abaixo"
        cor_agua = "#16a34a" if agua >= media_agua else "#dc2626"
        linhas.append(f'<div style="display:flex;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid #f3f4f6">'
                      f'<span>Água</span>'
                      f'<span><strong>{agua:.1f}%</strong> <span style="color:{cor_agua};font-size:.75rem">{cmp_agua} da média ({media_agua:.1f}%)</span></span></div>')
    if pd.notna(esgoto):
        cmp_esgoto = "acima" if esgoto >= media_esgoto else "abaixo"
        cor_esgoto = "#16a34a" if esgoto >= media_esgoto else "#dc2626"
        linhas.append(f'<div style="display:flex;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid #f3f4f6">'
                      f'<span>Esgoto</span>'
                      f'<span><strong>{esgoto:.1f}%</strong> <span style="color:{cor_esgoto};font-size:.75rem">{cmp_esgoto} da média ({media_esgoto:.1f}%)</span></span></div>')
    if pd.notna(esgoto) and pd.notna(pop) and pop > 0:
        sem_esgoto = pop * (1 - esgoto / 100)
        if sem_esgoto > 1000:
            linhas.append(f'<div style="display:flex;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid #f3f4f6">'
                          f'<span>Sem coleta de esgoto</span>'
                          f'<span><strong style="color:#dc2626">{fmt_pop(sem_esgoto, abrev=False)} hab</strong></span></div>')
    if pd.notna(perda) and perda > 25:
        cor_perda = "#dc2626" if perda > 40 else "#d97706"
        linhas.append(f'<div style="display:flex;justify-content:space-between;padding:.4rem 0">'
                      f'<span>Perda na distribuição</span>'
                      f'<span><strong style="color:{cor_perda}">{perda:.0f}%</strong> <span style="font-size:.75rem;color:{cor_perda}">acima do limite (25%)</span></span></div>')

    if linhas:
        t.markdown(f'<div style="font-size:.82rem;margin:.5rem 0">{"".join(linhas)}</div>', unsafe_allow_html=True)

    if df_hist is not None:
        hist = df_hist[df_hist["nome_municipio"] == mun].sort_values("ano")
        if len(hist) > 1:
            cols_ev = [c for c in ["indice_atendimento_total_agua", "indice_coleta_esgoto",
                                   "indice_tratamento_esgoto", "indice_perda_distribuicao_agua"]
                       if hist[c].notna().any()]
            if cols_ev:
                cores = {"indice_atendimento_total_agua": "#3b82f6", "indice_coleta_esgoto": "#f97316",
                         "indice_tratamento_esgoto": "#22c55e", "indice_perda_distribuicao_agua": "#ef4444"}
                fig = go.Figure()
                for c in cols_ev:
                    s = hist[["ano", c]].dropna()
                    fig.add_trace(go.Scatter(
                        x=s["ano"], y=s[c], mode="lines+markers",
                        name=LABEL.get(c, c), line=dict(color=cores.get(c), width=2),
                        marker=dict(size=4), hovertemplate="%{y:.1f}%",
                    ))
                fig.add_hline(y=META_AGUA, line_dash="dot", line_color="#d1d5db", line_width=1,
                              annotation_text="Meta Água 99%", annotation_font_size=8)
                fig.add_hline(y=META_ESGOTO, line_dash="dot", line_color="#d1d5db", line_width=1,
                              annotation_text="Meta Esgoto 90%", annotation_font_size=8)
                fig.update_layout(
                    height=220, margin=dict(l=10, r=10, t=5, b=5),
                    legend=dict(orientation="h", y=-0.25, font=dict(size=8)),
                    hovermode="x unified", yaxis=dict(range=[-5, 110]),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch",
                                key=f"card_evol_{key_prefix}_{mun}")

def render_perfil(df, df_atual, geojson, ultimo_ano):
    st.markdown('<div class="section-title">Perfil do Município</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Selecione um município para ver seus indicadores detalhados e evolução histórica</div>',
                unsafe_allow_html=True)
    nomes = sorted(df_atual["nome_municipio"].dropna().unique())
    mun = st.selectbox("Município", nomes, index=nomes.index("Rio de Janeiro") if "Rio de Janeiro" in nomes else 0)
    row = df_atual[df_atual["nome_municipio"] == mun].iloc[0]
    _render_municipio_card(row, df_atual, df_hist=df, key_prefix="perfil")
