import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import ANO_META, META_AGUA, META_ESGOTO

def render_evolucao(df, ultimo_ano):
    st.markdown('<div class="section-title">Evolução dos Indicadores (2014–2024)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight">
        O <strong>Marco Legal do Saneamento</strong> (Lei 14.026/2020) estabelece metas de universalização até
        <strong>{ANO_META}</strong>: <strong>99%</strong> de cobertura de água e <strong>90%</strong> de coleta e
        tratamento de esgoto. Os gráficos mostram a evolução média dos municípios do RJ.
    </div>
    """, unsafe_allow_html=True)

    indicadores = [
        ("indice_atendimento_total_agua", "Atendimento de Água", "#3b82f6", META_AGUA),
        ("indice_coleta_esgoto", "Coleta de Esgoto", "#f97316", META_ESGOTO),
        ("indice_tratamento_esgoto", "Tratamento de Esgoto", "#22c55e", None),
        ("indice_perda_distribuicao_agua", "Perda na Distribuição", "#ef4444", 25),
    ]
    cols = st.columns(2)
    df_hist = df[df["ano"].between(2014, ultimo_ano)]

    for i, (col_name, titulo, cor, meta) in enumerate(indicadores):
        stats = df_hist.groupby("ano")[col_name].agg(["mean", "min", "max", "median"]).dropna()
        if stats.empty:
            continue
        r, g_c, b = int(cor[1:3], 16), int(cor[3:5], 16), int(cor[5:7], 16)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stats.index, y=stats["max"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=stats.index, y=stats["min"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=f"rgba({r},{g_c},{b},0.08)",
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=stats.index, y=stats["mean"], mode="lines+markers",
                                 line=dict(color=cor, width=3), marker=dict(size=5),
                                 name="Média", hovertemplate="%{y:.1f}%"))
        fig.add_trace(go.Scatter(x=stats.index, y=stats["median"], mode="lines",
                                 line=dict(color="#94a3b8", width=1, dash="dot"), name="Mediana"))
        if meta:
            label_meta = f"Limite aceitável: {meta}%" if "perda" in col_name else f"Meta {ANO_META}: {meta}%"
            fig.add_hline(y=meta, line_dash="dash", line_color="#dc2626", line_width=1,
                          annotation_text=label_meta, annotation_font_size=9)
        fig.update_layout(title=dict(text=titulo, font=dict(size=13, color="#0c1d36")),
                          height=300, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis=dict(range=[-5, 110] if "perda" not in col_name else None),
                          legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
                          hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)")
        cols[i % 2].plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key=f"evol_{col_name}")

    delta_agua = (df_hist[df_hist["ano"] == ultimo_ano]["indice_atendimento_total_agua"].mean() -
                  df_hist[df_hist["ano"] == 2014]["indice_atendimento_total_agua"].mean())
    delta_esgoto = (df_hist[df_hist["ano"] == ultimo_ano]["indice_coleta_esgoto"].mean() -
                    df_hist[df_hist["ano"] == 2014]["indice_coleta_esgoto"].mean())
    n_anos = ultimo_ano - 2014
    ritmo_agua = abs(delta_agua) / n_anos if n_anos > 0 else 0
    ritmo_esgoto = abs(delta_esgoto) / n_anos if n_anos > 0 else 0

    media_agua_atual = df_hist[df_hist["ano"] == ultimo_ano]["indice_atendimento_total_agua"].mean()
    media_esgoto_atual = df_hist[df_hist["ano"] == ultimo_ano]["indice_coleta_esgoto"].mean()
    necessario_agua = (99 - media_agua_atual) / (2033 - ultimo_ano) if (2033 - ultimo_ano) > 0 else 0
    necessario_esgoto = (90 - media_esgoto_atual) / (2033 - ultimo_ano) if (2033 - ultimo_ano) > 0 else 0
    fator_insuf = necessario_esgoto / ritmo_esgoto if ritmo_esgoto > 0 else 0

    n_estagnados = 0
    for _, g in df_hist[["ano", "id_municipio", "indice_coleta_esgoto"]].dropna().groupby("id_municipio"):
        if len(g) >= 3:
            anos_g = g["ano"].values.astype(float)
            pesos_g = np.exp(0.3 * (anos_g - anos_g.min()))
            coef = np.polyfit(anos_g, g["indice_coleta_esgoto"].values, 1, w=pesos_g)
            if coef[0] <= 0:
                n_estagnados += 1

    st.markdown(f"""
    <div class="insight">
        Em {n_anos} anos, a cobertura média de água avançou <strong>{abs(delta_agua):.1f} pontos percentuais</strong>
        e a coleta de esgoto apenas <strong>{abs(delta_esgoto):.1f} pp</strong> entre os municípios do RJ
        - menos de meio ponto por ano.
        Para cumprir o Marco Legal até 2033, seriam necessários avanços médios de
        <strong>~{necessario_agua:.1f} pp/ano</strong> em água e <strong>~{necessario_esgoto:.1f} pp/ano</strong> em esgoto.
        O ritmo atual é <strong>{fator_insuf:.0f}x insuficiente</strong> para o esgoto.
        Dos 92 municípios, <strong>{n_estagnados}</strong> estão em retrocesso ou estagnados em coleta de esgoto
        - sem reversão de trajetória, descumprirão a lei.
    </div>
    """, unsafe_allow_html=True)


def render_marco_legal(df, ultimo_ano):
    st.markdown('<div class="section-title">Projeção para o Marco Legal 2033</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight">
        <strong>Metodologia:</strong> Para cada município, é calculada a <strong>tendência linear ponderada</strong>
        (Weighted Least Squares) com base nos dados de 2014 a {ultimo_ano}.
        Anos recentes recebem peso exponencialmente maior, priorizando a trajetória mais atual do município.
        A projeção indica qual seria o valor do indicador em {ANO_META} se a tendência recente se mantiver.
        Municípios em <strong style="color:#7c3aed">retrocesso</strong> apresentam tendência de queda nos últimos anos.
    </div>
    """, unsafe_allow_html=True)

    opcoes = {"Água - meta 99%": ("indice_atendimento_total_agua", META_AGUA),
              "Esgoto - meta 90%": ("indice_coleta_esgoto", META_ESGOTO)}
    escolha = st.selectbox("Selecione o indicador", list(opcoes.keys()), key="marco_indicador")
    indicador, meta = opcoes[escolha]

    df_h = df[df["ano"].between(2014, ultimo_ano)][["ano", "id_municipio", "nome_municipio", indicador]].dropna()
    res = []
    for mid, g in df_h.groupby("id_municipio"):
        if len(g) < 3:
            continue
        nome = g["nome_municipio"].iloc[0]
        anos_m = g["ano"].values.astype(float)
        pesos_m = np.exp(0.3 * (anos_m - anos_m.min()))
        coef = np.polyfit(anos_m, g[indicador].values, 1, w=pesos_m)
        taxa = coef[0]
        atual = float(np.polyval(coef, ultimo_ano))
        proj = float(np.polyval(coef, ANO_META))
        if atual >= meta:
            status = "Já atingiu"
        elif taxa <= 0:
            status = "Em retrocesso"
        else:
            try:
                ano_p = int(np.ceil((meta - coef[1]) / coef[0]))
            except (ZeroDivisionError, OverflowError):
                ano_p = 9999
            status = "No prazo" if ano_p <= ANO_META else "Não atingirá"
        res.append({"Município": nome, "Atual (%)": round(atual, 1), "pp/ano": round(taxa, 2),
                    "Projeção 2033 (%)": round(min(max(proj, 0), 100), 1), "Status": status})

    df_p = pd.DataFrame(res)
    if df_p.empty:
        st.info("Dados insuficientes para projeção.")
        return

    cor_st = {"Já atingiu": "#16a34a", "No prazo": "#2563eb",
              "Não atingirá": "#dc2626", "Em retrocesso": "#7c3aed"}

    def _render_marco_chart(status, container):
        df_s = df_p[df_p["Status"] == status].sort_values("Projeção 2033 (%)")
        if df_s.empty:
            container.info(f"Nenhum município em \"{status}\"")
            return
        cor = cor_st[status]
        container.markdown(f'<div style="margin-top:.5rem;font-weight:700;color:{cor}">{status} ({len(df_s)})</div>',
                           unsafe_allow_html=True)
        fig = px.bar(df_s, x="Projeção 2033 (%)", y="Município",
                     orientation="h",
                     height=max(200, len(df_s) * 22))
        fig.update_traces(marker_color=cor)
        fig.add_vline(x=meta, line_dash="dash", line_color="#dc2626",
                      annotation_text=f"Meta {meta}%", annotation_font_size=9)
        fig.update_layout(margin=dict(l=0, r=10, t=10, b=10), showlegend=False)
        container.plotly_chart(fig, config={"displayModeBar": False}, width="stretch",
                               key=f"marco_{indicador}_{status}")

    col_a, col_b = st.columns(2)
    _render_marco_chart("Já atingiu", col_a)
    _render_marco_chart("No prazo", col_b)

    col_c, col_d = st.columns(2)
    _render_marco_chart("Não atingirá", col_c)
    _render_marco_chart("Em retrocesso", col_d)

    nao = df_p[df_p["Status"].isin(["Não atingirá", "Em retrocesso"])]
    if not nao.empty:
        nomes = ", ".join(nao.nsmallest(5, "Projeção 2033 (%)")["Município"].tolist())
        st.markdown(f"""
        <div class="insight alert">
            <strong>{len(nao)} municípios</strong> não atingirão a meta de {escolha.split(' -')[0].strip().lower()}
            até {ANO_META} no ritmo atual. Projeções mais distantes da meta: <strong>{nomes}</strong>.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Ver tabela completa"):
        st.dataframe(df_p.sort_values("Projeção 2033 (%)"), hide_index=True, width="stretch")
