import plotly.express as px
import streamlit as st

def render_ranking(df_atual, ultimo_ano):
    st.markdown('<div class="section-title">Classificação dos Municípios</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="section-subtitle">
        Classificação de {ultimo_ano} com base no índice composto de saneamento
        (média de água, esgoto, tratamento e resíduos).
        Municípios agrupados por porte populacional conforme Censo 2022.
    </div>""", unsafe_allow_html=True)

    df_rank = df_atual.dropna(subset=["score"]).sort_values("score").copy()
    n = len(df_rank)
    cnt = df_rank["risco"].value_counts()

    n_adequado = cnt.get("Atendimento adequado", 0)
    n_precario = cnt.get("Atendimento precário", 0)
    n_deficit = sum(v for k, v in cnt.items() if "Déficit" in k)

    c1, c2, c3 = st.columns(3)
    for col, (lbl, val, cor) in zip([c1, c2, c3], [
        ("Atendimento adequado", n_adequado, "#16a34a"),
        ("Atendimento precário", n_precario, "#ca8a04"),
        ("Com algum déficit", n_deficit, "#dc2626"),
    ]):
        col.markdown(f"""
        <div style="text-align:center;padding:.7rem;border-radius:10px;background:{cor}12;border:1px solid {cor}30">
            <div style="font-size:2rem;font-weight:900;color:{cor}">{val}</div>
            <div style="font-size:.72rem;color:#64748b;text-transform:uppercase">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight">
        <strong>Metodologia:</strong> O índice composto é calculado como a <strong>média aritmética</strong> de 4 indicadores:
        atendimento de água, coleta de esgoto, tratamento de esgoto e cobertura de resíduos sólidos (escala 0-100).
        Municípios com dados disponíveis em pelo menos 2 dos 4 indicadores são incluídos.<br><br>
        <strong>Classificação (PLANSAB):</strong> Dos <strong>{n} municípios</strong> avaliados em {ultimo_ano},
        <strong>{n_adequado}</strong> têm <strong style="color:#16a34a">atendimento adequado</strong> e
        <strong>{n_deficit}</strong> apresentam algum tipo de <strong style="color:#dc2626">déficit</strong>
        (esgotamento sanitário, abastecimento de água, resíduos sólidos ou estrutural).
    </div>
    """, unsafe_allow_html=True)

    porte_sel = st.radio("Filtrar por porte", ["Todos", "Grande", "Médio-grande", "Médio", "Pequeno"], horizontal=True)
    df_vis = df_rank if porte_sel == "Todos" else df_rank[df_rank["porte"] == porte_sel]

    col_p, col_m = st.columns(2)
    with col_p:
        st.markdown("##### 20 municípios com menor índice")
        d20 = df_vis.head(20)
        fig = px.bar(d20, x="score", y="nome_municipio", orientation="h",
                     color="score", color_continuous_scale="RdYlGn", range_color=[0, 100],
                     labels={"score": "Índice", "nome_municipio": ""})
        fig.update_layout(height=550, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key="rank_menor")

    with col_m:
        st.markdown("##### 20 municípios com maior índice")
        d20m = df_vis.tail(20).sort_values("score", ascending=True)
        fig = px.bar(d20m, x="score", y="nome_municipio", orientation="h",
                     color="score", color_continuous_scale="RdYlGn", range_color=[0, 100],
                     labels={"score": "Índice", "nome_municipio": ""})
        fig.update_layout(height=550, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key="rank_maior")

    com_deficit = df_vis[df_vis["risco"].str.contains("Déficit", na=False)]
    if not com_deficit.empty:
        nomes = ", ".join(sorted(com_deficit.nsmallest(10, "score")["nome_municipio"].tolist()))
        st.markdown(f'<div class="insight alert"><strong>Municípios com déficit mais severo:</strong> {nomes}</div>',
                    unsafe_allow_html=True)
