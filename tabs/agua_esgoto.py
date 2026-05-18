import plotly.graph_objects as go
import streamlit as st

def render_gap(df_atual, ultimo_ano):
    st.markdown('<div class="section-title">Diferença entre cobertura de abastecimento de água e coleta de esgoto em 2024</div>', unsafe_allow_html=True)

    df_gap = df_atual.dropna(subset=["indice_atendimento_total_agua", "indice_coleta_esgoto"]).copy()
    df_gap["gap"] = df_gap["indice_atendimento_total_agua"] - df_gap["indice_coleta_esgoto"]
    df_gap = df_gap.sort_values("gap", ascending=False)
    grande_gap = df_gap[df_gap["gap"] > 40]

    col_t, col_g = st.columns([1, 2])
    col_t.markdown(f"""
Um dos problemas mais graves do saneamento fluminense é o **descompasso
entre água e esgoto**. Municípios com redes de água extensas muitas vezes
não possuem infraestrutura correspondente de coleta de esgoto.

Em {ultimo_ano}, **{len(grande_gap)} municípios** apresentam diferença
superior a **40 pontos percentuais**. A diferença média estadual é de
**{df_gap['gap'].mean():.0f} pp**.
    """)

    if not grande_gap.empty:
        top = grande_gap.iloc[0]
        col_t.markdown(f"""
<div class="insight warn">
    <strong>{top['nome_municipio']}</strong> apresenta a maior diferença:
    <strong>{top['indice_atendimento_total_agua']:.0f}%</strong> de água
    contra <strong>{top['indice_coleta_esgoto']:.0f}%</strong> de esgoto
    ({top['gap']:.0f} pp de diferença).
</div>
        """, unsafe_allow_html=True)

    df_top = df_gap.head(20)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_top["nome_municipio"], x=df_top["indice_atendimento_total_agua"],
                         name="Água (%)", orientation="h"))
    fig.add_trace(go.Bar(y=df_top["nome_municipio"], x=df_top["indice_coleta_esgoto"],
                         name="Esgoto (%)", orientation="h", marker_color="#f97316"))
    fig.update_layout(barmode="overlay", height=550, legend=dict(orientation="h", y=-0.06),
                      margin=dict(l=0, r=10, t=10, b=30), xaxis_title="Cobertura (%)")
    col_g.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key="gap_chart")
    col_g.markdown(f'<div class="source">Fonte: SNIS/SINISA ({ultimo_ano}). Diferença = Água − Esgoto.</div>',
                   unsafe_allow_html=True)
