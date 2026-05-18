import plotly.express as px
import streamlit as st

def render_perdas(df, df_atual, ultimo_ano):
    st.markdown('<div class="section-title">Perdas na Distribuição e Investimento</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Relação entre investimento per capita e perdas de água nos municípios fluminenses</div>',
                unsafe_allow_html=True)

    d = df_atual.dropna(subset=["indice_perda_distribuicao_agua"]).copy()
    media_perda = d["indice_perda_distribuicao_agua"].mean()
    mais_perda = d.loc[d["indice_perda_distribuicao_agua"].idxmax()]
    menos_perda = d.loc[d["indice_perda_distribuicao_agua"].idxmin()]
    acima_40 = (d["indice_perda_distribuicao_agua"] > 40).sum()

    col_txt, col_g = st.columns([1, 2])
    col_txt.markdown(f"""
A **perda na distribuição** mede o percentual de água tratada que é desperdiçada
antes de chegar ao consumidor - seja por vazamentos, ligações clandestinas
ou problemas de medição.

Em {ultimo_ano}, a perda média no estado do RJ foi de **{media_perda:.1f}%**.
O município com maior perda é **{mais_perda['nome_municipio']}** ({mais_perda['indice_perda_distribuicao_agua']:.0f}%),
enquanto **{menos_perda['nome_municipio']}** registra a menor ({menos_perda['indice_perda_distribuicao_agua']:.0f}%).

**{acima_40} municípios** perdem mais de 40% da água tratada.
    """)

    d_both = d.dropna(subset=["investimento_per_capita", "indice_perda_distribuicao_agua"])
    if len(d_both) > 5:
        fig = px.scatter(
            d_both, x="investimento_per_capita", y="indice_perda_distribuicao_agua",
            size="populacao_total_residente", size_max=30,
            hover_name="nome_municipio", color="indice_perda_distribuicao_agua",
            color_continuous_scale="RdYlGn_r",
            labels={"investimento_per_capita": "Investimento per capita (R$/hab)",
                    "indice_perda_distribuicao_agua": "Perda na Distribuição (%)",
                    "populacao_total_residente": "População"})
        fig.add_hrect(y0=25, y1=d_both["indice_perda_distribuicao_agua"].max() + 5,
                      fillcolor="#dc2626", opacity=0.06, line_width=0)
        fig.add_hline(y=25, line_dash="solid", line_color="#dc2626", line_width=1.5,
                      annotation_text="Limite aceitável 25% (ANA/SNIS)", annotation_font_size=9,
                      annotation_position="top left")
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10),
                          coloraxis_colorbar=dict(title="Perda %", thickness=12, len=0.5))
        col_g.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key="perda_scatter")
    else:
        fig = px.bar(d.nlargest(25, "indice_perda_distribuicao_agua"),
                     x="indice_perda_distribuicao_agua", y="nome_municipio", orientation="h",
                     color="indice_perda_distribuicao_agua", color_continuous_scale="RdYlGn_r",
                     labels={"indice_perda_distribuicao_agua": "Perda (%)", "nome_municipio": ""})
        fig.update_layout(height=500, margin=dict(l=0, r=10, t=10, b=0), coloraxis_showscale=False)
        col_g.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key="perda_bar")

    col_g.markdown(f'<div class="source">Fonte: SNIS ({ultimo_ano}). Tamanho das bolhas proporcional à população.</div>',
                   unsafe_allow_html=True)

    acima_25 = (d["indice_perda_distribuicao_agua"] > 25).sum()
    vol_produzido = d["volume_agua_produzido"].sum() if "volume_agua_produzido" in d.columns else 0
    excesso_perda = (d["indice_perda_distribuicao_agua"].mean() - 25) / 100
    vol_perdido_excesso = vol_produzido * excesso_perda if excesso_perda > 0 else 0
    consumo_per_capita_dia = 0.15
    pop_equiv = vol_perdido_excesso / (consumo_per_capita_dia * 365) if consumo_per_capita_dia > 0 else 0

    st.markdown(f"""
    <div class="insight alert">
        <strong>{acima_25} municípios</strong> estão acima do limite de 25% de perda (referência ANA/SNIS), que indica
        ineficiência operacional. Com perda média de <strong>{media_perda:.0f}%</strong>, o volume desperdiçado acima
        do aceitável equivale ao abastecimento de aproximadamente
        <strong>{pop_equiv/1e6:.1f} milhões de pessoas</strong>.
    </div>
    """, unsafe_allow_html=True)

    hist = df[df["ano"].between(2014, ultimo_ano)]
    stats = hist.groupby("ano")["indice_perda_distribuicao_agua"].mean().dropna()
    if len(stats) > 2:
        delta = stats.iloc[-1] - stats.iloc[0]
        st.markdown(f"""
        <div class="insight {"ok" if delta < 0 else "warn"}">
            Entre {int(stats.index.min())} e {int(stats.index.max())}, a perda média estadual
            {"<strong>reduziu</strong>" if delta < 0 else "<strong>aumentou</strong>"} em
            <strong>{abs(delta):.1f} pontos percentuais</strong>.
            {"Embora haja redução, o nível atual ainda é elevado." if delta < 0 else "A tendência de aumento demanda atenção."}
        </div>
        """, unsafe_allow_html=True)
