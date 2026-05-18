import streamlit as st
from utils import LABEL

def render_dados(df):
    st.markdown('<div class="section-title">Base de Dados Completa</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Explore e baixe os dados consolidados de todos os municípios</div>',
                unsafe_allow_html=True)
    ano = st.slider("Ano", int(df["ano"].min()), int(df["ano"].max()), int(df["ano"].max()))
    d = df[df["ano"] == ano].copy()
    vis = ["nome_municipio"] + [c for c in d.columns
           if c not in ("ano", "id_municipio", "nome_municipio", "sigla_uf") and d[c].notna().any()]
    st.dataframe(d[vis].rename(columns=LABEL).sort_values(vis[0]),
                 hide_index=True, height=600, width="stretch")
    st.download_button("Baixar dados em CSV", d.to_csv(index=False), f"saneamento_rj_{ano}.csv", "text/csv")
