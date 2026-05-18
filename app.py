import json
import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np

from utils import calc_score, classificar_porte, classificar_saneamento
from tabs.visao_geral import render_hero, render_mapa
from tabs.perfil_municipio import render_perfil
from tabs.classificacao import render_ranking
from tabs.agua_esgoto import render_gap
from tabs.perdas_investimentos import render_perdas
from tabs.evolucao_temporal import render_evolucao, render_marco_legal
from tabs.agrupamento import render_clusters
from tabs.dados import render_dados
from tabs.internacoes import render_internacoes

st.set_page_config(
    page_title="Radar Risco Sanitário - RJ",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #ffffff; }
    .block-container { padding-top: .5rem; max-width: 1280px; }
    .hero {
        background: #fff; color: #1a1a1a; padding: 1.5rem 0 1rem;
        margin-bottom: 1rem;
    }
    .hero h1 { font-size: 1.6rem; font-weight: 800; margin: 0 0 .2rem; color: #1a1a1a; }
    .hero .sub { color: #6b7280; font-size: .8rem; margin-bottom: 1rem; }
    .bn-row { display: flex; gap: .6rem; flex-wrap: wrap; margin: 0; }
    .bn-card {
        flex: 1 1 140px; background: #fff;
        border-radius: 8px; padding: .9rem .8rem;
        border: 1px solid #e5e7eb;
    }
    .bn-card .number { font-size: 1.6rem; font-weight: 800; line-height: 1.1; color: #1a1a1a; }
    .bn-card .label { font-size: .65rem; color: #6b7280; margin-top: .2rem; text-transform: uppercase; letter-spacing: .2px; }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #1a1a1a;
        margin: 1.5rem 0 .4rem;
    }
    .section-subtitle { font-size: .8rem; color: #6b7280; margin: 0 0 .8rem 0; }
    .insight {
        background: #f8fafc; border-radius: 8px;
        padding: .9rem 1rem; margin: .6rem 0; font-size: .82rem;
        line-height: 1.6; color: #374151;
    }
    .insight.alert { background: #fef2f2; }
    .insight.warn  { background: #fffbeb; }
    .insight.ok    { background: #f0fdf4; }
    .insight strong { color: #1a1a1a; }
    .profile-header {
        background: #fff; color: #1a1a1a; border-radius: 8px;
        padding: 1.2rem; margin-bottom: .8rem;
        border: 1px solid #e5e7eb;
    }
    .profile-header h2 { font-size: 1.3rem; font-weight: 800; margin: 0; color: #1a1a1a; }
    .profile-header .badge {
        display: inline-block; margin-top: .4rem; padding: .2rem .6rem;
        border-radius: 4px; font-size: .7rem; font-weight: 700; text-transform: uppercase;
    }
    .profile-grid {
        display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
        gap: .4rem; margin: .6rem 0;
    }
    .profile-metric {
        background: #fff; border-radius: 6px;
        padding: .6rem .5rem; border: 1px solid #e5e7eb;
    }
    .profile-metric .val { font-size: 1.1rem; font-weight: 800; color: #1a1a1a; }
    .profile-metric .lbl { font-size: .6rem; color: #6b7280; text-transform: uppercase; letter-spacing: .2px; margin-top: .1rem; }
    .source { font-size: .68rem; color: #9ca3af; margin-top: .4rem; }
    .footer {
        background: #f8fafc; color: #6b7280;
        padding: 1.2rem; border-radius: 8px; margin-top: 2rem;
        font-size: .72rem; line-height: 1.6;
    }
    .footer a { color: #c2703a; }
    .footer h4 { color: #1a1a1a; margin: 0 0 .4rem; font-size: .82rem; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stExpander"] details summary span[data-testid="stMarkdownContainer"] p { display: inline; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #e5e7eb; }
    .stTabs [data-baseweb="tab"] { font-size: .78rem; font-weight: 600; padding: .5rem 1rem; color: #6b7280; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #c2703a; border-bottom: 2px solid #c2703a; }
    .metric-mini { display: inline-flex; align-items: baseline; gap: .3rem; margin-right: 1.2rem; margin-bottom: .4rem; }
    .metric-mini .val { font-size: 1.4rem; font-weight: 800; color: #1a1a1a; }
    .metric-mini .lbl { font-size: .72rem; color: #6b7280; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------
RENAME_COLS = {
    "Município": "nome_municipio",
    "Sigla UF": "sigla_uf",
    "População Total Residente": "populacao_total_residente",
    "População Rural Residente": "populacao_rural_residente",
    "População Urbana Residente": "populacao_urbana_residente",
    "População total atendida com abastecimento de água": "populacao_atendida_agua",
    "População urbana atendida com rede de abastecimento de água": "populacao_urbana_atendida_agua",
    "Atendimento da população urbana com rede de abastecimento de água": "indice_atendimento_urbano_agua",
    "Atendimento da população total com rede de abastecimento de água": "indice_atendimento_total_agua",
    "População total atendida com esgotamento sanitário": "populacao_atendida_esgoto",
    "População urbana atendida com rede de esgotamento sanitário": "populacao_urbana_atendida_esgoto",
    "Extensão de rede de distribuição de água": "extensao_rede_agua",
    "Extensão da rede pública de esgotamento sanitário": "extensao_rede_esgoto",
    "Quantidade de ligações ativas de água": "ligacoes_ativas_agua",
    "Quantidade de ligações ativas de esgoto": "ligacoes_ativas_esgoto",
    "Volume de água produzido": "volume_agua_produzido",
    "Volume de água tratada em ETAs": "volume_agua_tratada_eta",
    "Volume de água consumido": "volume_agua_consumido",
    "Volume de água tratada por simples desinfecção": "volume_agua_desinfeccao",
    "Volume de água tratada importado": "volume_agua_importado",
    "Volume de água tratada exportado": "volume_agua_exportado",
    "Volume de água fluoretada": "volume_agua_fluoretada",
    "Extensão de rede de distribuição de água.1": "extensao_rede_agua_2",
    "Volume total de esgoto coletado": "volume_esgoto_coletado",
    "Volume total de esgoto tratado ": "volume_esgoto_tratado",
    "Volume total de esgoto bruto exportado ": "volume_esgoto_exportado",
    "Volume total de esgoto bruto importado para tratamento ": "volume_esgoto_importado",
    "Extensão da rede pública de esgotamento sanitário.1": "extensao_rede_esgoto_2",
    "Atendimento da população urbana com rede coletora de esgoto": "indice_coleta_esgoto",
    "Esgoto tratado referido ao esgoto coletado": "indice_tratamento_esgoto",
    "Perda na Distribuição de Água": "indice_perda_distribuicao_agua",
    "Índice de Macromedição": "indice_macromedicao",
    "Índice de Hidrometração": "indice_hidrometracao",
    "Investimento Total (prestador + município + estado)": "investimento_total",
    "Investimento per Capita": "investimento_per_capita",
    "Quantidade de domicílios totais": "domicilios_totais",
    "Quantidade de domicílios urbanos": "domicilios_urbanos",
    "Quantidade de domicílios rurais": "domicilios_rurais",
    "Área do município": "area_municipio",
    "Cobertura da população total com coleta de resíduos sólidos domiciliares": "cobertura_residuos_solidos",
    "Cobertura da população urbana com coleta de resíduos sólidos domiciliares": "cobertura_residuos_urbana",
    "Cobertura da população rural com coleta de resíduos sólidos domiciliares": "cobertura_residuos_rural",
    "Cobertura da população urbana com coleta direta de resíduos sólidos": "cobertura_residuos_urbana_direta",
    "Cobertura da população total com coleta seletiva de resíduos sólidos": "cobertura_coleta_seletiva",
    "Cobertura da população urbana com coleta seletiva direta de resíduos sólidos": "cobertura_coleta_seletiva_urbana",
    "Incidência do transbordo de resíduos sólidos urbanos": "incidencia_transbordo_rsu",
    "Capacidade média utilizada dos veículos motorizados na coleta de RSU": "capacidade_veiculos_coleta",
    "Quantidade média de pontos de entrega voluntária (PEV) de recicláveis por mil habitantes": "pev_por_mil_hab",
    "Massa média per capita de resíduos sólidos urbanos coletados": "massa_rsu_per_capita",
    "Massa média per capita de resíduos sólidos domiciliares coletados": "massa_rsd_per_capita",
    "Massa média per capita de resíduos de limpeza urbana coletados": "massa_limpeza_urbana_per_capita",
    "Massa média per capita de resíduos domiciliares coletados na coleta seletiva": "massa_coleta_seletiva_per_capita",
    "Massa média per capita de resíduos domiciliares secos e orgânicos recuperados": "massa_recuperados_per_capita",
    "Desempenho da coleta seletiva": "desempenho_coleta_seletiva",
    "Disposição final inadequada de resíduos sólidos urbanos": "disposicao_final_inadequada_rsu",
    "Recuperação de recicláveis secos em relação à composição gravimétrica": "recuperacao_secos_gravimetria",
    "Recuperação de recicláveis orgânicos em relação à composição gravimétrica": "recuperacao_organicos_gravimetria",
    "Recuperação de recicláveis secos e orgânicos em relação ao total coletado": "recuperacao_total_coletado",
    "Recuperação de recicláveis secos em relação ao total coletado": "recuperacao_secos_total",
    "Recuperação de recicláveis orgânicos em relação ao total coletado": "recuperacao_organicos_total",
}

@st.cache_data
def load_saneamento():
    path = "data/processed/snis_sinisa_merge_ibge_populacao.csv"
    df = pd.read_csv(path, encoding="utf-8")
    df.rename(columns=RENAME_COLS, inplace=True)
    df["ano"] = df["ano"].astype(int)
    df["id_municipio"] = df["id_municipio"].astype(str).str.strip()
    return df.sort_values(["nome_municipio", "ano"]).reset_index(drop=True)

@st.cache_data
def load_geo():
    p = Path("data/raw/municipios_rj.geojson")
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def enrich_df(df_atual):
    """Adiciona colunas calculadas ao dataframe do ano atual."""
    df_atual["score"] = df_atual.apply(calc_score, axis=1)
    df_atual["porte"] = df_atual["populacao_total_residente"].apply(classificar_porte)

    pop_urb = df_atual["populacao_urbana_residente"]
    pop_tot = df_atual["populacao_total_residente"]
    df_atual["pct_populacao_urbana"] = np.where(
        pop_tot.notna() & (pop_tot > 0),
        (pop_urb / pop_tot * 100).round(1),
        np.nan,
    )
    area = df_atual["area_municipio"]
    df_atual["densidade_demografica"] = np.where(
        area.notna() & (area > 0),
        (pop_tot / area).round(1),
        np.nan,
    )
    df_atual["risco"] = df_atual.apply(classificar_saneamento, axis=1)
    return df_atual

# ── Rodapé ──
def render_footer():
    st.markdown("""
    <div class="footer">
        <div style="display:flex;gap:3rem;flex-wrap:wrap">
            <div style="flex:1;min-width:200px">
                <h4>Sobre</h4>
                O <strong>Radar de Risco Sanitário</strong> consolida dados públicos sobre
                saneamento básico nos 92 municípios do estado do Rio de Janeiro, com foco
                em apoiar decisões de políticas públicas e fiscalização.
            </div>
            <div style="flex:1;min-width:200px">
                <h4>Fontes de Dados</h4>
                • <a href="https://www.gov.br/cidades/pt-br/acesso-a-informacao/acoes-e-programas/saneamento/snis" target="_blank">SNIS</a><br>
                • <a href="https://www.gov.br/cidades/pt-br/acesso-a-informacao/acoes-e-programas/saneamento/sinisa" target="_blank">SINISA</a><br>
                • <a href="https://censo2022.ibge.gov.br/" target="_blank">IBGE - Censo 2022</a><br>
                • <a href="https://sidra.ibge.gov.br/" target="_blank">IBGE SIDRA - PIB municipal</a>
            </div>
            <div style="flex:1;min-width:200px">
                <h4>Metodologia</h4>
                <strong>Índice composto:</strong> média aritmética de atendimento de água, coleta de esgoto,
                tratamento de esgoto e cobertura de resíduos sólidos.<br>
                <strong>Projeções:</strong> regressão linear dos últimos 10 anos.<br>
                <strong>Clusterização:</strong> K-Médias com normalização mínimo-máximo e escolha por silhueta.<br>
                <strong>Marco Legal:</strong> Lei 14.026/2020 - metas de 99% (água) e 90% (esgoto) até 2033.
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,.1);margin:1.5rem 0 .75rem">
        <div style="text-align:center;font-size:.7rem;opacity:.5">
            Radar Risco Sanitário - Rio de Janeiro · 2024–2026
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    df = load_saneamento()
    geo = load_geo()
    ultimo_ano = df["ano"].max()

    df_atual = df[df["ano"] == ultimo_ano].copy()
    df_atual = enrich_df(df_atual)

    tabs = st.tabs([
        "Visão Geral",
        "Perfil do Município",
        "Classificação",
        "Água vs Esgoto",
        "Perdas e Investimento",
        "Evolução Temporal",
        # "Marco Legal 2033",
        "Agrupamento",
        "Internações SUS",
        "Dados",
    ])

    with tabs[0]:
        render_hero(df_atual, ultimo_ano)
        render_mapa(df_atual, df, geo, ultimo_ano, key_suffix="home")

    with tabs[1]:   
        render_perfil(df, df_atual, geo, ultimo_ano)

    with tabs[2]:
        render_ranking(df_atual, ultimo_ano)

    with tabs[3]:
        render_gap(df_atual, ultimo_ano)

    with tabs[4]:
        render_perdas(df, df_atual, ultimo_ano)

    with tabs[5]:
        render_evolucao(df, ultimo_ano)

    # with tabs[X]:
    #     render_marco_legal(df, ultimo_ano)

    with tabs[6]:
        render_clusters(df_atual, geo, ultimo_ano)

    with tabs[7]:
        render_internacoes()

    with tabs[8]:
        render_dados(df)

    render_footer()

if __name__ == "__main__":
    main()
