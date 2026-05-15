import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Radar Risco Sanitário - RJ",
    page_icon="",
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
    .block-container { padding-top: .5rem; max-width: 1280px; }
    .hero {
        background: linear-gradient(135deg, #0c1d36 0%, #183b5c 45%, #1a5276 100%);
        color: white; padding: 2.5rem 2.5rem 2rem; border-radius: 16px;
        margin-bottom: 1.5rem; position: relative; overflow: hidden;
    }
    .hero::after {
        content: ''; position: absolute; top: -40%; right: -10%;
        width: 450px; height: 450px; border-radius: 50%;
        background: radial-gradient(circle, rgba(33,150,243,.12) 0%, transparent 70%);
    }
    .hero h1 { font-size: 2.2rem; font-weight: 900; margin: 0 0 .25rem; letter-spacing: -.5px; }
    .hero .sub { opacity: .7; font-size: .9rem; margin-bottom: 1.2rem; }
    .bn-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: .5rem 0 0; }
    .bn-card {
        flex: 1 1 170px; background: rgba(255,255,255,.07);
        border-radius: 12px; padding: 1.1rem 1rem;
        backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,.1);
    }
    .bn-card .number { font-size: 2rem; font-weight: 900; line-height: 1.1; }
    .bn-card .label { font-size: .72rem; opacity: .65; margin-top: .25rem; text-transform: uppercase; letter-spacing: .3px; }
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #0c1d36;
        padding-left: .75rem;
        margin: 2rem 0 .75rem;
    }
    .section-subtitle { font-size: .85rem; color: #64748b; margin: -.5rem 0 1rem 1rem; }
    .insight {
        background: #f0f4f8; border-radius: 10px;
        padding: 1rem 1.2rem; margin: .75rem 0; font-size: .86rem;
        line-height: 1.6; color: #334155;
    }
    .insight.alert { background: #fef2f2; }
    .insight.warn  { background: #fffbeb; }
    .insight.ok    { background: #f0fdf4; }
    .insight strong { color: #0c1d36; }
    .profile-header {
        background: linear-gradient(135deg, #0c1d36, #1a5276);
        color: #fff; border-radius: 14px; padding: 2rem 2rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .profile-header h2 { font-size: 1.6rem; font-weight: 800; margin: 0; }
    .profile-header .badge {
        display: inline-block; margin-top: .5rem; padding: .2rem .7rem;
        border-radius: 6px; font-size: .72rem; font-weight: 700; text-transform: uppercase;
    }
    .profile-grid {
        display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: .6rem; margin: 1rem 0;
    }
    .profile-metric {
        background: rgba(255,255,255,.06); border-radius: 10px;
        padding: .8rem .7rem; border: 1px solid rgba(255,255,255,.08);
    }
    .profile-metric .val { font-size: 1.3rem; font-weight: 800; }
    .profile-metric .lbl { font-size: .65rem; opacity: .6; text-transform: uppercase; letter-spacing: .3px; margin-top: .15rem; }
    .source { font-size: .7rem; color: #94a3b8; margin-top: .5rem; }
    .footer {
        background: #0c1d36; color: rgba(255,255,255,.6);
        padding: 2rem; border-radius: 14px; margin-top: 3rem;
        font-size: .78rem; line-height: 1.7;
    }
    .footer a { color: #60a5fa; }
    .footer h4 { color: #fff; margin: 0 0 .5rem; font-size: .9rem; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stExpander"] details summary span[data-testid="stMarkdownContainer"] p { display: inline; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; }
    .stTabs [data-baseweb="tab"] { font-size: .82rem; font-weight: 600; padding: .6rem 1.2rem; }
    .metric-mini { display: inline-flex; align-items: baseline; gap: .4rem; margin-right: 1.5rem; margin-bottom: .5rem; }
    .metric-mini .val { font-size: 1.6rem; font-weight: 800; color: #0c1d36; }
    .metric-mini .lbl { font-size: .78rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
LABEL = {
    "indice_atendimento_total_agua": "Atendimento de Água (%)",
    "indice_coleta_esgoto": "Coleta de Esgoto (%)",
    "indice_tratamento_esgoto": "Tratamento de Esgoto (%)",
    "indice_perda_distribuicao_agua": "Perda na Distribuição (%)",
    "cobertura_residuos_solidos": "Cobertura de Resíduos Sólidos (%)",
    "investimento_per_capita": "Investimento per capita (R$/hab)",
    "disposicao_final_inadequada_rsu": "Disposição Final Inadequada (%)",
    "cobertura_coleta_seletiva": "Coleta Seletiva (%)",
    "indice_hidrometracao": "Índice de Hidrometração (%)",
    "indice_macromedicao": "Índice de Macromedição (%)",
    "volume_agua_produzido": "Volume de Água Produzido (m³)",
    "massa_rsu_per_capita": "RSU per capita (kg/hab/dia)",
    "score": "Índice Composto de Saneamento",
    "populacao_urbana_residente": "População Urbana",
    "populacao_rural_residente": "População Rural",
    "area_municipio": "Área (km²)",
}

META_AGUA = 99.0
META_ESGOTO = 90.0
ANO_META = 2033

SCORE_COLS = [
    "indice_atendimento_total_agua", "indice_coleta_esgoto",
    "indice_tratamento_esgoto", "cobertura_residuos_solidos",
]

COR_RISCO = {
    "Atendimento adequado": "#16a34a",
    "Atendimento precário": "#ca8a04",
    "Déficit de esgotamento sanitário": "#ea580c",
    "Déficit severo de esgotamento sanitário": "#dc2626",
    "Déficit de abastecimento de água": "#e11d48",
    "Déficit crítico - água e esgoto": "#7f1d1d",
    "Déficit de manejo de resíduos sólidos": "#d97706",
    "Déficit de manejo de resíduos - contexto rural": "#92400e",
    "Déficit estrutural de saneamento": "#9333ea",
    "Déficit estrutural - município rural vulnerável": "#6b21a8",
    "Sem dados": "#cbd5e1",
}

MAP_INDICATORS = [
    "score", "indice_atendimento_total_agua", "indice_coleta_esgoto",
    "indice_tratamento_esgoto", "indice_perda_distribuicao_agua",
    "cobertura_residuos_solidos", "disposicao_final_inadequada_rsu",
    "investimento_per_capita",
]

CLUSTER_FEATURES = [
    "indice_atendimento_total_agua", "indice_coleta_esgoto",
    "cobertura_residuos_solidos", "disposicao_final_inadequada_rsu",
    "pct_populacao_urbana", "indice_perda_distribuicao_agua",
]

CLUSTER_NAMES = {
    0: "Atendimento adequado consolidado",
    1: "Déficit de esgotamento sanitário",
    2: "Déficit estrutural de saneamento",
    3: "Município rural com déficit múltiplo",
}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def fmt_pop(n, abrev=True):
    if pd.isna(n):
        return "N/D"
    n = float(n)
    if not abrev:
        return f"{n:,.0f}".replace(",", ".")
    if n >= 1e6:
        return f"{n/1e6:.2f} milhões"
    if n >= 1e3:
        return f"{n/1e3:,.0f} mil".replace(",", ".")
    return f"{n:.0f}"


def fmt_area(n):
    if pd.isna(n):
        return "N/D"
    return f"{n:,.0f} km²".replace(",", ".")


def calc_score(row):
    vals = [row[c] for c in SCORE_COLS if pd.notna(row.get(c))]
    return np.mean(vals) if len(vals) >= 2 else np.nan


def classificar_saneamento(row):
    """
    Classificação baseada na terminologia do PLANSAB (Lei 11.445/2007
    atualizada pela Lei 14.026/2020 - Marco Legal do Saneamento).

    Limiares de referência:
    - Água >= 90%: próximo da universalização (meta Marco Legal: 99% até 2033)
    - Esgoto >= 75%: atendimento adequado urbano (meta Marco Legal: 90% até 2033)
    - RSU >= 80%: atendimento adequado (PLANSAB)
    - Água < 60%: sem atendimento ou atendimento precário grave
    - Esgoto < 30%: déficit severo de esgotamento sanitário
    """
    agua = row.get("indice_atendimento_total_agua", np.nan)
    esgoto = row.get("indice_coleta_esgoto", np.nan)
    residuos = row.get("cobertura_residuos_solidos", np.nan)
    pct_urb = row.get("pct_populacao_urbana", np.nan)

    if pd.isna(agua) and pd.isna(esgoto):
        return "Sem dados"

    rural = pd.notna(pct_urb) and pct_urb < 50

    if (pd.notna(agua) and agua >= 90 and
        pd.notna(esgoto) and esgoto >= 75 and
        (pd.isna(residuos) or residuos >= 80)):
        return "Atendimento adequado"

    if (pd.notna(agua) and agua >= 70 and
        pd.notna(esgoto) and esgoto < 50):
        if esgoto < 30:
            return "Déficit severo de esgotamento sanitário"
        return "Déficit de esgotamento sanitário"

    if pd.notna(agua) and agua < 60:
        if pd.notna(esgoto) and esgoto < 30:
            return "Déficit crítico - água e esgoto"
        return "Déficit de abastecimento de água"

    if (pd.notna(residuos) and residuos < 50 and
        pd.notna(agua) and agua >= 70 and
        pd.notna(esgoto) and esgoto >= 50):
        if rural:
            return "Déficit de manejo de resíduos - contexto rural"
        return "Déficit de manejo de resíduos sólidos"

    n_deficit = sum([
        pd.notna(agua) and agua < 70,
        pd.notna(esgoto) and esgoto < 50,
        pd.notna(residuos) and residuos < 50,
    ])
    if n_deficit >= 2:
        if rural:
            return "Déficit estrutural - município rural vulnerável"
        return "Déficit estrutural de saneamento"

    return "Atendimento precário"


def classificar_porte(pop):
    if pd.isna(pop):
        return "Sem dados"
    if pop > 500_000:
        return "Grande Porte"   
    if pop > 100_000:
        return "Médio-grande Porte"
    if pop > 20_000:
        return "Médio Porte"
    return "Pequeno Porte"


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


# ---------------------------------------------------------------------------
# Seções do dashboard
# ---------------------------------------------------------------------------

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


# ── Perfil reutilizável ──
def _render_municipio_card(row, df_atual, df_hist=None, container=None):
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

    interpretacoes = []
    if pd.notna(agua):
        if agua >= META_AGUA:
            interpretacoes.append(f"Já atingiu a meta do Marco Legal para água ({META_AGUA}%).")
        elif agua >= media_agua:
            interpretacoes.append(f"Atendimento de água ({agua:.1f}%) acima da média estadual ({media_agua:.1f}%), mas ainda distante da meta de {META_AGUA}%.")
        else:
            interpretacoes.append(f"Atendimento de água ({agua:.1f}%) <strong>abaixo da média estadual</strong> ({media_agua:.1f}%).")

    if pd.notna(esgoto):
        if esgoto >= META_ESGOTO:
            interpretacoes.append(f"Coleta de esgoto ({esgoto:.1f}%) atinge a meta do Marco Legal ({META_ESGOTO}%).")
        elif esgoto < 30:
            interpretacoes.append(f"Coleta de esgoto de apenas <strong>{esgoto:.1f}%</strong> - situação que demanda atenção prioritária.")
        else:
            cmp = "acima" if esgoto >= media_esgoto else "<strong>abaixo</strong>"
            interpretacoes.append(f"Coleta de esgoto ({esgoto:.1f}%) {cmp} da média estadual ({media_esgoto:.1f}%).")

    if pd.notna(esgoto) and pd.notna(pop) and pop > 0:
        sem_esgoto = pop * (1 - esgoto / 100)
        if sem_esgoto > 1000:
            interpretacoes.append(f"Aproximadamente <strong>{fmt_pop(sem_esgoto, abrev=False)} habitantes</strong> sem coleta de esgoto.")

    gap = (agua - esgoto) if pd.notna(agua) and pd.notna(esgoto) else np.nan
    if pd.notna(gap) and gap > 30:
        interpretacoes.append(f"Descompasso de <strong>{gap:.0f} pontos percentuais</strong> entre água e esgoto.")
    if pd.notna(perda) and perda > 40:
        interpretacoes.append(f"Perda na distribuição de <strong>{perda:.0f}%</strong> - patamar considerado elevado pelo SNIS.")

    if interpretacoes:
        body = " · ".join(interpretacoes)
        tipo = "alert" if any("prioritária" in i for i in interpretacoes) else ""
        t.markdown(f'<div class="insight {tipo}" style="font-size:.82rem">{body}</div>', unsafe_allow_html=True)

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
                        name=LABEL.get(c, c), line=dict(color=cores.get(c), width=2.5),
                        marker=dict(size=5), hovertemplate="%{y:.1f}%",
                    ))
                fig.add_hline(y=META_AGUA, line_dash="dot", line_color="#64748b", line_width=1,
                              annotation_text="Meta Água 99%", annotation_font_size=9)
                fig.add_hline(y=META_ESGOTO, line_dash="dot", line_color="#64748b", line_width=1,
                              annotation_text="Meta Esgoto 90%", annotation_font_size=9)
                fig.update_layout(
                    height=300, margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", y=-0.18, font=dict(size=9)),
                    hovermode="x unified", yaxis=dict(range=[-5, 110]),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                t.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")


# ── Mapa interativo ──
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

    col_map, col_info = st.columns([2.2, 1])
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
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
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
                _render_municipio_card(rows.iloc[0], df_atual, df_hist, container=col_info)
        else:
            st.markdown(f"""
<div class="insight" style="margin-top:0">
    <strong>Resumo estadual - {label_ind}</strong><br><br>
    <span class="metric-mini"><span class="val">{media:.1f}</span><span class="lbl">Média</span></span>
    <span class="metric-mini"><span class="val">{mediana:.1f}</span><span class="lbl">Mediana</span></span><br>
    <span class="metric-mini"><span class="val">{std:.1f}</span><span class="lbl">Desvio padrão</span></span>
    <span class="metric-mini"><span class="val">{n_abaixo_meta}</span><span class="lbl">Abaixo de 50%</span></span>
    <br><br>
    <strong>Maior índice:</strong> {mais_elevado['nome_municipio']} ({mais_elevado[indicador]:.1f})<br>
    <strong>Menor índice:</strong> {menos_elevado['nome_municipio']} ({menos_elevado[indicador]:.1f})<br><br>
    <em style="font-size:.78rem;color:#94a3b8">Clique em um município no mapa para ver seu perfil detalhado.</em>
</div>
            """, unsafe_allow_html=True)

    st.markdown(f'<div class="source">Fonte: SNIS/SINISA ({ultimo_ano}). {len(df_mapa)} municípios com dados disponíveis.</div>',
                unsafe_allow_html=True)


# ── Perfil do Município ──
def render_perfil(df, df_atual, geojson, ultimo_ano):
    st.markdown('<div class="section-title">Perfil do Município</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Selecione um município para ver seus indicadores detalhados e evolução histórica</div>',
                unsafe_allow_html=True)
    nomes = sorted(df_atual["nome_municipio"].dropna().unique())
    mun = st.selectbox("Município", nomes, index=nomes.index("Rio de Janeiro") if "Rio de Janeiro" in nomes else 0)
    row = df_atual[df_atual["nome_municipio"] == mun].iloc[0]
    _render_municipio_card(row, df_atual, df_hist=df)


# ── Ranking com porte ──
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


# ── Descompasso Água vs Esgoto ──
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


# ── Perdas e Investimento ──
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
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10),
                          coloraxis_colorbar=dict(title="Perda %", thickness=12, len=0.5))
        fig.add_hline(y=40, line_dash="dash", line_color="#dc2626", line_width=1,
                      annotation_text="Patamar elevado 40%", annotation_font_size=9)
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


# ── Evolução Temporal ──
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
        ("indice_perda_distribuicao_agua", "Perda na Distribuição", "#ef4444", None),
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
            fig.add_hline(y=meta, line_dash="dash", line_color="#dc2626", line_width=1,
                          annotation_text=f"Meta {ANO_META}: {meta}%", annotation_font_size=9)
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
            coef = np.polyfit(g["ano"].values.astype(float), g["indice_coleta_esgoto"].values, 1)
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


# ── Marco Legal 2033 ──
def render_marco_legal(df, ultimo_ano):
    st.markdown('<div class="section-title">Projeção para o Marco Legal 2033</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight">
        <strong>Metodologia:</strong> Para cada município, é calculada a <strong>tendência linear</strong>
        (regressão de 1º grau) com base nos dados dos últimos 10 anos (2014-{ultimo_ano}).
        A projeção indica qual seria o valor do indicador em {ANO_META} se a tendência atual se mantiver.
        Municípios em <strong style="color:#7c3aed">retrocesso</strong> apresentam tendência de queda.
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
        coef = np.polyfit(g["ano"].values.astype(float), g[indicador].values, 1)
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

    with st.expander("Ver tabela completa", icon=""):
        st.dataframe(df_p.sort_values("Projeção 2033 (%)"), hide_index=True, width="stretch")


# ── Clusterização K-Means ──
@st.cache_data
def compute_clusters(df_atual):
    """Executa K-Means conforme especificação do regras.md (seção 4)."""
    feat_cols = [c for c in CLUSTER_FEATURES if c in df_atual.columns]
    df_cl = df_atual[["id_municipio", "nome_municipio"] + feat_cols].copy()

    n_missing = df_cl[feat_cols].isna().sum(axis=1)
    df_cl.loc[n_missing <= 2, feat_cols] = df_cl.loc[n_missing <= 2, feat_cols].fillna(
        df_cl[feat_cols].median()
    )
    df_cl = df_cl.dropna(subset=feat_cols).reset_index(drop=True)

    if len(df_cl) < 10:
        return None, None, None, None

    X = df_cl[feat_cols].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    sil_scores = []
    K_range = range(2, 7)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    best_k = list(K_range)[np.argmax(sil_scores)]
    if best_k < 3:
        best_k = 4

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_cl["cluster"] = km_final.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_cl["pca1"] = coords[:, 0]
    df_cl["pca2"] = coords[:, 1]

    profiles = df_cl.groupby("cluster")[feat_cols].mean().round(1)

    elbow_data = pd.DataFrame({"k": list(K_range), "Inércia": inertias,
                                "Silhueta": sil_scores})

    return df_cl, profiles, elbow_data, best_k


def _nome_cluster(profile_row, cluster_id):
    """
    Nomeia o cluster com base no perfil médio.
    Terminologia PLANSAB (Lei 11.445/2007, atualizada pela Lei 14.026/2020).
    """
    agua = profile_row.get("indice_atendimento_total_agua", 0)
    esgoto = profile_row.get("indice_coleta_esgoto", 0)
    residuos = profile_row.get("cobertura_residuos_solidos", 0)
    pct_urb = profile_row.get("pct_populacao_urbana", 50)
    rural = pct_urb < 50

    if agua >= 90 and esgoto >= 75 and residuos >= 80:
        return "Atendimento adequado consolidado"
    if agua >= 70 and esgoto < 50:
        if esgoto < 30:
            return "Déficit severo de esgotamento sanitário"
        return "Déficit de esgotamento sanitário"
    if agua < 60:
        return "Déficit de abastecimento de água"
    n_deficit = sum([agua < 70, esgoto < 50, residuos < 50])
    if n_deficit >= 2:
        if rural:
            return "Déficit estrutural - município rural vulnerável"
        return "Déficit estrutural de saneamento"
    return "Atendimento precário"


def render_clusters(df_atual, geojson, ultimo_ano):
    st.markdown('<div class="section-title">Agrupamento de Municípios (Clusterização)</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight">
        <strong>Metodologia:</strong> Os municípios são agrupados pelo algoritmo <strong>K-Médias</strong>
        (K-Means), uma técnica de aprendizado de máquina não supervisionado que identifica padrões
        em conjuntos de dados. O algoritmo divide os municípios em grupos com características
        semelhantes de saneamento.<br><br>
        <strong>Variáveis utilizadas:</strong> atendimento de água, coleta de esgoto, cobertura de resíduos sólidos,
        disposição inadequada de resíduos, percentual de população urbana e perda na distribuição de água.
        Todas as variáveis são normalizadas para uma escala de 0 a 1 (normalização mínimo-máximo)
        antes da aplicação do algoritmo.<br><br>
        <strong>Escolha do número de grupos:</strong> Testam-se de 2 a 6 grupos. A escolha final é
        baseada na <strong>pontuação de silhueta</strong>, que mede a coesão e separação entre os grupos
        (quanto maior, mais distintos os grupos).
    </div>
    """, unsafe_allow_html=True)

    result = compute_clusters(df_atual)
    if result[0] is None:
        st.warning("Dados insuficientes para realizar a clusterização.")
        return

    df_cl, profiles, elbow_data, best_k = result

    cluster_labels = {}
    for cid in profiles.index:
        cluster_labels[cid] = _nome_cluster(profiles.loc[cid], cid)
    df_cl["grupo"] = df_cl["cluster"].map(cluster_labels)

    st.markdown(f"**Número de grupos selecionado: {best_k}**")

    col_elbow, col_sil = st.columns(2)
    with col_elbow:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=elbow_data["k"], y=elbow_data["Inércia"], mode="lines+markers",
                                 line=dict(color="#3b82f6", width=2), marker=dict(size=7)))
        fig.update_layout(title="Método do Cotovelo (Inércia)", height=280,
                          margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title="Número de grupos (k)", yaxis_title="Inércia")
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key="cl_elbow")

    with col_sil:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=elbow_data["k"], y=elbow_data["Silhueta"],
                             marker_color=["#16a34a" if k == best_k else "#94a3b8"
                                           for k in elbow_data["k"]]))
        fig.update_layout(title="Pontuação de Silhueta", height=280,
                          margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title="Número de grupos (k)", yaxis_title="Silhueta")
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key="cl_silhueta")

    cores_cluster = px.colors.qualitative.Set2[:best_k]
    cor_map = {cluster_labels.get(i, f"Grupo {i}"): cores_cluster[i % len(cores_cluster)]
               for i in range(best_k)}

    col_pca, col_map = st.columns([1, 1.5])

    with col_pca:
        st.markdown("**Visualização em 2 dimensões (ACP)**")
        st.markdown("""<div class="source" style="margin-bottom:.5rem">
            A Análise de Componentes Principais (ACP) reduz as 6 variáveis a 2 eixos
            para permitir a visualização. Cada ponto é um município.
        </div>""", unsafe_allow_html=True)
        fig = px.scatter(df_cl, x="pca1", y="pca2", color="grupo", hover_name="nome_municipio",
                         color_discrete_map=cor_map,
                         labels={"pca1": "Componente 1", "pca2": "Componente 2", "grupo": "Grupo"})
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=-0.12, font=dict(size=10)))
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key="cl_pca")

    with col_map:
        if geojson:
            st.markdown("**Mapa dos grupos**")
            fig = px.choropleth_map(
                df_cl, geojson=geojson, locations="id_municipio",
                featureidkey="properties.codarea", color="grupo",
                color_discrete_map=cor_map, hover_name="nome_municipio",
                center={"lat": -22.25, "lon": -43.3}, zoom=6.5, opacity=0.85,
                labels={"grupo": "Grupo"})
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key="cl_mapa")

    st.markdown("**Perfil médio de cada grupo**")
    st.markdown("""<div class="source" style="margin-bottom:.5rem">
        Valores médios de cada indicador por grupo. Permite identificar as
        características predominantes de cada agrupamento.
    </div>""", unsafe_allow_html=True)

    profiles_display = profiles.copy()
    profiles_display.index = [cluster_labels.get(i, f"Grupo {i}") for i in profiles_display.index]
    profiles_display = profiles_display.rename(columns=LABEL)
    st.dataframe(profiles_display, width="stretch")

    st.markdown("**Municípios por grupo**")
    for cid in sorted(df_cl["cluster"].unique()):
        nome_grupo = cluster_labels.get(cid, f"Grupo {cid}")
        muns = sorted(df_cl[df_cl["cluster"] == cid]["nome_municipio"].tolist())
        with st.expander(f"{nome_grupo} ({len(muns)} municípios)", icon="📌"):
            st.write(", ".join(muns))


# ── Dados ──
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
        # "Visão Geral",
        # "Perfil do Município",
        # "Classificação",
        "Água vs Esgoto",
        "Perdas e Investimento",
        "Evolução Temporal",
        "Marco Legal 2033",
        "Agrupamento",
        "Dados",
    ])

    # with tabs[X]:
    #     render_hero(df_atual, ultimo_ano)
    #     render_mapa(df_atual, df, geo, ultimo_ano, key_suffix="home")

    # with tabs[X]:
    #     render_perfil(df, df_atual, geo, ultimo_ano)

    # with tabs[X]:
    #     render_ranking(df_atual, ultimo_ano)

    with tabs[0]:
        render_gap(df_atual, ultimo_ano)

    with tabs[1]:
        render_perdas(df, df_atual, ultimo_ano)

    with tabs[2]:
        render_evolucao(df, ultimo_ano)

    with tabs[3]:
        render_marco_legal(df, ultimo_ano)

    with tabs[4]:
        render_clusters(df_atual, geo, ultimo_ano)

    with tabs[5]:
        render_dados(df)

    render_footer()


if __name__ == "__main__":
    main()
