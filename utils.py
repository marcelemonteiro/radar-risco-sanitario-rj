import numpy as np
import pandas as pd

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
