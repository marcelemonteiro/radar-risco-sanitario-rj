import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from utils import CLUSTER_FEATURES, LABEL

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
        with st.expander(f"{nome_grupo} ({len(muns)} municípios)"):
            st.write(", ".join(muns))
