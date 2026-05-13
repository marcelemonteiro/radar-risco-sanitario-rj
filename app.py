import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Radar Saneamento RJ",
    page_icon="💧",
    layout="wide",
)

COLUMN_NAMES = {
    "ano": "Ano",
    "id_municipio": "Código IBGE",
    "nome_municipio": "Município",
    "sigla_uf": "UF",
    "populacao_total_residente": "População Total",
    "populacao_urbana": "População Urbana",
    "populacao_atendida_agua": "Pop. Atendida Água",
    "populacao_urbana_atendida_agua": "Pop. Urbana Atendida Água",
    "indice_atendimento_urbano_agua": "Atendimento Urbano Água (%)",
    "indice_atendimento_total_agua": "Atendimento Total Água (%)",
    "populacao_atentida_esgoto": "Pop. Atendida Esgoto",
    "populacao_urbana_atendida_esgoto": "Pop. Urbana Atendida Esgoto",
    "extensao_rede_agua": "Extensão Rede Água (km)",
    "extensao_rede_esgoto": "Extensão Rede Esgoto (km)",
    "quantidade_ligacao_ativa_agua": "Ligações Ativas Água",
    "quantidade_ligacao_ativa_esgoto": "Ligações Ativas Esgoto",
    "volume_agua_produzido": "Vol. Água Produzido (mil m³/ano)",
    "volume_agua_tratada_eta": "Vol. Água Tratada ETA (mil m³/ano)",
    "volume_agua_consumido": "Vol. Água Consumido (mil m³/ano)",
    "volume_agua_tratada_desinfeccao": "Vol. Água Desinfecção (mil m³/ano)",
    "volume_agua_tratada_importado": "Vol. Água Importado (mil m³/ano)",
    "volume_agua_tratada_exportado": "Vol. Água Exportado (mil m³/ano)",
    "volume_agua_fluoretada": "Vol. Água Fluoretada (mil m³/ano)",
    "indice_extensao_agua_ligacao": "Extensão Rede Água (km) [2]",
    "volume_esgoto_coletado": "Vol. Esgoto Coletado (mil m³/ano)",
    "volume_esgoto_tratado": "Vol. Esgoto Tratado (mil m³/ano)",
    "volume_esgoto_bruto_exportado": "Vol. Esgoto Exportado (mil m³/ano)",
    "volume_esgoto_importado": "Vol. Esgoto Importado (mil m³/ano)",
    "indice_extensao_esgoto_ligacao": "Extensão Rede Esgoto (km) [2]",
    "indice_coleta_esgoto": "Coleta de Esgoto (%)",
    "indice_tratamento_esgoto": "Tratamento de Esgoto (%)",
}

INDICADORES_PERCENTUAIS = [
    "indice_atendimento_urbano_agua",
    "indice_atendimento_total_agua",
    "indice_coleta_esgoto",
    "indice_tratamento_esgoto",
]

INDICADORES_VOLUME = [
    "volume_agua_produzido",
    "volume_agua_tratada_eta",
    "volume_agua_consumido",
    "volume_esgoto_coletado",
    "volume_esgoto_tratado",
]

INDICADORES_INFRA = [
    "extensao_rede_agua",
    "extensao_rede_esgoto",
    "quantidade_ligacao_ativa_agua",
    "quantidade_ligacao_ativa_esgoto",
]

INDICADORES_POP = [
    "populacao_atendida_agua",
    "populacao_urbana_atendida_agua",
    "populacao_atentida_esgoto",
    "populacao_urbana_atendida_esgoto",
]


def parse_br_number(val):
    """Converte número em formato brasileiro (1.000,50) para float."""
    if pd.isna(val) or val == "":
        return None
    s = str(val).strip().strip('"')
    if s == "" or s == " ":
        return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/processed/dados_saneamento_snis_sinisa.csv",
        skiprows=[1, 2],
        dtype=str,
        encoding="utf-8",
    )

    col_names = [
        "ano", "id_municipio", "nome_municipio", "sigla_uf",
        "populacao_total_residente", "populacao_urbana",
        "populacao_atendida_agua", "populacao_urbana_atendida_agua",
        "indice_atendimento_urbano_agua", "indice_atendimento_total_agua",
        "populacao_atentida_esgoto", "populacao_urbana_atendida_esgoto",
        "extensao_rede_agua", "extensao_rede_esgoto",
        "quantidade_ligacao_ativa_agua", "quantidade_ligacao_ativa_esgoto",
        "volume_agua_produzido", "volume_agua_tratada_eta",
        "volume_agua_consumido", "volume_agua_tratada_desinfeccao",
        "volume_agua_tratada_importado", "volume_agua_tratada_exportado",
        "volume_agua_fluoretada", "indice_extensao_agua_ligacao",
        "volume_esgoto_coletado", "volume_esgoto_tratado",
        "volume_esgoto_bruto_exportado", "volume_esgoto_importado",
        "indice_extensao_esgoto_ligacao", "indice_coleta_esgoto",
        "indice_tratamento_esgoto",
    ]
    df.columns = col_names

    numeric_cols = [c for c in col_names if c not in ("ano", "id_municipio", "nome_municipio", "sigla_uf")]
    for col in numeric_cols:
        df[col] = df[col].apply(parse_br_number)

    df["ano"] = df["ano"].astype(int)
    df = df.sort_values(["nome_municipio", "ano"]).reset_index(drop=True)

    return df


def main():
    st.title("🚰 Radar de Saneamento — Rio de Janeiro")
    st.markdown("Série histórica **1995–2024** · Dados SNIS + SINISA")

    df = load_data()

    # --- Sidebar ---
    st.sidebar.header("Filtros")

    municipios = sorted(df["nome_municipio"].dropna().unique())
    municipios_sel = st.sidebar.multiselect(
        "Municípios",
        municipios,
        default=["Rio de Janeiro", "Niterói", "Nova Iguaçu"],
    )

    ano_min, ano_max = int(df["ano"].min()), int(df["ano"].max())
    ano_range = st.sidebar.slider("Período", ano_min, ano_max, (2010, ano_max))

    df_filtered = df[
        (df["nome_municipio"].isin(municipios_sel)) &
        (df["ano"] >= ano_range[0]) &
        (df["ano"] <= ano_range[1])
    ]

    if df_filtered.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
        return

    # --- KPIs do último ano disponível ---
    st.markdown("---")
    ultimo_ano = df_filtered["ano"].max()
    df_ultimo = df_filtered[df_filtered["ano"] == ultimo_ano]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = df_ultimo["indice_atendimento_total_agua"].mean()
        st.metric("Atendimento Água (média)", f"{val:.1f}%" if pd.notna(val) else "N/D")
    with col2:
        val = df_ultimo["indice_coleta_esgoto"].mean()
        st.metric("Coleta Esgoto (média)", f"{val:.1f}%" if pd.notna(val) else "N/D")
    with col3:
        val = df_ultimo["indice_tratamento_esgoto"].mean()
        st.metric("Tratamento Esgoto (média)", f"{val:.1f}%" if pd.notna(val) else "N/D")
    with col4:
        val = df_ultimo["populacao_atendida_agua"].sum()
        st.metric("Pop. Atendida Água", f"{val:,.0f}" if pd.notna(val) else "N/D")

    # --- Abas de visualização ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Série Temporal",
        "📊 Comparativo",
        "🔀 Gap Água vs Esgoto",
        "🏗️ Infraestrutura",
        "📋 Dados",
    ])

    # --- Tab 1: Série Temporal ---
    with tab1:
        st.subheader("Evolução temporal de indicadores")

        indicador_opts = INDICADORES_PERCENTUAIS + INDICADORES_VOLUME + INDICADORES_POP
        indicador_sel = st.selectbox(
            "Indicador",
            indicador_opts,
            format_func=lambda x: COLUMN_NAMES.get(x, x),
        )

        fig = px.line(
            df_filtered,
            x="ano",
            y=indicador_sel,
            color="nome_municipio",
            markers=True,
            labels={
                "ano": "Ano",
                indicador_sel: COLUMN_NAMES.get(indicador_sel, indicador_sel),
                "nome_municipio": "Município",
            },
        )
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: Comparativo (ranking) ---
    with tab2:
        st.subheader(f"Ranking de municípios — {ultimo_ano}")

        indicador_rank = st.selectbox(
            "Indicador para ranking",
            INDICADORES_PERCENTUAIS + INDICADORES_INFRA,
            format_func=lambda x: COLUMN_NAMES.get(x, x),
            key="rank_indicator",
        )

        df_rank = (
            df[df["ano"] == ultimo_ano]
            .dropna(subset=[indicador_rank])
            .sort_values(indicador_rank, ascending=False)
            .head(20)
        )

        fig = px.bar(
            df_rank,
            x=indicador_rank,
            y="nome_municipio",
            orientation="h",
            labels={
                indicador_rank: COLUMN_NAMES.get(indicador_rank, indicador_rank),
                "nome_municipio": "",
            },
            color=indicador_rank,
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: Gap Água vs Esgoto ---
    with tab3:
        st.subheader(f"Diferença entre cobertura de água e esgoto — {ultimo_ano}")

        df_gap = df[df["ano"] == ultimo_ano].dropna(
            subset=["indice_atendimento_total_agua", "indice_coleta_esgoto"]
        ).copy()
        df_gap["gap"] = df_gap["indice_atendimento_total_agua"] - df_gap["indice_coleta_esgoto"]
        df_gap = df_gap.sort_values("gap", ascending=False).head(25)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_gap["nome_municipio"],
            x=df_gap["indice_atendimento_total_agua"],
            name="Água (%)",
            orientation="h",
            marker_color="#2196F3",
        ))
        fig.add_trace(go.Bar(
            y=df_gap["nome_municipio"],
            x=df_gap["indice_coleta_esgoto"],
            name="Esgoto (%)",
            orientation="h",
            marker_color="#FF9800",
        ))
        fig.update_layout(
            barmode="overlay",
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            xaxis_title="Cobertura (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Municípios com maior diferença entre cobertura de água e coleta de esgoto.")

    # --- Tab 4: Infraestrutura ---
    with tab4:
        st.subheader("Evolução da infraestrutura")

        infra_sel = st.selectbox(
            "Métrica",
            INDICADORES_INFRA,
            format_func=lambda x: COLUMN_NAMES.get(x, x),
            key="infra_metric",
        )

        fig = px.area(
            df_filtered.dropna(subset=[infra_sel]),
            x="ano",
            y=infra_sel,
            color="nome_municipio",
            labels={
                "ano": "Ano",
                infra_sel: COLUMN_NAMES.get(infra_sel, infra_sel),
                "nome_municipio": "Município",
            },
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Dispersão: Rede vs Pop. Atendida — {ultimo_ano}")
        df_scatter = df[df["ano"] == ultimo_ano].dropna(
            subset=["extensao_rede_agua", "populacao_atendida_agua"]
        )
        fig = px.scatter(
            df_scatter,
            x="extensao_rede_agua",
            y="populacao_atendida_agua",
            size="populacao_total_residente",
            hover_name="nome_municipio",
            labels={
                "extensao_rede_agua": "Extensão Rede Água (km)",
                "populacao_atendida_agua": "Pop. Atendida Água",
                "populacao_total_residente": "Pop. Total",
            },
            color="indice_atendimento_total_agua",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 5: Dados brutos ---
    with tab5:
        st.subheader("Dados filtrados")
        display_cols = ["ano", "nome_municipio"] + [
            c for c in df_filtered.columns
            if c not in ("ano", "id_municipio", "nome_municipio", "sigla_uf")
            and df_filtered[c].notna().any()
        ]
        st.dataframe(
            df_filtered[display_cols].rename(columns=COLUMN_NAMES),
            use_container_width=True,
            height=500,
        )

        csv = df_filtered.to_csv(index=False)
        st.download_button(
            "📥 Baixar CSV filtrado",
            csv,
            "saneamento_rj_filtrado.csv",
            "text/csv",
        )


if __name__ == "__main__":
    main()
