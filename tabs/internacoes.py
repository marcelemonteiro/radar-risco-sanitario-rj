import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def carregar_dados_internacoes():
    df = pd.read_excel('data/processed/sih-sus/internacoes_saneamento_inadequado.xlsx')
    return df

def render_internacoes():
    st.markdown('<div class="section-title">🏥 Internações por Saneamento Inadequado — Rio de Janeiro</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Série histórica · Dados SIH-SUS</div>', unsafe_allow_html=True)
    
    try:
        df = carregar_dados_internacoes()
    except FileNotFoundError:
        st.error("Arquivo de dados não encontrado. Verifique se o caminho 'data/processed/sih-sus/internacoes_saneamento_inadequado.xlsx' está correto.")
        return
        
    # --- Filtros (dentro da aba) ---
    st.markdown("#### Filtros")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    anos_disponiveis = sorted(df['ano'].unique().tolist(), reverse=True)
    with col_f1:
        anos_selecionados = st.multiselect(
            "Ano(s)",
            options=anos_disponiveis,
            default=anos_disponiveis,
            key="int_anos"
        )
    
    categorias_disponiveis = sorted(df['categoria'].unique().tolist())
    with col_f2:
        categorias_selecionadas = st.multiselect(
            "Categoria(s) de Doença",
            options=categorias_disponiveis,
            default=categorias_disponiveis,
            key="int_cat"
        )
    
    municipios_disponiveis = sorted(df['municipio_estabelecimento_aih'].unique().tolist())
    with col_f3:
        todos_municipios = st.checkbox("Todos os Municípios", value=True, key="int_todos_mun")
        if todos_municipios:
            municipios_selecionados = municipios_disponiveis
        else:
            municipios_selecionados = st.multiselect(
                "Selecione o(s) Município(s)",
                options=municipios_disponiveis,
                default=[],
                key="int_mun"
            )
    
    # --- Aplicar Filtros ---
    if not anos_selecionados or not categorias_selecionadas or not municipios_selecionados:
        st.warning("Por favor, selecione ao menos um ano, uma categoria e um município.")
        return

    df_filtrado = df[
        (df['ano'].isin(anos_selecionados)) &
        (df['categoria'].isin(categorias_selecionadas)) &
        (df['municipio_estabelecimento_aih'].isin(municipios_selecionados))
    ]
    
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
        return

    # --- KPIs ---
    st.markdown("---")
    total_internacoes = int(df_filtrado['total_internacoes'].sum())
    total_municipios = int(df_filtrado['municipio_estabelecimento_aih'].nunique())
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style="text-align:center;padding:.7rem;border-radius:10px;background:#f8fafc;border:1px solid #e5e7eb">
            <div style="font-size:2rem;font-weight:900;color:#1a1a1a">{total_internacoes:,}</div>
            <div style="font-size:.72rem;color:#64748b;text-transform:uppercase">Total de Internações</div>
        </div>
        """.replace(',', '.'), unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="text-align:center;padding:.7rem;border-radius:10px;background:#f8fafc;border:1px solid #e5e7eb">
            <div style="font-size:2rem;font-weight:900;color:#1a1a1a">{total_municipios}</div>
            <div style="font-size:.72rem;color:#64748b;text-transform:uppercase">Municípios Afetados</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        media_anual = "N/D"
        if len(anos_selecionados) > 0:
            media_anual = f"{total_internacoes / len(anos_selecionados):,.0f}".replace(',', '.')
        st.markdown(f"""
        <div style="text-align:center;padding:.7rem;border-radius:10px;background:#f8fafc;border:1px solid #e5e7eb">
            <div style="font-size:2rem;font-weight:900;color:#1a1a1a">{media_anual}</div>
            <div style="font-size:.72rem;color:#64748b;text-transform:uppercase">Média Anual</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Abas de visualização internas ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral",
        "📈 Série Temporal",
        "📍 Comparativo",
        "📋 Dados",
    ])
    
    # --- Tab 1: Visão Geral ---
    with tab1:
        st.markdown("##### Causas de Internação")
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            df_cat = df_filtrado.groupby('categoria')['total_internacoes'].sum().reset_index()
            df_cat = df_cat.sort_values(by='total_internacoes', ascending=True)
            fig_cat = px.bar(df_cat, x='total_internacoes', y='categoria', orientation='h',
                             labels={'total_internacoes': 'Total', 'categoria': 'Categoria'},
                             color='total_internacoes', color_continuous_scale='Blues',
                             title="Por Categoria de Doença")
            fig_cat.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0), height=350)
            st.plotly_chart(fig_cat, use_container_width=True, key="fig_cat_int")
            
        with col_graf2:
            df_doenca = df_filtrado.groupby('doenca')['total_internacoes'].sum().reset_index()
            df_doenca = df_doenca.sort_values(by='total_internacoes', ascending=True).tail(10)
            fig_doenca = px.bar(df_doenca, x='total_internacoes', y='doenca', orientation='h',
                                labels={'total_internacoes': 'Total', 'doenca': 'Doença'},
                                color='total_internacoes', color_continuous_scale='Greens',
                                title="Top 10 Doenças Específicas")
            fig_doenca.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0), height=350)
            st.plotly_chart(fig_doenca, use_container_width=True, key="fig_doenca_int")
            
    # --- Tab 2: Série Temporal ---
    with tab2:
        st.markdown("##### Evolução Anual das Internações")
        df_ano = df_filtrado.groupby('ano')['total_internacoes'].sum().reset_index()
        fig_ano = px.line(df_ano, x='ano', y='total_internacoes', markers=True,
                          labels={'total_internacoes': 'Total de Internações', 'ano': 'Ano'})
        fig_ano.update_xaxes(type='category') 
        fig_ano.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig_ano, use_container_width=True, key="fig_ano_int")
        
    # --- Tab 3: Comparativo Municipal ---
    with tab3:
        st.markdown("##### Ranking de Municípios — Total de Internações")
        df_mun = df_filtrado.groupby('municipio_estabelecimento_aih')['total_internacoes'].sum().reset_index()
        df_mun = df_mun.sort_values(by='total_internacoes', ascending=False).head(20)
        
        df_mun = df_mun.sort_values(by='total_internacoes', ascending=True) 
        fig_mun = px.bar(df_mun, x='total_internacoes', y='municipio_estabelecimento_aih', orientation='h',
                         labels={'total_internacoes': 'Total de Internações', 'municipio_estabelecimento_aih': ''},
                         color='total_internacoes', color_continuous_scale='Reds')
        fig_mun.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False, height=500)
        st.plotly_chart(fig_mun, use_container_width=True, key="fig_mun_int")
        
    # --- Tab 4: Dados brutos ---
    with tab4:
        st.markdown("##### Dados Filtrados")
        st.dataframe(df_filtrado, use_container_width=True, hide_index=True, height=400)
        
        csv = df_filtrado.to_csv(index=False)
        st.download_button(
            "📥 Baixar CSV filtrado",
            csv,
            "internacoes_filtradas.csv",
            "text/csv",
            key="download_int"
        )
