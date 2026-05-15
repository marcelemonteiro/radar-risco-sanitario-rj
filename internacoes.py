import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Internações Saneamento RJ",
    page_icon="🏥",
    layout="wide",
)

@st.cache_data
def carregar_dados():
    df = pd.read_excel('data/processed/sih-sus/internacoes_saneamento_inadequado.xlsx')
    return df

def main():
    st.title("🏥 Internações por Saneamento Inadequado — Rio de Janeiro")
    st.markdown("Série histórica · Dados SIH-SUS")
    
    try:
        df = carregar_dados()
    except FileNotFoundError:
        st.error("Arquivo de dados não encontrado. Verifique se o caminho 'data/processed/sih-sus/internacoes_saneamento_inadequado.xlsx' está correto.")
        return
        
    # --- Sidebar - Filtros ---
    st.sidebar.header("Filtros")
    
    anos_disponiveis = sorted(df['ano'].unique().tolist(), reverse=True)
    anos_selecionados = st.sidebar.multiselect(
        "Ano(s)",
        options=anos_disponiveis,
        default=anos_disponiveis
    )
    
    categorias_disponiveis = sorted(df['categoria'].unique().tolist())
    categorias_selecionadas = st.sidebar.multiselect(
        "Categoria(s) de Doença",
        options=categorias_disponiveis,
        default=categorias_disponiveis
    )
    
    municipios_disponiveis = sorted(df['municipio_estabelecimento_aih'].unique().tolist())
    todos_municipios = st.sidebar.checkbox("Todos os Municípios", value=True)
    
    if todos_municipios:
        municipios_selecionados = municipios_disponiveis
    else:
        municipios_selecionados = st.sidebar.multiselect(
            "Selecione o(s) Município(s)",
            options=municipios_disponiveis,
            default=[]
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
    
    col1, col2, col3, _ = st.columns(4) # O app.py usa 4 colunas para KPIs, vamos alinhar
    with col1:
        st.metric("Total de Internações", f"{total_internacoes:,}".replace(',', '.'))
    with col2:
        st.metric("Municípios Afetados", total_municipios)
    with col3:
        if len(anos_selecionados) > 0:
            media_anual = total_internacoes / len(anos_selecionados)
            st.metric("Média Anual de Internações", f"{media_anual:,.0f}".replace(',', '.'))
        else:
            st.metric("Média Anual", "N/D")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Abas de visualização ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral",
        "📈 Série Temporal",
        "📍 Comparativo",
        "📋 Dados",
    ])
    
    # --- Tab 1: Visão Geral ---
    with tab1:
        st.subheader("Causas de Internação")
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            df_cat = df_filtrado.groupby('categoria')['total_internacoes'].sum().reset_index()
            df_cat = df_cat.sort_values(by='total_internacoes', ascending=True)
            fig_cat = px.bar(df_cat, x='total_internacoes', y='categoria', orientation='h',
                             labels={'total_internacoes': 'Total', 'categoria': 'Categoria'},
                             color='total_internacoes', color_continuous_scale='Blues',
                             title="Por Categoria de Doença")
            fig_cat.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_cat, use_container_width=True)
            
        with col_graf2:
            df_doenca = df_filtrado.groupby('doenca')['total_internacoes'].sum().reset_index()
            df_doenca = df_doenca.sort_values(by='total_internacoes', ascending=True).tail(10)
            fig_doenca = px.bar(df_doenca, x='total_internacoes', y='doenca', orientation='h',
                                labels={'total_internacoes': 'Total', 'doenca': 'Doença'},
                                color='total_internacoes', color_continuous_scale='Greens',
                                title="Top 10 Doenças Específicas")
            fig_doenca.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_doenca, use_container_width=True)
            
    # --- Tab 2: Série Temporal ---
    with tab2:
        st.subheader("Evolução Anual das Internações")
        df_ano = df_filtrado.groupby('ano')['total_internacoes'].sum().reset_index()
        fig_ano = px.line(df_ano, x='ano', y='total_internacoes', markers=True,
                          labels={'total_internacoes': 'Total de Internações', 'ano': 'Ano'})
        fig_ano.update_xaxes(type='category') 
        fig_ano.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_ano, use_container_width=True)
        
    # --- Tab 3: Comparativo Municipal ---
    with tab3:
        st.subheader("Ranking de Municípios — Total de Internações")
        df_mun = df_filtrado.groupby('municipio_estabelecimento_aih')['total_internacoes'].sum().reset_index()
        # Top 20 para o ranking horizontal
        df_mun = df_mun.sort_values(by='total_internacoes', ascending=False).head(20)
        
        # Ascending=True para o Plotly ordenar de cima para baixo
        df_mun = df_mun.sort_values(by='total_internacoes', ascending=True) 
        fig_mun = px.bar(df_mun, x='total_internacoes', y='municipio_estabelecimento_aih', orientation='h',
                         labels={'total_internacoes': 'Total de Internações', 'municipio_estabelecimento_aih': ''},
                         color='total_internacoes', color_continuous_scale='Reds')
        fig_mun.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig_mun, use_container_width=True)
        
    # --- Tab 4: Dados brutos ---
    with tab4:
        st.subheader("Dados Filtrados")
        st.dataframe(df_filtrado, use_container_width=True, hide_index=True, height=500)
        
        csv = df_filtrado.to_csv(index=False)
        st.download_button(
            "📥 Baixar CSV filtrado",
            csv,
            "internacoes_filtradas.csv",
            "text/csv",
        )

if __name__ == "__main__":
    main()
