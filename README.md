# Radar de Risco Sanitário - RJ

Projeto Final do Datathon: Radar de Risco Sanitário nos municípios do Rio de Janeiro.

## Objetivo do projeto

Construir um radar de risco sanitário para os 92 municípios do RJ, integrando dados de saneamento (SNIS, SINISA) e contexto social (IBGE), para classificar municípios conforme a terminologia do PLANSAB:

- Atendimento adequado
- Atendimento precário
- Déficit de esgotamento sanitário
- Déficit de abastecimento de água
- Déficit estrutural de saneamento

## Configuração do ambiente

### 1. Criar ambiente virtual (venv)

```bash
# Criar o venv
python -m venv venv

# Ativar (Windows - PowerShell)
.\venv\Scripts\Activate.ps1

# Ativar (Windows - Git Bash / terminal)
source venv/Scripts/activate

# Ativar (Linux/Mac)
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o dashboard Streamlit

```bash
streamlit run app.py
```

O dashboard abrirá automaticamente no navegador em `http://localhost:8501`.

### 4. Executar os notebooks

```bash
jupyter notebook
```

Navegar até `notebooks/` e abrir o notebook desejado.

## Estrutura de pastas

```text
radar-risco-sanitario-rj/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                  # Dashboard Streamlit
├── data/
│   ├── raw/                # Dados originais sem alteração
│   │   ├── sinisa/
│   │   │   ├── agua/
│   │   │   ├── esgoto/
│   │   │   ├── residuos/
│   │   │   └── gestao_municipal/
│   │   ├── ibge_rj/
│   │   ├── ibge_pib/
│   │   └── municipios_rj.geojson
│   └── processed/          # Dados consolidados para análise/dashboard
│       ├── snis_sinisa_merge_ibge_populacao.csv
│       └── clusterizacao_municipios_rj.csv
├── notebooks/
│   ├── adicionar_colunas_faltantes.ipynb
│   └── clusterizacao_municipios.ipynb
├── scripts/
│   └── baixar_pib_ibge.py
└── regras/
    └── regras.md
```

## Fontes de dados

| Fonte | Descrição | Período |
|-------|-----------|---------|
| SNIS | Água, esgoto, perdas, investimento | 2014-2024 |
| SINISA | Resíduos sólidos, gestão municipal | 2023-2024 |
| IBGE Censo 2022 | População total, urbana, rural, domicílios, área | 2022 |
| IBGE SIDRA | PIB municipal | 2014-2021 |

## Classificação dos municípios (PLANSAB)

A classificação segue a terminologia do PLANSAB (Lei 11.445/2007, atualizada pela Lei 14.026/2020):

| Classificação | Critério |
|---------------|----------|
| Atendimento adequado | Água >= 90%, Esgoto >= 75%, RSU >= 80% |
| Déficit de esgotamento sanitário | Água >= 70%, Esgoto < 50% |
| Déficit severo de esgotamento sanitário | Água >= 70%, Esgoto < 30% |
| Déficit de abastecimento de água | Água < 60% |
| Déficit crítico - água e esgoto | Água < 60%, Esgoto < 30% |
| Déficit estrutural de saneamento | 2+ componentes abaixo do limiar |
| Atendimento precário | Acesso existe mas em condições insatisfatórias |

## Clusterização (K-Means)

O agrupamento de municípios utiliza 6 variáveis normalizadas (MinMaxScaler):

1. Atendimento total de água (%)
2. Coleta de esgoto (%)
3. Cobertura de resíduos sólidos (%)
4. Disposição final inadequada de RSU (%)
5. Percentual de população urbana (%)
6. Perda na distribuição de água (%)

Método de seleção de k: Cotovelo + Silhueta (k entre 2 e 6).

## Padrões de formato

### `data/raw/`

- Manter exatamente o arquivo original baixado da fonte.
- Não editar manualmente arquivos dessa pasta.

### `data/processed/`

- Base única e limpa para análise/modelagem/dashboard.
- Formato: CSV UTF-8 com header simples.

## Referências

- PLANSAB - Plano Nacional de Saneamento Básico
- Marco Legal do Saneamento (Lei 14.026/2020)
- SNIS - Sistema Nacional de Informações sobre Saneamento
- SINISA - Sistema Nacional de Informações em Saneamento (Ministério das Cidades)
- IBGE - Instituto Brasileiro de Geografia e Estatística
