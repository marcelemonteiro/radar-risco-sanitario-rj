# radar-risco-sanitario-rj

Projeto Final do Datathon: Radar de Risco Sanitário nos municípios do Rio de Janeiro.

## Objetivo do projeto

Construir um radar de risco sanitário para os 92 municípios do RJ, integrando dados de saneamento e contexto social , para classificar e priorizar municípios em faixas de risco:

- Critico
- Atencao
- Adequado

## Estrutura de pastas

Estrutura padrão do repositório:

```text
radar-risco-sanitario-rj/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/           # dados originais, sem alteracao
│   │   ├── sinisa/
│   │   │   ├── agua/
│   │   │   ├── esgoto/
│   │   │   ├── gestao_municipal/
│   │   │   └── snis/
│   │   ├── ibge/
intermediarios/convertidos
│   └── processed/     # dados finais prontos para modelo/dashboard
├── src/ # scripts reutilizáveis
└── app/ # dashboard 
```

## Padrões de formato

### `data/raw/`

- Manter exatamente o arquivo original baixado da fonte.
- Nao editar manualmente arquivos dessa pasta.

### `data/processed/`

- Base unica e limpa para analise/modelagem/dashboard.
- Preferir `.parquet` (desempenho) e/ou `.csv` UTF-8 (portabilidade).

## Padrão de nomes de arquivos

Regra geral:

- usar minusculas
- sem acentos
- sem espacos
- separar com `_`
- incluir o ano no final

Formato recomendado:

`<fonte>_<tema>_<nivel>_<tipo>_<ano>.<ext>`

Exemplos:

- `sinisa_esgoto_municipal_indicadores_2024.xlsx`
- `sinisa_gestao_municipal_informacoes_2024.xlsx`
- `snis_agua_esgoto_municipal_1995_2022.csv`

## Inventario de dados já adicionados

No momento, ja existem dados na pasta `data/raw/sinisa`, incluindo:

- SINISA - gestao municipal (informacoes 2024)
- SINISA - agua (base municipal: indicadores e informacoes)
- arquivos de esgoto (base SINISA, quando aplicavel)
- SNIS em CSV (`br_mdr_snis_municipio_agua_esgoto.csv`)

Obs.: Alguns nomes ainda estão no formato original de download. 

## Entregavel minimo esperado em dados processados

Gerar um arquivo final unificado com os 92 municipios do RJ:

- `data/processed/rj_municipios_risco_sanitario_2024.parquet` (ou `.csv`)

Colunas minimas sugeridas:

- `codigo_ibge`
- `municipio`
- `ano_referencia`
- indicadores de agua/esgoto/perdas
- variaveis socioeconomicas
- `score_risco_sanitario`
- `faixa_risco` (`critico`, `atencao`, `adequado`)

## Boas praticas de versionamento

- Evitar versionar arquivos brutos muito grandes quando nao necessario.
- Se `data/raw` ficar pesado, manter no `.gitignore` e documentar no README como baixar.

## Referencias de fonte

- SINISA (Ministerio das Cidades): dados de saneamento
- IBGE (Censo/IBGE Cidades/SIDRA): dados demograficos e sociais
- SNIS: Sistema Nacional de Informações sobre Saneamento