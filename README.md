# Projeto 14 — Classificação de Câncer de Mama

## Visão Geral
Desafio do curso A.I — Grupo 14. Objetivo: construir um classificador binário capaz de distinguir tumores benignos de malignos usando o dataset Breast Cancer Wisconsin (Kaggle: uciml/breast-cancer-wisconsin-data).

## Dados
- Amostras: 569
- Atributos preditores: 30
- Distribuição de classes: ~62,7% benignos / ~37,3% malignos

## Introdução e EDA
O objetivo principal foi identificar características morfológicas das células que permitissem discriminar tumores benignos e malignos. A análise exploratória mostrou que atributos relacionados ao tamanho e irregularidade do núcleo (médias e máximos) tendem a ser maiores em tumores malignos.

## Pré-processamento e Metodologia
- Limpeza: remoção de colunas não informativas (por exemplo, `ID`) e tratamento de inconsistências.
- Normalização: `StandardScaler` aplicado para padronizar as variáveis.
- Divisão de dados: particionamento estratificado em treino (40%), validação (30%) e teste (30%).
- Seleção de atributos: `SelectKBest` (teste ANOVA) para selecionar as 15 variáveis com maior poder preditivo.

## Modelos e Resultado Principal
O estudo comparou vários classificadores (Regressão Logística, Random Forest, SVM). O melhor desempenho geral foi obtido com Regressão Logística.

- **Regressão Logística** — Acurácia: **94,74%**, F1-Score: **0,9256**, AUC-ROC: **0,9893**. Melhor equilíbrio entre precisão e sensibilidade; menor número de falsos negativos (7 casos no conjunto de teste).
- **Random Forest** — desempenho próximo, com boa capacidade discriminativa.
- **SVM** — apresentou alta precisão, mas gerou mais falsos negativos (12 casos) neste experimento.

## Importância das Features
As medidas do tipo `*_worst` (piores valores observados) foram as mais discriminativas. Entre as features mais indicativas de malignidade:
- `concave points_worst` — aponta deformações no contorno celular.
- `perimeter_worst`, `radius_worst` — indicam crescimento desordenado e tamanho do núcleo.

## Discussão Clínica
Em problemas clínicos a assimetria do custo dos erros é crítica: falsos negativos são muito mais graves que falsos positivos. Por isso, priorizar modelos com maior sensibilidade (recall) é geralmente desejável, mesmo que custe algum aumento de falsos positivos.

## Como Reproduzir
1. Baixar o dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
2. Instalar dependências (ex.: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`).
3. Executar o notebook `projeto_14_breast_cancer_v2.ipynb` que acompanha este repositório.

## Observações
Se quiser, posso:
- Inserir a tabela completa de métricas (acurácia / precisão / recall / F1 / AUC) no README;
- Adicionar gráficos (boxplots, ROC) gerados a partir do notebook;
- Ajustar o texto para inglês ou outro estilo de formatação.

---
Arquivo gerado a partir do relatório do desafio (PDF) e integrado ao README.
