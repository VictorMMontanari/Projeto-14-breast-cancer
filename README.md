# Projeto 14 — Classificação de Câncer de Mama (Entrega P2)

## Identificação
- **Nome do projeto:** Classificação de Câncer de Mama — P2

- Heitor Shoji Kimura — 2077610
- Victor Marinelli Montanari — 2046734
- João Pedro Parussolo Santos — 2031928

## Descrição do problema
Classificação binária para distinguir tumores benignos e malignos usando o dataset Breast Cancer Wisconsin (UCI / Kaggle). O objetivo é treinar um classificador com boa sensibilidade (recall) para reduzir falsos negativos em contexto clínico.

## Objetivo do projeto
- Revisar e melhorar o notebook da P1 conforme devolutiva; aplicar validação cruzada estratificada, enriquecer a EDA, salvar o modelo final e preparar o projeto para deploy em Streamlit.

## Dataset
- Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Observações: 569 amostras, 30 atributos preditores. Ajuste o caminho do arquivo na célula de carregamento do notebook se necessário.

## Tipo de problema
- Machine Learning: classificação binária (Benigno = 0, Maligno = 1)

## Metodologia (resumo)
- Limpeza: remoção de colunas não informativas (`id`, `Unnamed: 32`).
- Seleção de features: `SelectKBest` (ANOVA) para selecionar as 15 melhores features.
- Normalização: `StandardScaler` aplicado quando necessário (ex.: Regressão Logística, SVM).
- Validação: divisão treino/validação/teste (40% treino / 30% val / 30% teste) e validação cruzada estratificada (`StratifiedKFold`, 5 folds) para estimativas robustas.

## Modelos treinados
- Regressão Logística (`LogisticRegression`)
- Random Forest (`RandomForestClassifier`)
- SVM (`SVC`, com `probability=True`)

## Modelo final escolhido
- **Modelo escolhido:** Regressão Logística (melhor equilíbrio entre acurácia e sensibilidade; AUC alta). Justificativa: apresenta menor número de falsos negativos e boa interpretabilidade para o contexto clínico.

## Métricas de avaliação (principais)
- Accuracy, Precision, Recall (sensibilidade), F1-score, AUC-ROC.
- Valores principais obtidos (exemplo do experimento): Regressão Logística — Accuracy: 94.74%, F1: 0.9256, AUC: 0.9893.

## Resultados principais
- As features do tipo `*_worst` mostraram maior poder discriminativo (ex.: `concave points_worst`, `perimeter_worst`, `radius_worst`).
- A validação cruzada estratificada (5 folds) está implementada no notebook e apresenta médias ± desvios-padrão para as métricas.

## Estrutura do repositório
```
projeto_14_breast_cancer/
│
├── app.py                    
├── requirimente.txt
├── README.md                
│
├── notebooks/
│   └── projeto_14_breast_cancer_v2.ipynb
│
├── model/
│   └── features.joblib
│   └── model_final.joblib    
│   └── scaler.joblib         
├── reports/
│   └── README.md
│   └── all_features_stats.csv
│   └── metrics_summary.json
│
└── data/
    └── all_features_stats.csv
    └── data.csv
```

## Tecnologias utilizadas
- Python 3
- Pandas e NumPy para manipulação de dados
- Matplotlib e Seaborn para visualizações
- Scikit-learn para modelagem e avaliação
- Joblib para serialização do modelo
- Jupyter Notebook para experimentação
- Streamlit para interface web

## Como reproduzir (notebook)
1. Criar e ativar um ambiente virtual Python recomendado:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
```

2. Instalar dependências:

```bash
pip install -r requirements.txt
```

3. Abrir e executar `notebooks/projeto_14_breast_cancer_v2.ipynb` em Jupyter Notebook / JupyterLab.
4. Ajustar a célula de carregamento de dados se o dataset estiver em outro caminho.

## Uso do modelo salvo (Joblib)
O notebook salva o melhor modelo em `model/model_final.joblib` e, quando aplicável, o `StandardScaler` em `model/scaler.joblib`.

Exemplo de carregamento em Python (app Streamlit ou script):

```python
import joblib
from pathlib import Path

model = joblib.load(Path('model') / 'model_final.joblib')
scaler_path = Path('model') / 'scaler.joblib'
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    # aplicar scaler.transform(X) antes de model.predict()

# predição de exemplo (X deve ser DataFrame ou ndarray com colunas na ordem esperada)
# X_proc = scaler.transform(X) if scaler is not None else X
# y_pred = model.predict(X_proc)
```

## Instruções para executar o app Streamlit
1. Ative o ambiente virtual:

```bash
source .venv/bin/activate
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o aplicativo:

```bash
streamlit run app.py
```

4. Abra no navegador o endereço exibido no terminal (normalmente `http://localhost:8501`).

## Link do app publicado
- Ainda não publicado. Inserir aqui o link do Streamlit Community Cloud após o deploy.

## Limitações e observações
- Este repositório contém o notebook finalizado e o código para salvar o modelo; a aplicação Streamlit e o relatório em PDF devem ser gerados e adicionados antes da submissão final no Moodle.
- Em contexto clínico, resultados devem ser interpretados por especialistas e o modelo não substitui exame clínico ou patologia.

## Conclusão
O projeto atingiu o objetivo de construir e avaliar um classificador para câncer de mama com bom desempenho, destacando a Regressão Logística como modelo final pelo equilíbrio entre desempenho, sensibilidade e interpretabilidade. Como próximos passos, recomenda-se publicar o app em produção, versionar resultados por experimento e ampliar validações para reforçar robustez clínica.
