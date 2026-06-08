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

## Estrutura mínima do repositório (P2)
```
nome-do-projeto/
│
├── app.py                    # Aplicação Streamlit (a implementar)
├── requirements.txt          # Dependências com versões travadas
├── README.md                 # README original (não alterado)
├── README_P2.md              # Este arquivo (P2 - formato solicitado)
│
├── notebooks/
│   └── projeto_14_breast_cancer_v2.ipynb
│
├── model/
│   └── model_final.joblib    # Modelo salvo com joblib
│   └── scaler.joblib         # Scaler salvo (se aplicável)
│
├── reports/
│   └── relatorio_atualizado.pdf
│
└── data/
    └── dataset.csv
```

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

## Instruções para a aplicação Streamlit (quando for implementada)
- Carregar `model/model_final.joblib` e `model/scaler.joblib` conforme exemplo acima.
- Criar inputs compatíveis com as 15 features utilizadas pelo modelo final.
- Incluir botão de predição e interpretação simples do resultado (Benigno / Maligno), com indicação de probabilidade e aviso sobre limitações clínicas.

## Checklist para entrega (P2)
- Consulte a seção "Checklist de Submissão (conforme orientações do PDF)" mais abaixo para o checklist consolidado e itens de verificação finais.

## Limitações e observações
- Este repositório contém o notebook finalizado e o código para salvar o modelo; a aplicação Streamlit e o relatório em PDF devem ser gerados e adicionados antes da submissão final no Moodle.
- Em contexto clínico, resultados devem ser interpretados por especialistas e o modelo não substitui exame clínico ou patologia.

## Contato
- Dúvidas ou ajustes: adicionar seção com e-mail dos integrantes ou usar issues no repositório GitHub (quando criado).

---
*Arquivo gerado automaticamente para atender ao padrão de entrega da P2.*

## Checklist de Submissão (conforme orientações do PDF)

Antes de realizar a entrega no Moodle, verifique os itens abaixo e confirme que estão presentes e funcionais no repositório:

- [ ] Link do notebook atualizado (`notebooks/projeto_14_breast_cancer_v2.ipynb`) — o notebook deve conter todas as melhorias realizadas após a P1.
- [ ] Relatório atualizado em PDF (`reports/relatorio_atualizado.pdf`) — coerente com os resultados e com o notebook.
- [ ] Link do repositório GitHub público com todos os arquivos necessários.
- [ ] Modelo final salvo em `model/` (`model_final.joblib` ou .pkl) e carregável pelo app.
- [ ] `requirements.txt` atualizado com versões travadas.
- [ ] App Streamlit implementado (`app.py`) e testado localmente.
- [ ] Deploy do app publicado (Streamlit Community Cloud) e link disponível no `README.md` principal.
- [ ] Vídeos individuais (cada integrante) com câmera aberta, áudio claro, compartilhamento de tela e link do Google Drive com permissão de visualização.
- [ ] Todos os links testados e acessíveis (GitHub, app, Google Drive).

## Checklist técnico pré-envio (verificações rápidas)

Execute estas verificações finais em um ambiente limpo antes de submeter:

1. `pip install -r requirements.txt` roda sem erros.
2. `notebooks/projeto_14_breast_cancer_v2.ipynb` executa em sequência (Run All) sem erros no seu ambiente local (ajuste caminhos de `data/` se necessário).
3. `model/model_final.joblib` carrega com `joblib.load()` e permite `model.predict()` no conjunto de teste.
4. `app.py` (quando implementado) importa `joblib`, carrega o `scaler.joblib` (se existir) e produz predições consistentes com o notebook.
5. Relatório PDF (`reports/relatorio_atualizado.pdf`) descreve as mudanças feitas após a P1, justificativas para decisões, e inclui principais gráficos e métricas.

## Roteiro de entrega sugerido (passo a passo)

1. Atualizar `notebooks/projeto_14_breast_cancer_v2.ipynb` e executar tudo localmente.
2. Gerar `reports/relatorio_atualizado.pdf` a partir do notebook (ou compilar relatório manualmente), colocando em `reports/`.
3. Salvar o modelo final com `joblib` em `model/model_final.joblib` e salvar `model/scaler.joblib` se aplicável.
4. Implementar `app.py` em Streamlit que carregue o modelo salvo e aceite entradas do usuário.
5. Testar o app localmente e, se OK, publicar no Streamlit Community Cloud.
6. Garantir que `README.md` (principal) contenha o link do app, instruções e o checklist de submissão.
7. Reunir vídeos individuais e disponibilizar links no Google Drive com permissão de visualização.
8. Fazer a submissão única no Moodle (apenas um integrante) incluindo links e arquivos requisitados.

## Observações finais (compatibilidade com o PDF)

Este `README_P2.md` foi atualizado para refletir os requisitos detalhados no documento de P2 e inclui o checklist que deve ser seguido antes da submissão no Moodle. Se quiser, posso:

- Gerar um template mínimo de `app.py` em Streamlit que já carregue `model/model_final.joblib` e aceite as 15 features selecionadas.
- Gerar o esboço do `reports/relatorio_atualizado.pdf` a partir dos resultados do notebook.

Informe qual desses itens você prefere que eu implemente a seguir.
