import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


MODEL_PATH = Path('model') / 'model_final.joblib'
SCALER_PATH = Path('model') / 'scaler.joblib'
FEATURES_PATH = Path('model') / 'feature_names.txt'


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    features = []
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
    return model, scaler, features


def predict_df(model, scaler, df):
    X = df.values
    if scaler is not None:
        X = scaler.transform(X)
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    out = df.copy()
    out['proba_malign'] = proba
    out['pred'] = pred
    return out


def main():
    st.set_page_config(page_title='Breast Cancer Classifier', layout='wide')
    st.title('Classificador de Câncer de Mama — Demo')

    st.markdown('Carrega `model/model_final.joblib` e `model/scaler.joblib` (se existir).')

    model, scaler, features = load_model()
    if model is None:
        st.warning('Modelo não encontrado em `model/model_final.joblib`. Rode o notebook para gerar o modelo ou coloque o arquivo no diretório `model/`.')
        return

    st.sidebar.header('Opções')
    mode = st.sidebar.selectbox('Modo', ['Predição única (manual)', 'Upload CSV'])

    if mode == 'Upload CSV':
        uploaded = st.file_uploader('Envie um CSV com as colunas das features', type=['csv'])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write('Amostras carregadas:', len(df))
            # If features list available, try to select the subset
            if features:
                missing = [f for f in features if f not in df.columns]
                if missing:
                    st.warning(f'As colunas do modelo esperadas não estão todas presentes. Colunas faltando: {missing[:5]}{"..." if len(missing)>5 else ""}')
                X_df = df[[c for c in features if c in df.columns]]
            else:
                X_df = df.select_dtypes(include=[np.number])

            if st.button('Executar predição (CSV)'):
                try:
                    out = predict_df(model, scaler, X_df)
                    st.dataframe(out.head(20))
                    st.success('Predição realizada com sucesso.')
                    st.download_button('Baixar resultados (CSV)', out.to_csv(index=False).encode('utf-8'), file_name='predicoes.csv')
                except Exception as e:
                    st.error(f'Erro ao prever: {e}')

    else:
        st.subheader('Predição única (manual)')
        if features:
            st.write('Preencha as features abaixo:')
            cols = st.columns(3)
            vals = []
            for i, feat in enumerate(features):
                with cols[i % 3]:
                    v = st.number_input(feat, value=0.0, format='%.6f')
                    vals.append(v)

            if st.button('Executar predição (manual)'):
                try:
                    sample = pd.DataFrame([vals], columns=features)
                    out = predict_df(model, scaler, sample)
                    st.write(out.T)
                    prob = out['proba_malign'].iloc[0]
                    label = 'Maligno (1)' if out['pred'].iloc[0] == 1 else 'Benigno (0)'
                    st.metric('Predição', label)
                    st.metric('Probabilidade Maligno', f'{prob:.3f}')
                except Exception as e:
                    st.error(f'Erro na predição: {e}')
        else:
            st.info('Nenhuma lista de features encontrada em `model/feature_names.txt`. Você pode enviar um CSV com as colunas numéricas ou criar esse arquivo com a lista de features esperadas.')
            csv_text = st.text_area('Cole valores separados por vírgula (ordem das features esperada)', height=100)
            if st.button('Executar predição (CSV paste)'):
                try:
                    parts = [float(x.strip()) for x in csv_text.split(',') if x.strip()]
                    sample = pd.DataFrame([parts])
                    out = predict_df(model, scaler, sample)
                    st.write(out.T)
                except Exception as e:
                    st.error(f'Erro ao converter os valores: {e}')

    st.markdown('---')
    st.markdown('Observação: Este app é um esqueleto de demonstração. Verifique a ordem e os nomes das features ao usar o modelo salvo.')


if __name__ == '__main__':
    main()
