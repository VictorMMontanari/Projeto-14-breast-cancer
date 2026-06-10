import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS personalizado ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #f7f6f2; }

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 40px 48px; margin-bottom: 32px;
    position: relative; overflow: hidden;
}
.main-header::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.main-title {
    font-family: 'DM Serif Display', serif; font-size: 2.4rem;
    color: #ffffff; margin: 0 0 6px 0; letter-spacing: -0.5px;
}
.main-subtitle {
    font-size: 0.95rem; color: rgba(255,255,255,0.55);
    margin: 0; font-weight: 300; letter-spacing: 0.3px;
}
.badge {
    display: inline-block; background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15); color: rgba(255,255,255,0.75);
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 14px;
    letter-spacing: 1px; text-transform: uppercase;
}
.section-label {
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    letter-spacing: 2px; text-transform: uppercase; color: #888;
    margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #e0ddd5;
}
.result-card {
    border-radius: 14px; padding: 32px 36px; margin-top: 8px;
    text-align: center; position: relative; overflow: hidden;
}
.result-card.benigno { background: linear-gradient(135deg, #e8f5e9, #f1f8f2); border: 1.5px solid #a5d6a7; }
.result-card.maligno { background: linear-gradient(135deg, #fce4ec, #fdf0f4); border: 1.5px solid #f48fb1; }
.result-icon { font-size: 3rem; margin-bottom: 8px; }
.result-label { font-family: 'DM Serif Display', serif; font-size: 2rem; font-weight: 400; margin: 0 0 4px 0; }
.result-label.benigno { color: #2e7d32; }
.result-label.maligno { color: #c62828; }
.result-prob { font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #666; margin: 0 0 16px 0; }
.result-desc { font-size: 0.88rem; color: #555; line-height: 1.6; max-width: 420px; margin: 0 auto; }
.disclaimer { font-size: 0.75rem; color: #999; margin-top: 14px; font-style: italic; }
.info-box {
    background: #fff; border: 1px solid #e8e5de; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 24px; font-size: 0.88rem;
    color: #555; line-height: 1.65;
}
.info-box strong { color: #222; }
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    color: white; border: none; border-radius: 10px; padding: 14px 24px;
    font-family: 'DM Sans', sans-serif; font-size: 1rem; font-weight: 600;
    letter-spacing: 0.3px; cursor: pointer; transition: opacity 0.2s; margin-top: 8px;
}
div.stButton > button:hover { opacity: 0.88; }
div[data-testid="stNumberInput"] input { font-family: 'DM Mono', monospace; font-size: 0.9rem; border-radius: 8px; }
div[data-testid="stNumberInput"] label,
div[data-testid="stNumberInput"] label p {
    color: #222222 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}
.metric-row { display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap; }
.metric-chip {
    background: #fff; border: 1px solid #e0ddd5; border-radius: 8px;
    padding: 10px 16px; font-size: 0.82rem; flex: 1; min-width: 100px; text-align: center;
}
.metric-chip .chip-label {
    font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #999;
    text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 4px;
}
.metric-chip .chip-val { font-weight: 600; color: #1a1a2e; font-size: 0.9rem; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Carregamento do modelo, scaler e features ───────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path = Path("model/model_final.joblib")
    scaler_path = Path("model/scaler.joblib")
    features_path = Path("model/features.joblib")

    errors = []
    if not model_path.exists():
        errors.append("model/model_final.joblib não encontrado")
    if not features_path.exists():
        errors.append("model/features.joblib não encontrado")

    if errors:
        return None, None, None, " | ".join(errors)

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, scaler, features, None


model, scaler, feature_names, load_error = load_artifacts()


def get_model_name(m):
    if m is None:
        return "—"
    return {
        "LogisticRegression":    "Logistic Regression",
        "RandomForestClassifier":"Random Forest",
        "SVC":                   "SVM (Support Vector Machine)",
    }.get(type(m).__name__, type(m).__name__)


# ── Defaults e limites por feature (usados nos inputs) ─────────────────────
FEATURE_META = {
    "radius_mean":             ("Radius Mean",             14.1,  6.9,   29.0,   0.01),
    "texture_mean":            ("Texture Mean",            19.3,  9.7,   40.0,   0.1),
    "perimeter_mean":          ("Perimeter Mean",          92.0,  43.0,  190.0,  0.1),
    "area_mean":               ("Area Mean",               655.0, 140.0, 2510.0, 1.0),
    "smoothness_mean":         ("Smoothness Mean",         0.096, 0.05,  0.17,   0.001),
    "compactness_mean":        ("Compactness Mean",        0.104, 0.02,  0.35,   0.001),
    "concavity_mean":          ("Concavity Mean",          0.089, 0.0,   0.43,   0.001),
    "concave points_mean":     ("Concave Points Mean",     0.049, 0.0,   0.20,   0.001),
    "symmetry_mean":           ("Symmetry Mean",           0.181, 0.10,  0.31,   0.001),
    "fractal_dimension_mean":  ("Fractal Dimension Mean",  0.063, 0.04,  0.10,   0.0001),
    "radius_se":               ("Radius SE",               0.405, 0.11,  2.90,   0.01),
    "texture_se":              ("Texture SE",              1.22,  0.36,  4.90,   0.01),
    "perimeter_se":            ("Perimeter SE",            2.87,  0.76,  22.0,   0.01),
    "area_se":                 ("Area SE",                 40.3,  6.0,   550.0,  0.1),
    "smoothness_se":           ("Smoothness SE",           0.007, 0.001, 0.032,  0.0001),
    "compactness_se":          ("Compactness SE",          0.025, 0.002, 0.14,   0.001),
    "concavity_se":            ("Concavity SE",            0.032, 0.0,   0.40,   0.001),
    "concave points_se":       ("Concave Points SE",       0.012, 0.0,   0.053,  0.0001),
    "symmetry_se":             ("Symmetry SE",             0.021, 0.007, 0.080,  0.0001),
    "fractal_dimension_se":    ("Fractal Dimension SE",    0.004, 0.0,   0.030,  0.0001),
    "radius_worst":            ("Radius Worst",            16.3,  7.0,   40.0,   0.01),
    "texture_worst":           ("Texture Worst",           25.7,  12.0,  50.0,   0.1),
    "perimeter_worst":         ("Perimeter Worst",         107.0, 50.0,  260.0,  0.1),
    "area_worst":              ("Area Worst",              880.0, 185.0, 4260.0, 1.0),
    "smoothness_worst":        ("Smoothness Worst",        0.132, 0.07,  0.22,   0.001),
    "compactness_worst":       ("Compactness Worst",       0.254, 0.03,  1.10,   0.001),
    "concavity_worst":         ("Concavity Worst",         0.272, 0.0,   1.25,   0.001),
    "concave points_worst":    ("Concave Points Worst",    0.115, 0.0,   0.30,   0.001),
    "symmetry_worst":          ("Symmetry Worst",          0.290, 0.15,  0.66,   0.001),
    "fractal_dimension_worst": ("Fractal Dimension Worst", 0.084, 0.05,  0.21,   0.0001),
}

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge">🔬 Machine Learning · Classificação Médica</div>
    <h1 class="main-title">Breast Cancer Classifier</h1>
    <p class="main-subtitle">
        Predição de tumor benigno ou maligno com base em características nucleares de células extraídas por FNAC
    </p>
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────────────────────
col_inputs, col_result = st.columns([3, 2], gap="large")

with col_inputs:
    if load_error:
        st.error(f"⚠️ {load_error}")
        st.info("Execute a célula de salvamento no notebook para gerar os arquivos na pasta `model/`.")
        st.stop()

    n = len(feature_names)
    st.markdown(f'<div class="info-box">Preencha as <strong>{n} características morfológicas</strong> do núcleo celular. Os valores padrão correspondem à média do dataset Wisconsin Breast Cancer.</div>', unsafe_allow_html=True)

    # Separar por sufixo para exibição organizada
    mean_feats  = [f for f in feature_names if f.endswith("_mean")]
    se_feats    = [f for f in feature_names if f.endswith("_se")]
    worst_feats = [f for f in feature_names if f.endswith("_worst")]
    other_feats = [f for f in feature_names if f not in mean_feats + se_feats + worst_feats]

    input_vals = {}

    def render_inputs(feats, cols=2):
        grid = st.columns(cols)
        for i, feat in enumerate(feats):
            meta = FEATURE_META.get(feat)
            if meta:
                label, default, mn, mx, step = meta
            else:
                label, default, mn, mx, step = feat, 0.0, 0.0, 9999.0, 0.01
            with grid[i % cols]:
                input_vals[feat] = st.number_input(
                    label, min_value=mn, max_value=mx,
                    value=default, step=step, key=feat
                )

    if mean_feats:
        st.markdown('<div class="section-label">📐 Características — Média (_mean)</div>', unsafe_allow_html=True)
        render_inputs(mean_feats)
        st.markdown("<br>", unsafe_allow_html=True)

    if se_feats:
        st.markdown('<div class="section-label">📏 Características — Erro Padrão (_se)</div>', unsafe_allow_html=True)
        render_inputs(se_feats)
        st.markdown("<br>", unsafe_allow_html=True)

    if worst_feats:
        st.markdown('<div class="section-label">⚠️ Características — Pior Caso (_worst)</div>', unsafe_allow_html=True)
        render_inputs(worst_feats)
        st.markdown("<br>", unsafe_allow_html=True)

    if other_feats:
        st.markdown('<div class="section-label">📋 Outras Características</div>', unsafe_allow_html=True)
        render_inputs(other_feats)
        st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button("🔍  Executar Predição", use_container_width=True)

# ── Resultado ────────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-label">📊 Resultado da Predição</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <strong>Modelo carregado:</strong><br>
        {get_model_name(model)}<br><br>
        <strong>Features utilizadas:</strong> {len(feature_names) if feature_names else '—'}<br>
        <strong>Pré-processamento:</strong> {"StandardScaler aplicado" if scaler else "Sem normalização"}<br>
        <strong>Dataset:</strong> Wisconsin Breast Cancer (UCI)
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        try:
            # Monta DataFrame com as features NA ORDEM EXATA do treino
            input_df = pd.DataFrame(
                [[input_vals[f] for f in feature_names]],
                columns=feature_names
            )

            X_scaled = scaler.transform(input_df) if scaler else input_df

            pred  = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            prob_b, prob_m = float(proba[0]), float(proba[1])

            st.session_state["last_result"] = {
                "pred": int(pred), "prob_m": prob_m, "prob_b": prob_b
            }
        except Exception as e:
            st.error(f"Erro na predição: {e}")
            st.stop()

    if "last_result" in st.session_state:
        res    = st.session_state["last_result"]
        pred   = res["pred"]
        prob_m = res["prob_m"]
        prob_b = res["prob_b"]

        if pred == 0:
            st.markdown(f"""
            <div class="result-card benigno">
                <div class="result-icon">✅</div>
                <p class="result-label benigno">Benigno</p>
                <p class="result-prob">Probabilidade: {prob_b:.1%}</p>
                <p class="result-desc">
                    O modelo classificou este tumor como <strong>benigno</strong>.
                    Tumores benignos não são cancerosos e geralmente não se espalham para outros tecidos.
                </p>
                <p class="disclaimer">⚕️ Este resultado é apenas uma predição de ML e não substitui avaliação médica profissional.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card maligno">
                <div class="result-icon">⚠️</div>
                <p class="result-label maligno">Maligno</p>
                <p class="result-prob">Probabilidade: {prob_m:.1%}</p>
                <p class="result-desc">
                    O modelo classificou este tumor como <strong>maligno</strong>.
                    Tumores malignos são cancerosos e podem invadir tecidos próximos.
                    Busque imediatamente avaliação oncológica.
                </p>
                <p class="disclaimer">⚕️ Este resultado é apenas uma predição de ML e não substitui avaliação médica profissional.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-chip">
                <span class="chip-label">P(Benigno)</span>
                <span class="chip-val">{prob_b:.1%}</span>
            </div>
            <div class="metric-chip">
                <span class="chip-label">P(Maligno)</span>
                <span class="chip-val">{prob_m:.1%}</span>
            </div>
            <div class="metric-chip">
                <span class="chip-label">Classe Predita</span>
                <span class="chip-val">{"B" if pred == 0 else "M"}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Distribuição de probabilidade**")
        prob_df = pd.DataFrame({
            "Diagnóstico": ["Benigno", "Maligno"],
            "Probabilidade": [prob_b, prob_m]
        })
        st.bar_chart(prob_df.set_index("Diagnóstico"), color=["#3266ad"])

    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #bbb;">
            <div style="font-size: 3rem; margin-bottom: 12px;">🔬</div>
            <p style="font-size: 0.9rem;">Preencha os campos ao lado<br>e clique em <strong>Executar Predição</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ── Rodapé ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.78rem; color: #aaa; font-family: 'DM Mono', monospace;">
    Breast Cancer Classifier · UNIMAR 2026 · Dataset: Wisconsin Breast Cancer (UCI ML Repository)
</div>
""", unsafe_allow_html=True)