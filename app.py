import os
import io
import secrets
import textwrap
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

# ==========================================================
# CONFIGURAÇÕES GERAIS
# (precisam bater com o script de treino: pneumonia_model_v2.py)
# ==========================================================

# Caminhos relativos à pasta onde este arquivo está (funcionam em qualquer computador).
# Basta deixar o arquivo .keras na mesma pasta do app.
PASTA_APP = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "Pneumonia-Model-melhor.keras"
MODEL_PATH = os.path.join(PASTA_APP, MODEL_FILENAME)
IMAGE_SIZE = 128
DISPLAY_SIZE = 384  # tamanho só para EXIBIR imagem/mapa de calor (não afeta o modelo)

# Métricas do modelo (aparecem na aba "Sobre o sistema").
# Preencha com os valores do treino, em fração de 0 a 1 (ex.: 0.93 = 93%).
# Deixe None nas que você não tiver.
METRICAS_MODELO = {
    "Acurácia": None,
    "Sensibilidade": None,
    "Especificidade": None,
    "Precisão": None,
    "F1-score": None,
    "AUC": None,
}


def formatar_metrica(nome, valor):
    if "AUC" in nome.upper():
        return f"{valor:.3f}".replace(".", ",")
    return f"{valor * 100:.1f}%".replace(".", ",")


# Threshold otimizado encontrado na validação durante o treino
THRESHOLD_PADRAO = 0.46
THRESHOLD = THRESHOLD_PADRAO   # sobrescrito pelo controle da barra lateral
USAR_TTA = True                # idem
ALPHA_HEATMAP = 0.45           # idem

HISTORICO_PATH = os.path.join(PASTA_APP, "historico_analises.csv")
IMAGENS_DIR = os.path.join(PASTA_APP, "laudos_imagens")
os.makedirs(IMAGENS_DIR, exist_ok=True)

st.set_page_config(
    page_title="Apoio ao Diagnóstico — Pneumonia",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# ESTILO
# Paleta: petróleo (primária), grafite (texto), cinza-gelo (fundo),
# vermelho/verde clínicos apenas para o resultado.
# Dica: crie .streamlit/config.toml com [theme] base="light" para
# o tema não mudar conforme o navegador (veja instruções na resposta).
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
    --bg: #F2F5F7;
    --surface: #FFFFFF;
    --ink: #14232E;
    --ink-soft: #52646F;
    --line: #DCE3E8;
    --primary: #0E6B78;
    --primary-dark: #0A525C;
    --primary-tint: #E3F1F3;
    --danger: #B3261E;
    --danger-tint: #FBECEA;
    --ok: #1B7A57;
    --ok-tint: #E5F3EC;
    --warn: #8A5A00;
    --warn-tint: #FFF4DC;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--bg) !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: var(--ink);
}
.block-container { padding-top: 4.5rem; max-width: 1280px; }
[data-testid="stHeader"] { background: transparent !important; }

h1, h2, h3 { font-family: 'IBM Plex Serif', Georgia, serif !important; color: var(--ink) !important; letter-spacing: -0.01em; }
h1 { font-size: 1.9rem !important; }
h2, h3 { font-size: 1.25rem !important; }
p, li, label, span, div { color: var(--ink); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--ink-soft) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background-color: #0F2A33 !important; }
[data-testid="stSidebar"] * { color: #E6EEF0 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15) !important; }
.side-title { font-family: 'IBM Plex Serif', serif; font-size: 1.15rem; font-weight: 600; margin-bottom: 2px; }
.side-sub { font-size: .82rem; opacity: .75; margin-bottom: 14px; }
.side-row { display:flex; justify-content:space-between; font-size:.88rem; padding:5px 0; border-bottom:1px solid rgba(255,255,255,.10); }
.side-row b { font-weight:600; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; }
.pill-ok { background:#1B7A57; color:#fff !important; }
.pill-err { background:#B3261E; color:#fff !important; }

/* ---------- Cabeçalho ---------- */
.app-header { display:flex; align-items:center; gap:16px; margin-bottom:6px; }
.app-logo { width:52px; height:52px; border-radius:14px; background:var(--primary); display:flex; align-items:center; justify-content:center; font-size:1.7rem; }
.app-title { font-family:'IBM Plex Serif',serif; font-size:1.7rem; font-weight:600; line-height:1.15; }
.app-sub { color:var(--ink-soft); font-size:.95rem; max-width:70ch; }

.notice {
    background: var(--warn-tint); border:1px solid #F0D9A0; color: var(--warn) !important;
    border-radius:10px; padding:10px 14px; font-size:.86rem; margin:12px 0 4px 0;
}
.notice * { color: var(--warn) !important; }

/* ---------- Abas ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 18px; background: transparent !important; border-radius: 8px 8px 0 0; font-weight:500; }
.stTabs [aria-selected="true"] { background: var(--surface) !important; border: 1px solid var(--line); border-bottom: 2px solid var(--primary); }
.stTabs [aria-selected="true"] * { color: var(--primary) !important; font-weight:600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--primary) !important; }

/* ---------- Containers com borda = cartões ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(20,35,46,.05);
}

/* ---------- Passos ---------- */
.step { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.step-n { width:26px; height:26px; border-radius:50%; background:var(--primary); color:#fff !important; font-size:.8rem; font-weight:600; display:flex; align-items:center; justify-content:center; }
.step-t { font-family:'IBM Plex Serif',serif; font-size:1.1rem; font-weight:600; }

/* ---------- Resultado ---------- */
.res { border-radius:12px; padding:18px 20px; border:1px solid; }
.res-normal { background:var(--ok-tint); border-color:#B7DEC9; }
.res-pneu   { background:var(--danger-tint); border-color:#F0BDB8; }
.res-label { font-size:.82rem; font-weight:600; margin-bottom:2px; }
.res-normal .res-label, .res-normal .res-title { color: var(--ok) !important; }
.res-pneu .res-label, .res-pneu .res-title { color: var(--danger) !important; }
.res-title { font-family:'IBM Plex Serif',serif; font-size:1.45rem; font-weight:600; line-height:1.2; }
.res-conf { margin-top:6px; font-size:.95rem; }

.gauge-wrap { margin-top:16px; }
.gauge { position:relative; height:12px; border-radius:6px; background:linear-gradient(90deg,#1B7A57 0%,#E8E3D0 50%,#B3261E 100%); }
.gauge-dot { position:absolute; top:-4px; width:20px; height:20px; border-radius:50%; background:#fff; border:3px solid var(--ink); transform:translateX(-50%); }
.gauge-thr { position:absolute; top:-6px; width:2px; height:24px; background:var(--ink); opacity:.55; }
.gauge-scale { display:flex; justify-content:space-between; font-size:.78rem; color:var(--ink-soft); margin-top:8px; }

.chip { display:inline-block; background:var(--primary-tint); color:var(--primary-dark) !important; border:1px solid #BEDDE1; border-radius:999px; padding:4px 14px; font-size:.88rem; font-weight:600; margin:2px 6px 2px 0; }
.chip-warn { background:var(--warn-tint); border-color:#F0D9A0; color:var(--warn) !important; }

.legend { display:flex; align-items:center; gap:10px; font-size:.82rem; color:var(--ink-soft); }
.legend-bar { width:140px; height:10px; border-radius:5px; background:linear-gradient(90deg,#1c2a8f,#2aa3d6,#7bd36b,#f3d835,#d92b1f); }

/* ---------- Botões ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius:8px; font-weight:600; border:1px solid var(--primary) !important;
    background: var(--surface) !important; color: var(--primary) !important;
}
.stButton > button *, .stDownloadButton > button * { color: inherit !important; }
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--primary-tint) !important; }
.stButton > button[kind="primary"] { background: var(--primary) !important; color:#fff !important; }
.stButton > button[kind="primary"]:hover { background: var(--primary-dark) !important; }
.stButton > button[kind="primary"] * { color:#fff !important; }

/* ---------- Campos ---------- */
/* wrappers reais dos campos (é neles que o Streamlit pinta fundo e borda) */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div, [data-testid="stDateInput"] [data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1px solid #C5D0D7 !important;
    border-radius: 8px !important;
}
[data-baseweb="base-input"] { border: none !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within, [data-baseweb="select"] > div:focus-within {
    border-color: var(--primary) !important; box-shadow: 0 0 0 2px var(--primary-tint) !important;
}
input, textarea {
    background-color: transparent !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    caret-color: var(--ink);
}
input:disabled { opacity: 1 !important; -webkit-text-fill-color: var(--ink-soft) !important; background-color: #EEF2F4 !important; }
[data-baseweb="input"]:has(input:disabled) { background-color: #EEF2F4 !important; }
input::placeholder, textarea::placeholder { color: #8795A0 !important; -webkit-text-fill-color: #8795A0 !important; opacity: 1 !important; }
[data-baseweb="input"] svg, [data-baseweb="select"] svg { fill: var(--ink-soft) !important; }

/* calendário e listas suspensas */
[data-baseweb="calendar"], [data-baseweb="calendar"] *, [data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: #FFFFFF !important; color: var(--ink) !important;
}
[data-baseweb="calendar"] [aria-selected="true"], [data-baseweb="calendar"] [aria-selected="true"] * {
    background-color: var(--primary) !important; color: #FFFFFF !important;
}
[data-baseweb="tab-highlight"] { background-color: var(--primary) !important; }
[data-baseweb="tab-border"] { background-color: var(--line) !important; }
[data-baseweb="popover"] *, ul[role="listbox"] * { background:#fff; color: var(--ink) !important; }
[data-baseweb="tag"] { background: var(--primary) !important; }
[data-baseweb="tag"] * { color:#fff !important; }
input[type="checkbox"], input[type="radio"] { accent-color: var(--primary); }

/* ---------- Upload ---------- */
[data-testid="stFileUploaderDropzone"] { background: var(--primary-tint) !important; border: 1.5px dashed var(--primary) !important; border-radius: 12px !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
[data-testid="stFileUploaderDropzone"] button { background:#fff !important; }

/* ---------- Métricas ---------- */
[data-testid="stMetric"] { background: var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px 18px; }
[data-testid="stMetricValue"] { font-family:'IBM Plex Serif',serif; color: var(--primary-dark) !important; }

/* ---------- Expander ---------- */
[data-testid="stExpander"] { background:#fff !important; border:1px solid var(--line) !important; border-radius:10px !important; }
[data-testid="stExpander"] summary { background: transparent !important; }

/* ---------- Tabela do histórico ---------- */
.th { font-size:.8rem; font-weight:600; color:var(--ink-soft) !important; padding:6px 0; border-bottom:2px solid var(--line); }
.badge { color:#fff !important; padding:2px 12px; border-radius:999px; font-size:.82rem; font-weight:600; }
.badge-pneu { background: var(--danger); }
.badge-norm { background: var(--ok); }
.disclaimer { font-size:.84rem; color: var(--ink-soft); border-top:1px solid var(--line); padding-top:8px; margin-top:10px; }

code { background: var(--primary-tint) !important; color: var(--primary-dark) !important; border-radius:4px; }
/* ---------- Barra lateral: campos, botões e expander sobre fundo escuro ---------- */
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
    color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important;
}
[data-testid="stSidebar"] input::placeholder { color: #8795A0 !important; -webkit-text-fill-color: #8795A0 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,.06) !important; border: 1px solid rgba(255,255,255,.18) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,.08) !important; border: 1px solid rgba(255,255,255,.35) !important; color: #E6EEF0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,.16) !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #17A2B3 !important; border-color: #17A2B3 !important; color:#fff !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { background: #128796 !important; }
.side-h { font-size:.78rem; font-weight:600; opacity:.7; margin:14px 0 6px 0; }
.mini-bar { display:flex; height:10px; border-radius:5px; overflow:hidden; background:rgba(255,255,255,.12); }
.mini-leg { display:flex; justify-content:space-between; font-size:.78rem; opacity:.85; margin-top:5px; }
.side-last { display:flex; justify-content:space-between; align-items:center; gap:8px; font-size:.84rem; padding:6px 0; border-bottom:1px solid rgba(255,255,255,.10); }
.side-last small { display:block; opacity:.65; font-size:.74rem; }

footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# CARREGAMENTO DO MODELO (cacheado)
# ==========================================================

@st.cache_resource(show_spinner="Carregando modelo de IA...")
def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


model = carregar_modelo()


# ==========================================================
# PRÉ-PROCESSAMENTO (idêntico ao do treino)
# ==========================================================

def preprocess_image(pil_image):
    image = pil_image.convert("L")
    image_np = np.array(image).astype("float32")

    image_tensor = tf.convert_to_tensor(image_np)
    image_tensor = tf.expand_dims(image_tensor, axis=-1)
    image_tensor = tf.image.grayscale_to_rgb(image_tensor)
    image_tensor = tf.image.resize(image_tensor, (IMAGE_SIZE, IMAGE_SIZE))
    return tf.cast(image_tensor, tf.float32)  # 0-255; preprocess_input normaliza depois


def aplicar_view(image_tensor, flip=False, brightness_delta=0.0):
    img = image_tensor
    if flip:
        img = tf.image.flip_left_right(img)
    if brightness_delta != 0.0:
        img = tf.image.adjust_brightness(img, brightness_delta)
    return tf.keras.applications.efficientnet.preprocess_input(img)


TTA_VIEWS = [(False, 0.0), (True, 0.0), (False, 0.05), (False, -0.05)]


def prever_com_tta(pil_image):
    base_tensor = preprocess_image(pil_image)
    probs = []
    views = TTA_VIEWS if USAR_TTA else TTA_VIEWS[:1]
    for flip, delta in views:
        view = aplicar_view(base_tensor, flip=flip, brightness_delta=delta)
        prob = model.predict(tf.expand_dims(view, axis=0), verbose=0)[0][0]
        probs.append(prob)
    return float(np.mean(probs)), base_tensor


# ==========================================================
# GRAD-CAM
# ==========================================================

def avaliar_imagem(pil_image):
    """
    Checagens simples para detectar imagens que provavelmente NÃO são radiografias
    (foto colorida, imagem minúscula, quase uniforme, proporção estranha).
    Retorna a lista de motivos; lista vazia = nada suspeito.
    """
    motivos = []
    w, h = pil_image.size
    if min(w, h) < 64:
        motivos.append(f"resolução muito baixa ({w}×{h} px)")

    amostra = pil_image.copy()
    amostra.thumbnail((512, 512))
    rgb = np.asarray(amostra.convert("RGB"), dtype="float32") / 255.0

    croma = rgb.max(axis=2) - rgb.min(axis=2)
    if float(np.mean(croma > 0.06)) > 0.02:
        motivos.append("contém áreas coloridas (radiografias são em tons de cinza)")
    cinza = rgb.mean(axis=2)
    if float(np.mean(cinza > 0.92)) > 0.35:
        motivos.append("fundo predominantemente branco (parece documento, diagrama ou captura de tela)")
    if float(cinza.std()) < 0.05:
        motivos.append("contraste muito baixo (imagem quase uniforme)")
    if max(w, h) / max(1, min(w, h)) > 1.8:
        motivos.append("proporção incomum para uma radiografia de tórax")
    return motivos


def gerar_grad_cam(keras_model, image_tensor):
    image_batch = tf.expand_dims(
        tf.keras.applications.efficientnet.preprocess_input(image_tensor), axis=0
    )

    backbone = keras_model.get_layer(index=1)
    inputs = keras_model.input
    conv_output = backbone(inputs, training=False)

    x = conv_output
    for layer in keras_model.layers[2:]:
        x = layer(x, training=False)

    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_output, x])

    with tf.GradientTape() as tape:
        conv_output_val, predictions_gc = grad_model(image_batch, training=False)
        loss = predictions_gc[:, 0]

    grads = tape.gradient(loss, conv_output_val)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output_val = conv_output_val[0]
    heatmap = conv_output_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def sobrepor_heatmap(pil_image, heatmap, alpha=0.45):
    # Só para exibição: amplia com interpolação suave (o modelo continua vendo 128x128)
    img_resized = pil_image.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
    img_array = np.array(img_resized).astype("float32") / 255.0

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (DISPLAY_SIZE, DISPLAY_SIZE), method="bicubic"
    ).numpy().squeeze()
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    colormap = matplotlib.colormaps["jet"]  # cm.get_cmap foi removido nas versões novas
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]

    return np.clip(img_array * (1 - alpha) + heatmap_colored * alpha, 0, 1)


def classificar_padrao_heuristico(heatmap):
    """
    IMPORTANTE: NÃO é uma predição do modelo. O modelo é binário (Normal x
    Pneumonia). Esta função é só uma heurística geométrica sobre a distribuição
    espacial do Grad-CAM — não deve ser tratada como diagnóstico de subtipo.
    """
    hot_mask = heatmap > 0.6
    hot_fraction = hot_mask.sum() / heatmap.size

    h, w = heatmap.shape
    quadrantes = [
        hot_mask[0:h // 2, 0:w // 2], hot_mask[0:h // 2, w // 2:],
        hot_mask[h // 2:, 0:w // 2], hot_mask[h // 2:, w // 2:],
    ]
    quadrantes_ativos = sum(1 for q in quadrantes if (q.sum() / q.size) > 0.05)

    if quadrantes_ativos <= 1 and hot_fraction < 0.15:
        padrao = "Localizado (consolidação única)"
        hipotese = "Mais associado a pneumonia bacteriana"
    elif quadrantes_ativos >= 3:
        padrao = "Difuso / bilateral"
        hipotese = "Mais associado a pneumonia viral ou atípica"
    else:
        padrao = "Misto / intermediário"
        hipotese = "Não conclusivo — requer avaliação clínica adicional"

    return padrao, hipotese, hot_fraction, quadrantes_ativos


# ==========================================================
# HISTÓRICO (CSV local)
# ==========================================================

COLUNAS_HISTORICO = [
    "ID do Diagnóstico", "Data/Hora", "Nome do Paciente", "Sexo",
    "Data de Nascimento", "Idade", "Código do Exame", "Arquivo",
    "Resultado do Modelo", "Confiança (%)", "Threshold",
    "Conclusão do Médico", "Observações do Médico",
    "Médico Responsável", "CRM", "TTA",
    "Caminho Imagem Original", "Caminho Heatmap"
]


def gerar_id_diagnostico():
    """ID curto e praticamente único. Ex: D-7K2F9A1C"""
    return "D-" + secrets.token_hex(4).upper()


def carregar_historico():
    if os.path.exists(HISTORICO_PATH):
        df = pd.read_csv(HISTORICO_PATH)
        for col in COLUNAS_HISTORICO:
            if col not in df.columns:
                df[col] = "-"
        return df[COLUNAS_HISTORICO]
    return pd.DataFrame(columns=COLUNAS_HISTORICO)


def salvar_no_historico(linha: dict):
    df = carregar_historico()
    df = pd.concat([df, pd.DataFrame([linha])], ignore_index=True)
    df.to_csv(HISTORICO_PATH, index=False)


def excluir_do_historico(id_diagnostico: str):
    df = carregar_historico()
    df = df[df["ID do Diagnóstico"] != id_diagnostico]
    df.to_csv(HISTORICO_PATH, index=False)


def gerar_pdf_laudo(row) -> bytes:
    """
    Laudo em PDF (A4) com cabeçalho, identificação do paciente, resultado da IA
    (com barra de probabilidade), imagens, avaliação e assinatura do médico.
    Desenhado em milímetros (210 x 297) usando apenas matplotlib.
    """
    from matplotlib.colors import LinearSegmentedColormap

    PETROL, INK, SOFT, LINE, PAPER = "#0E6B78", "#14232E", "#6B7C87", "#DCE3E8", "#F4F8FA"
    eh_pneu = row.get("Resultado do Modelo") == "Pneumonia"
    cor = "#B3261E" if eh_pneu else "#1B7A57"
    cor_fundo = "#FBECEA" if eh_pneu else "#E5F3EC"

    def val(chave, padrao="-"):
        v = row.get(chave, padrao)
        if v is None or pd.isna(v):
            return padrao
        v = str(v).strip()
        return v if v else padrao

    def cortar(texto, n):
        return texto if len(texto) <= n else texto[: n - 1] + "…"

    def num(x, casas=1):
        return f"{float(x):.{casas}f}".replace(".", ",")

    try:
        dt_exame = datetime.strptime(val("Data/Hora"), "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M")
    except ValueError:
        dt_exame = val("Data/Hora")

    idade = val("Idade")
    if idade.isdigit():
        idade += " anos"

    try:
        confianca = float(row["Confiança (%)"])
    except Exception:
        confianca = 0.0
    prob_pneu = confianca / 100 if eh_pneu else 1 - confianca / 100
    try:
        limiar = float(row["Threshold"])
    except Exception:
        limiar = THRESHOLD_PADRAO

    medico = val("Médico Responsável")
    if medico != "-" and not medico.lower().startswith(("dr", "dra")):
        medico = f"Dr(a). {medico}"
    crm = val("CRM")
    crm_txt = crm if crm.upper().startswith("CRM") else f"CRM: {crm}"

    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    def txt(x, y, s, size=9, weight="normal", color=INK, ha="left", va="center", style="normal"):
        ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style)

    def rect(x, y, w, h, fc, ec="none", lw=0.8):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw))

    def secao(y, titulo):
        txt(15, y, titulo, 10.5, "bold", PETROL)
        ax.plot([15, 195], [y + 3, y + 3], color=LINE, lw=1)

    def campo(x, y, rotulo, valor):
        txt(x, y, rotulo, 7.5, color=SOFT)
        txt(x, y + 4.5, valor, 10, "bold")

    # ---------- Cabeçalho ----------
    rect(0, 0, 210, 28, PETROL)
    txt(15, 11, "Laudo de Apoio ao Diagnóstico por Imagem", 15, "bold", "white")
    txt(15, 19, "Radiografia de tórax · Detecção de pneumonia por inteligência artificial", 8.5, color="#CFE6EA")
    txt(195, 10, "ID do diagnóstico", 7.5, color="#CFE6EA", ha="right")
    txt(195, 16, val("ID do Diagnóstico"), 11, "bold", "white", ha="right")
    txt(195, 22, f"Emitido em {datetime.now().strftime('%d/%m/%Y %H:%M')}", 7.5, color="#CFE6EA", ha="right")

    # ---------- 1. Identificação ----------
    secao(37, "Identificação do paciente")
    rect(15, 43, 180, 34, PAPER, LINE)
    campo(20, 47.5, "Paciente", cortar(val("Nome do Paciente"), 42))
    campo(130, 47.5, "Código do exame", cortar(val("Código do Exame"), 20))
    campo(20, 58, "Sexo", val("Sexo"))
    campo(75, 58, "Data de nascimento", val("Data de Nascimento"))
    campo(130, 58, "Idade", idade)
    campo(20, 68.5, "Data e hora do exame", dt_exame)
    campo(75, 68.5, "Arquivo da imagem", cortar(val("Arquivo"), 34))

    # ---------- 2. Resultado da IA ----------
    secao(86, "Resultado da análise por IA")
    rect(15, 93, 180, 34, cor_fundo)
    rect(15, 93, 2.5, 34, cor)
    txt(22, 98, "Resultado do modelo", 7.5, color=SOFT)
    txt(22, 105.5, "Padrão sugestivo de pneumonia" if eh_pneu else "Sem sinais de pneumonia",
        13, "bold", cor)
    txt(140, 98, "Confiança", 7.5, color=SOFT)
    txt(140, 105.5, f"{num(confianca)}%", 13, "bold", cor)
    txt(170, 98, "Limiar", 7.5, color=SOFT)
    txt(170, 105.5, num(limiar, 2), 13, "bold", INK)

    tta = val("TTA")
    extra = f"   ·   TTA: {tta}" if tta != "-" else ""
    txt(22, 113, f"Probabilidade estimada de pneumonia: {num(prob_pneu * 100)}%{extra}", 8, color=SOFT)
    x0, x1 = 22, 188
    cmap = LinearSegmentedColormap.from_list("risco", ["#1B7A57", "#E8E3D0", "#B3261E"])
    ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(x0, x1, 120, 117), cmap=cmap, aspect="auto", zorder=2)
    ax.plot([x0 + (x1 - x0) * limiar] * 2, [115.5, 121.5], color=INK, lw=1, alpha=0.6, zorder=3)
    ax.plot([x0 + (x1 - x0) * prob_pneu], [118.5], marker="o", markersize=7,
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.5, zorder=4)
    txt(x0, 124, "0 · Normal", 7, color=SOFT)
    txt((x0 + x1) / 2, 124, f"Linha = limiar de decisão ({num(limiar, 2)})", 7, color=SOFT, ha="center")
    txt(x1, 124, "1 · Pneumonia", 7, color=SOFT, ha="right")

    # ---------- 3. Imagens ----------
    secao(136, "Imagens analisadas")

    def imagem(caminho, x, y, w, h, legenda):
        rect(x, y, w, h, PAPER, LINE)
        try:
            img = plt.imread(caminho)
            ih, iw = img.shape[:2]
            esc = min((w - 4) / iw, (h - 4) / ih)
            dw, dh = iw * esc, ih * esc
            cx, cy = x + w / 2, y + h / 2
            ax.imshow(img, extent=(cx - dw / 2, cx + dw / 2, cy + dh / 2, cy - dh / 2),
                      aspect="auto", zorder=2, cmap="gray" if img.ndim == 2 else None)
        except Exception:
            txt(x + w / 2, y + h / 2, "Imagem indisponível", 8.5, color=SOFT, ha="center")
        txt(x + w / 2, y + h + 4, legenda, 8.5, color=SOFT, ha="center")

    imagem(val("Caminho Imagem Original"), 15, 143, 86, 60, "Radiografia original")
    imagem(val("Caminho Heatmap"), 109, 143, 86, 60, "Mapa de atenção (Grad-CAM)")

    # ---------- 4. Avaliação do médico ----------
    secao(216, "Avaliação do médico responsável")
    campo(15, 224, "Conclusão em relação ao resultado da IA", cortar(val("Conclusão do Médico"), 52))
    txt(125, 224, "Médico responsável", 7.5, color=SOFT)
    txt(125, 228.5, cortar(medico, 30), 10, "bold")
    txt(125, 233, crm_txt, 8.5, color=SOFT)

    txt(15, 235, "Observações clínicas", 7.5, color=SOFT)
    obs = val("Observações do Médico", "")
    if obs:
        linhas = textwrap.wrap(obs, width=98)
        if len(linhas) > 3:
            linhas = linhas[:3]
            linhas[-1] = cortar(linhas[-1], 96)
        for k, linha in enumerate(linhas):
            txt(15, 240 + k * 4.4, linha, 9)
    else:
        txt(15, 240, "(nenhuma observação registrada)", 9, color=SOFT, style="italic")

    # ---------- Assinatura ----------
    ax.plot([15, 90], [266, 266], color=INK, lw=0.8)
    txt(15, 270, medico, 10.5, "bold")
    txt(15, 274.5, crm_txt, 8.5, color=SOFT)
    ax.plot([125, 195], [266, 266], color=INK, lw=0.8)
    txt(125, 270, "Data do laudo", 8.5, color=SOFT)
    txt(125, 274.5, dt_exame, 9, "bold")

    # ---------- Rodapé ----------
    ax.plot([15, 195], [281, 281], color=LINE, lw=1)
    aviso = ("Este laudo foi gerado com apoio de um sistema de inteligência artificial (uso acadêmico/"
             "experimental) e não substitui a avaliação de um médico responsável. O mapa de atenção "
             "(Grad-CAM) é uma técnica de interpretabilidade, não uma marcação diagnóstica.")
    for k, linha in enumerate(textwrap.wrap(aviso, width=140)):
        txt(15, 284.5 + k * 3.4, linha, 6.5, color=SOFT)

    ax.set_xlim(0, 210)
    ax.set_ylim(297, 0)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf")
    plt.close(fig)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def pdf_laudo_em_cache(dados: dict) -> bytes:
    """Evita regerar todos os PDFs a cada interação na tela."""
    return gerar_pdf_laudo(pd.Series(dados))


# ==========================================================
# COMPONENTES VISUAIS
# ==========================================================

def passo(numero: int, titulo: str):
    st.markdown(
        f"<div class='step'><div class='step-n'>{numero}</div>"
        f"<div class='step-t'>{titulo}</div></div>",
        unsafe_allow_html=True,
    )


def card_resultado(classe: int, confianca: float):
    if classe == 0:
        st.markdown(
            f"<div class='res res-normal'><div class='res-label'>Resultado da IA</div>"
            f"<div class='res-title'>Sem sinais de pneumonia</div>"
            f"<div class='res-conf'>Confiança: <b>{confianca * 100:.1f}%</b></div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='res res-pneu'><div class='res-label'>Resultado da IA</div>"
            f"<div class='res-title'>Padrão sugestivo de pneumonia</div>"
            f"<div class='res-conf'>Confiança: <b>{confianca * 100:.1f}%</b></div></div>",
            unsafe_allow_html=True,
        )


def gauge_probabilidade(prob: float):
    p = max(0.0, min(1.0, prob)) * 100
    t = THRESHOLD * 100
    st.markdown(
        f"<div class='gauge-wrap'>"
        f"<div class='gauge'><div class='gauge-thr' style='left:{t}%'></div>"
        f"<div class='gauge-dot' style='left:{p}%'></div></div>"
        f"<div class='gauge-scale'><span>0 · Normal</span>"
        f"<span>Limiar de decisão: {THRESHOLD:.2f}</span><span>1 · Pneumonia</span></div></div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Probabilidade estimada de pneumonia: {prob * 100:.1f}%")


# ==========================================================
# BARRA LATERAL (parte superior: ações e configurações)
# ==========================================================

OPCAO_CONCORDO = "Concordo com o resultado da IA"


def resetar_analise():
    """Limpa ficha, imagem, resultado e laudo para começar um novo paciente."""
    for k in ("ultimo_resultado", "arquivo_atual_id"):
        st.session_state.pop(k, None)
    st.session_state["p_nome"] = ""
    st.session_state["p_sexo"] = "Não informado"
    st.session_state["p_nasc"] = None
    st.session_state["p_exame"] = ""
    st.session_state["p_conclusao"] = OPCAO_CONCORDO
    st.session_state["p_obs"] = ""
    st.session_state["uploader_ver"] = st.session_state.get("uploader_ver", 0) + 1


def restaurar_limiar():
    st.session_state["limiar_ui"] = THRESHOLD_PADRAO


st.session_state.setdefault("limiar_ui", THRESHOLD_PADRAO)
st.session_state.setdefault("usar_tta_ui", True)
st.session_state.setdefault("alpha_ui", 0.45)

with st.sidebar:
    st.markdown("<div class='side-title'>🫁 Apoio ao Diagnóstico</div>"
                "<div class='side-sub'>Radiografia de tórax · Pneumonia</div>", unsafe_allow_html=True)

    st.button("Nova análise", type="primary", use_container_width=True, on_click=resetar_analise,
              help="Limpa a ficha do paciente, a imagem e o resultado atual.")

    st.markdown("<div class='side-h'>Médico responsável</div>", unsafe_allow_html=True)
    st.text_input("Nome do médico", key="medico_nome", placeholder="Ex: Ana Souza")
    st.text_input("CRM", key="medico_crm", placeholder="Ex: CRM-MG 12345")

    with st.expander("Configurações avançadas"):
        st.slider("Limiar de decisão", 0.10, 0.90, step=0.01, key="limiar_ui")
        st.button("Restaurar limiar padrão", on_click=restaurar_limiar, use_container_width=True)
        st.toggle("Usar TTA (4 visões)", key="usar_tta_ui")
        st.slider("Intensidade do mapa de calor", 0.10, 0.90, step=0.05, key="alpha_ui")
        if abs(st.session_state["limiar_ui"] - THRESHOLD_PADRAO) > 1e-9:
            st.warning(f"Limiar diferente do padrão de treino ({THRESHOLD_PADRAO:.2f}). "
                       "Isso muda o resultado; use apenas para testes.")

    # Valores efetivos usados pelas funções acima
    THRESHOLD = st.session_state["limiar_ui"]
    USAR_TTA = st.session_state["usar_tta_ui"]
    ALPHA_HEATMAP = st.session_state["alpha_ui"]

    status = ("<span class='pill pill-ok'>Carregado</span>" if model is not None
              else "<span class='pill pill-err'>Não encontrado</span>")
    st.markdown("<div class='side-h'>Modelo</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='side-row'><span>Modelo de IA</span>{status}</div>"
        f"<div class='side-row'><span>Arquitetura</span><b>EfficientNetB0</b></div>"
        f"<div class='side-row'><span>Entrada</span><b>{IMAGE_SIZE}×{IMAGE_SIZE} px</b></div>"
        f"<div class='side-row'><span>Limiar</span><b>{THRESHOLD:.2f}</b></div>"
        f"<div class='side-row'><span>Robustez</span><b>{'TTA × ' + str(len(TTA_VIEWS)) if USAR_TTA else 'Sem TTA'}</b></div>",
        unsafe_allow_html=True,
    )


# ==========================================================
# CABEÇALHO
# ==========================================================

st.markdown(
    "<div class='app-header'><div class='app-logo'>🫁</div><div>"
    "<div class='app-title'>Sistema de Apoio ao Diagnóstico por Imagem — Pneumonia</div>"
    "<div class='app-sub'>Triagem por rede neural convolucional (CNN) para auxiliar a análise "
    "de radiografias de tórax. Uso acadêmico/experimental.</div></div></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='notice'>⚠️ Ferramenta de <b>apoio</b> à decisão clínica: não substitui a avaliação "
    "de um médico radiologista ou pneumologista. O resultado é uma estimativa estatística de um modelo "
    "treinado em dados públicos (PneumoniaMNIST) e pode conter erros.</div>",
    unsafe_allow_html=True,
)
st.write("")

aba_analise, aba_historico, aba_tipos, aba_sobre = st.tabs(
    ["🔍 Nova análise", "📁 Histórico", "📚 Tipos de pneumonia", "ℹ️ Sobre o sistema"]
)

# ==========================================================
# ABA 1: NOVA ANÁLISE
# ==========================================================

with aba_analise:

    if model is None:
        st.error(
            f"Modelo não encontrado em:\n`{MODEL_PATH}`\n\n"
            "Verifique se o caminho está correto e se o arquivo .keras existe."
        )
    else:
        col_esq, col_dir = st.columns([1, 1], gap="large")

        # ---------------- Passo 1: ficha do paciente ----------------
        with col_esq:
            with st.container(border=True):
                passo(1, "Ficha do paciente")
                st.caption("Não utilize dados reais de pacientes neste ambiente de teste.")

                nome_paciente = st.text_input("Nome completo *", placeholder="Ex: Maria da Silva", key="p_nome")

                c1, c2 = st.columns(2)
                with c1:
                    sexo_paciente = st.selectbox("Sexo", ["Não informado", "Feminino", "Masculino", "Outro"], key="p_sexo")
                with c2:
                    data_nascimento = st.date_input(
                        "Data de nascimento", value=None,
                        min_value=datetime(1900, 1, 1), max_value=datetime.now(),
                        format="DD/MM/YYYY", key="p_nasc",
                    )

                idade_calculada = "-"
                if data_nascimento is not None:
                    hoje = datetime.now().date()
                    idade_calculada = hoje.year - data_nascimento.year - (
                        (hoje.month, hoje.day) < (data_nascimento.month, data_nascimento.day)
                    )

                c3, c4 = st.columns([1, 2])
                with c3:
                    st.text_input("Idade", value=str(idade_calculada), disabled=True)
                with c4:
                    id_exame_prontuario = st.text_input(
                        "Código do exame / prontuário (opcional)", placeholder="Ex: EXM-2026-0031", key="p_exame"
                    )

        # ---------------- Passo 2: upload ----------------
        with col_dir:
            with st.container(border=True):
                passo(2, "Imagem de raio-X")
                uploaded_file = st.file_uploader(
                    "Envie a radiografia de tórax (PNG, JPG ou JPEG)",
                    type=["png", "jpg", "jpeg"],
                    key=f"uploader_{st.session_state.get('uploader_ver', 0)}",
                )

                if uploaded_file is not None:
                    # Imagem nova => descarta o resultado anterior
                    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                    if st.session_state.get("arquivo_atual_id") != file_id:
                        st.session_state.pop("ultimo_resultado", None)
                        st.session_state["arquivo_atual_id"] = file_id

                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, width=300)

                    alertas = avaliar_imagem(image)
                    liberado = True
                    if alertas:
                        st.warning(
                            "Esta imagem pode não ser uma radiografia de tórax: "
                            + "; ".join(alertas) + ". O modelo só foi treinado com radiografias, "
                            "então o resultado pode não fazer sentido."
                        )
                        liberado = st.checkbox("Analisar mesmo assim", key=f"forcar_{file_id}")

                    analisar = st.button("Analisar imagem", type="primary",
                                         use_container_width=True, disabled=not liberado)
                else:
                    analisar = False
                    st.info("Envie uma imagem para habilitar a análise.")

        if uploaded_file is not None and (analisar or "ultimo_resultado" in st.session_state):

            if analisar:
                with st.spinner("Analisando imagem..."):
                    prob_pneumonia, image_tensor = prever_com_tta(image)
                    heatmap = gerar_grad_cam(model, image_tensor)
                    overlay = sobrepor_heatmap(image, heatmap)

                classe = 1 if prob_pneumonia > THRESHOLD else 0
                confianca = prob_pneumonia if classe == 1 else (1 - prob_pneumonia)

                st.session_state["ultimo_resultado"] = {
                    "classe": classe,
                    "prob_pneumonia": prob_pneumonia,
                    "confianca": confianca,
                    "overlay": overlay,
                    "heatmap": heatmap,
                    "arquivo": uploaded_file.name,
                    "imagem_pil": image,
                    "usou_tta": USAR_TTA,
                }

            resultado = st.session_state["ultimo_resultado"]
            # Reaplica os ajustes atuais da barra lateral (limiar e intensidade do mapa)
            resultado["classe"] = 1 if resultado["prob_pneumonia"] > THRESHOLD else 0
            resultado["confianca"] = (resultado["prob_pneumonia"] if resultado["classe"] == 1
                                      else 1 - resultado["prob_pneumonia"])
            resultado["overlay"] = sobrepor_heatmap(
                resultado["imagem_pil"], resultado["heatmap"], ALPHA_HEATMAP)

            st.write("")
            col_res, col_cam = st.columns([1, 1.25], gap="large")

            # ---------------- Passo 3: resultado ----------------
            with col_res:
                with st.container(border=True):
                    passo(3, "Resultado da análise")
                    card_resultado(resultado["classe"], resultado["confianca"])
                    gauge_probabilidade(resultado["prob_pneumonia"])

            # ---------------- Grad-CAM ----------------
            with col_cam:
                with st.container(border=True):
                    st.markdown("<div class='step-t'>🎯 Mapa de atenção (Grad-CAM)</div>", unsafe_allow_html=True)
                    g1, g2 = st.columns(2)
                    with g1:
                        st.image(image.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS),
                                 caption="Imagem original", use_container_width=True)
                    with g2:
                        st.image(resultado["overlay"],
                                 caption="Regiões que mais pesaram", use_container_width=True)
                    st.markdown(
                        "<div class='legend'><span>Pouca influência</span><div class='legend-bar'></div>"
                        "<span>Muita influência</span></div>",
                        unsafe_allow_html=True,
                    )
                    with st.expander("Como ler o mapa de calor"):
                        st.markdown("""
- 🔴 **Vermelho/amarelo:** regiões que mais pesaram na decisão do modelo.
- 🔵 **Azul/roxo:** regiões que pouco influenciaram o resultado.

Isso **não é uma marcação diagnóstica**: é uma visualização de **interpretabilidade** (Grad-CAM)
que mostra onde a rede concentrou "atenção". Em casos de pneumonia, espera-se que essas regiões
coincidam com áreas pulmonares, especialmente opacidades ou consolidações. A interpretação clínica
final é sempre responsabilidade do médico.
""")

            # ---------------- Classificação heurística ----------------
            if resultado["classe"] == 1:
                padrao, hipotese, hot_fraction, quadrantes_ativos = classificar_padrao_heuristico(
                    resultado["heatmap"]
                )
                with st.container(border=True):
                    st.markdown("<div class='step-t'>🧭 Classificação exploratória do padrão radiológico</div>",
                                unsafe_allow_html=True)
                    st.markdown(
                        f"<span class='chip'>Padrão espacial: {padrao}</span>"
                        f"<span class='chip chip-warn'>Hipótese: {hipotese}</span>",
                        unsafe_allow_html=True,
                    )
                    m1, m2 = st.columns(2)
                    m1.metric("Área de alta ativação", f"{hot_fraction * 100:.1f}%")
                    m2.metric("Quadrantes ativos", f"{quadrantes_ativos} de 4")
                    st.markdown(
                        "<div class='disclaimer'>⚠️ <b>Isto NÃO é uma predição do modelo de IA.</b> "
                        "O modelo foi treinado apenas para detectar a presença de pneumonia (tarefa binária) "
                        "e não aprendeu a distinguir subtipos. A classificação acima é uma <b>heurística "
                        "geométrica</b> sobre a distribuição do mapa de calor (localizado vs. difuso), "
                        "oferecida apenas como ponto de partida para investigação clínica. O subtipo real "
                        "depende de história clínica, exames laboratoriais e avaliação médica presencial. "
                        "Veja a aba 'Tipos de pneumonia' para mais contexto.</div>",
                        unsafe_allow_html=True,
                    )

            # ---------------- Passo 4: laudo do médico ----------------
            with st.container(border=True):
                passo(4, "Avaliação e laudo do médico responsável")

                conclusao_medico = st.radio(
                    "Conclusão do médico em relação ao resultado do modelo:",
                    ["Concordo com o resultado da IA", "Discordo do resultado da IA",
                     "Necessário exame complementar / inconclusivo"],
                    key="p_conclusao",
                )
                observacoes_medico = st.text_area(
                    "Observações clínicas, achados adicionais ou justificativa:",
                    height=120, key="p_obs",
                    placeholder="Ex: paciente com tosse produtiva há 5 dias, febre de 38,5 °C, "
                                "ausculta com estertores em base direita...",
                )

                col_salvar, _ = st.columns([1, 2])
                with col_salvar:
                    if st.button("Salvar laudo no histórico", type="primary", use_container_width=True):
                        if not nome_paciente or not nome_paciente.strip():
                            st.error(
                                "Não é possível salvar sem o **nome completo do paciente**. "
                                "Preencha a ficha do paciente (passo 1) e tente novamente."
                            )
                        elif not st.session_state.get("medico_nome", "").strip():
                            st.error(
                                "Informe o **nome do médico responsável** na barra lateral "
                                "antes de salvar o laudo."
                            )
                        else:
                            id_diagnostico = gerar_id_diagnostico()
                            caminho_original = os.path.join(IMAGENS_DIR, f"{id_diagnostico}_original.png")
                            caminho_heatmap = os.path.join(IMAGENS_DIR, f"{id_diagnostico}_heatmap.png")

                            resultado["imagem_pil"].convert("RGB").save(caminho_original)
                            plt.imsave(caminho_heatmap, resultado["overlay"])

                            data_nasc_str = (
                                data_nascimento.strftime("%d/%m/%Y")
                                if data_nascimento is not None else "-"
                            )

                            salvar_no_historico({
                                "ID do Diagnóstico": id_diagnostico,
                                "Data/Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Nome do Paciente": nome_paciente.strip(),
                                "Sexo": sexo_paciente,
                                "Data de Nascimento": data_nasc_str,
                                "Idade": idade_calculada,
                                "Código do Exame": id_exame_prontuario or "-",
                                "Arquivo": resultado["arquivo"],
                                "Resultado do Modelo": "Pneumonia" if resultado["classe"] == 1 else "Normal",
                                "Confiança (%)": round(resultado["confianca"] * 100, 2),
                                "Threshold": THRESHOLD,
                                "Conclusão do Médico": conclusao_medico,
                                "Observações do Médico": observacoes_medico,
                                "Médico Responsável": st.session_state.get("medico_nome", "").strip(),
                                "CRM": st.session_state.get("medico_crm", "").strip() or "-",
                                "TTA": (f"Ligado ({len(TTA_VIEWS)} visões)"
                                        if resultado.get("usou_tta", True) else "Desligado"),
                                "Caminho Imagem Original": caminho_original,
                                "Caminho Heatmap": caminho_heatmap,
                            })
                            st.success(f"Laudo salvo. ID do diagnóstico: **{id_diagnostico}**")

# ==========================================================
# ABA 2: HISTÓRICO
# ==========================================================

with aba_historico:
    df_historico = carregar_historico()

    if df_historico.empty:
        st.info("Nenhuma análise salva ainda. Abra **Nova análise** para registrar o primeiro laudo.")
    else:
        total = len(df_historico)
        n_pneu = int((df_historico["Resultado do Modelo"] == "Pneumonia").sum())
        n_norm = int((df_historico["Resultado do Modelo"] == "Normal").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total de análises", total)
        m2.metric("Casos 'Pneumonia'", n_pneu)
        m3.metric("Casos 'Normal'", n_norm)
        m4.metric("Proporção de pneumonia", f"{(n_pneu / total) * 100:.0f}%")

        st.write("")
        with st.container(border=True):
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                filtro_resultado = st.multiselect(
                    "Resultado do modelo",
                    options=df_historico["Resultado do Modelo"].unique(),
                    default=list(df_historico["Resultado do Modelo"].unique()),
                )
            with f2:
                busca_paciente = st.text_input("Buscar por nome do paciente")
            with f3:
                busca_id = st.text_input("Buscar por ID do diagnóstico")
            with f4:
                busca_medico = st.text_input("Buscar por médico")

            csv_bytes = df_historico.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar planilha completa (CSV)", data=csv_bytes,
                file_name="historico_analises.csv", mime="text/csv",
            )

        df_filtrado = df_historico[df_historico["Resultado do Modelo"].isin(filtro_resultado)]
        if busca_paciente:
            df_filtrado = df_filtrado[
                df_filtrado["Nome do Paciente"].astype(str).str.contains(busca_paciente, case=False, na=False)
            ]
        if busca_id:
            df_filtrado = df_filtrado[
                df_filtrado["ID do Diagnóstico"].astype(str).str.contains(busca_id, case=False, na=False)
            ]

        if busca_medico:
            df_filtrado = df_filtrado[
                df_filtrado["Médico Responsável"].astype(str).str.contains(busca_medico, case=False, na=False)
            ]

        with st.container(border=True):
            st.markdown("<div class='step-t'>Laudos salvos</div>", unsafe_allow_html=True)
            st.caption("Marque a caixa para excluir, abra 'Detalhes' para ver a ficha completa "
                       "ou baixe o laudo em PDF com as imagens do raio-X e do Grad-CAM.")

            if df_filtrado.empty:
                st.info("Nenhum laudo encontrado com esse filtro ou busca.")
            else:
                larguras = [0.45, 1.1, 1.6, 1.5, 1.4, 1.0, 0.9, 1.0]
                cab = st.columns(larguras)
                for c, t in zip(cab, ["Sel.", "ID", "Paciente", "Médico", "Data/Hora", "Resultado", "Confiança", "Laudo"]):
                    c.markdown(f"<div class='th'>{t}</div>", unsafe_allow_html=True)

                ids_selecionados = []

                for _, row in df_filtrado.iterrows():
                    rid = row["ID do Diagnóstico"]
                    c0, c1, c2, cm, c3, c4, c5, c6 = st.columns(larguras, vertical_alignment="center")

                    marcado = c0.checkbox("Selecionar", key=f"del_{rid}", label_visibility="collapsed")
                    c1.write(rid)
                    c2.write(row["Nome do Paciente"])
                    _med = row["Médico Responsável"]
                    cm.write("-" if pd.isna(_med) else str(_med))
                    c3.write(row["Data/Hora"])

                    eh_pneu = row["Resultado do Modelo"] == "Pneumonia"
                    c4.markdown(
                        f"<span class='badge {'badge-pneu' if eh_pneu else 'badge-norm'}'>"
                        f"{escape(str(row['Resultado do Modelo']))}</span>",
                        unsafe_allow_html=True,
                    )
                    c5.write(f"{row['Confiança (%)']}%")

                    with c6:
                        try:
                            pdf_bytes = pdf_laudo_em_cache(row.to_dict())
                            st.download_button(
                                "PDF", data=pdf_bytes, file_name=f"laudo_{rid}.pdf",
                                mime="application/pdf", key=f"pdf_{rid}", use_container_width=True,
                            )
                        except Exception:
                            st.caption("Sem imagem")

                    with st.expander(f"Detalhes — {rid}"):
                        st.markdown(f"""
- **Sexo:** {row['Sexo']}
- **Data de nascimento:** {row['Data de Nascimento']}
- **Idade:** {row['Idade']}
- **Código do exame:** {row['Código do Exame']}
- **Arquivo da imagem:** {row['Arquivo']}
- **Threshold usado:** {row['Threshold']}
- **Conclusão do médico:** {row['Conclusão do Médico']}
- **Observações do médico:** {row['Observações do Médico']}
- **Médico responsável:** {row['Médico Responsável']} (CRM: {row['CRM']})
- **TTA:** {row['TTA']}
""")

                    if marcado:
                        ids_selecionados.append(rid)

                st.write("")
                confirmar_exclusao = st.checkbox(
                    "Confirmo que desejo excluir o(s) laudo(s) marcado(s) permanentemente."
                )

                if st.button("Excluir laudo(s) selecionado(s)"):
                    if not ids_selecionados:
                        st.warning("Marque ao menos um laudo na lista antes de excluir.")
                    elif not confirmar_exclusao:
                        st.warning("Marque a caixa de confirmação antes de excluir.")
                    else:
                        for id_para_excluir in ids_selecionados:
                            excluir_do_historico(id_para_excluir)
                        st.success(f"{len(ids_selecionados)} laudo(s) excluído(s).")
                        st.rerun()

# ==========================================================
# ABA 3: TIPOS DE PNEUMONIA
# ==========================================================

with aba_tipos:
    st.markdown("""
O modelo de IA deste sistema foi treinado apenas para uma tarefa **binária**: identificar se a
radiografia apresenta **padrão sugestivo de pneumonia ou não**. Ele **não determina o agente causador
nem o subtipo clínico**: isso depende de contexto clínico, exames laboratoriais (hemograma, cultura
de escarro, PCR, sorologia) e avaliação médica. Abaixo, um resumo dos principais tipos, para referência
educacional.
""")

    with st.expander("🦠 Pneumonia bacteriana"):
        st.write(
            "Causada mais comumente pela bactéria *Streptococcus pneumoniae*, mas também por "
            "outras como *Haemophilus influenzae* e *Staphylococcus aureus*. Costuma ter início "
            "abrupto, com febre alta, tosse produtiva e dor torácica. No raio-X, tende a aparecer "
            "como uma consolidação localizada (frequentemente em um único lobo pulmonar)."
        )

    with st.expander("🦠 Pneumonia viral"):
        st.write(
            "Causada por vírus como influenza, vírus sincicial respiratório (VSR), adenovírus ou "
            "coronavírus. Geralmente tem evolução mais gradual, com padrão radiológico mais difuso "
            "e bilateral (infiltrado intersticial), em vez de uma consolidação localizada."
        )

    with st.expander("🦠 Pneumonia atípica (\"pneumonia ambulante\")"):
        st.write(
            "Associada a microrganismos como *Mycoplasma pneumoniae*, *Chlamydophila pneumoniae* "
            "e *Legionella*. Os sintomas costumam ser mais brandos que os da pneumonia bacteriana "
            "típica, e por isso o paciente muitas vezes continua com suas atividades normais "
            "(daí o nome popular). O padrão radiológico costuma ser mais difuso."
        )

    with st.expander("🍄 Pneumonia fúngica"):
        st.write(
            "Menos comum, geralmente associada a pacientes imunocomprometidos (transplantados, "
            "pacientes com HIV/AIDS, em quimioterapia). Fungos como *Aspergillus* e *Pneumocystis "
            "jirovecii* são exemplos de agentes causadores nesse grupo."
        )

    with st.expander("🥤 Pneumonia aspirativa"):
        st.write(
            "Ocorre quando alimentos, líquidos, saliva ou vômito são aspirados para os pulmões, "
            "em vez de seguirem para o esôfago. É mais comum em pessoas com dificuldade de "
            "deglutição, alteração do nível de consciência ou problemas neurológicos."
        )

    with st.expander("🏥 Classificação por local de aquisição"):
        st.write(
            "Também é comum classificar a pneumonia pelo contexto em que foi adquirida: "
            "**Pneumonia Adquirida na Comunidade (PAC)**, contraída fora do ambiente hospitalar, "
            "e **Pneumonia Hospitalar (nosocomial)**, adquirida durante uma internação — esta "
            "última costuma envolver agentes mais resistentes a antibióticos."
        )

# ==========================================================
# ABA 4: SOBRE
# ==========================================================

with aba_sobre:
    with st.container(border=True):
        st.markdown(f"""
Este sistema foi desenvolvido como parte de um Trabalho de Conclusão de Curso (TCC) sobre
**redes neurais convolucionais aplicadas à detecção de pneumonia em radiografias de tórax**.

**Detalhes técnicos do modelo**
- Arquitetura: transfer learning com backbone pré-treinado (EfficientNetB0 / ImageNet)
- Resolução de entrada: {IMAGE_SIZE}×{IMAGE_SIZE} pixels
- Saída: probabilidade (0 a 1) de pneumonia, com limiar de decisão em `{THRESHOLD}`
- Interpretabilidade: Grad-CAM
- Previsão robustecida com Test-Time Augmentation (TTA)

**Limitações importantes**
- O modelo foi treinado e validado no dataset público *PneumoniaMNIST* (derivado do dataset
  de Kermany et al.), que tem características e limitações próprias de qualquer benchmark
  acadêmico; o desempenho em imagens de outras fontes ou equipamentos pode variar.
- O sistema **não substitui** avaliação médica, exame físico, histórico clínico ou exames
  laboratoriais complementares.
- É um protótipo acadêmico/experimental, não aprovado como dispositivo médico.
""")

    with st.container(border=True):
        st.markdown("<div class='step-t'>Desempenho do modelo</div>", unsafe_allow_html=True)
        _mostrar = {k: v for k, v in METRICAS_MODELO.items() if v is not None}
        if _mostrar:
            for _col, (_nome, _v) in zip(st.columns(len(_mostrar)), _mostrar.items()):
                _col.metric(_nome, formatar_metrica(_nome, _v))
            st.caption("Métricas obtidas na avaliação do modelo durante o treino.")
        else:
            st.info("Métricas ainda não informadas. Preencha o dicionário `METRICAS_MODELO` "
                    "no topo do arquivo com os valores do treino.")

st.markdown("---")
st.caption("Sistema desenvolvido para fins acadêmicos (TCC) — não utilizar para diagnóstico clínico real.")


# ==========================================================
# BARRA LATERAL (parte inferior: resumo e últimos laudos)
# Fica no fim do script para refletir laudos salvos/excluídos nesta mesma execução.
# ==========================================================

with st.sidebar:
    _h = carregar_historico()
    _total = len(_h)
    _hoje = datetime.now().strftime("%Y-%m-%d")
    _hoje_n = int(_h["Data/Hora"].astype(str).str.startswith(_hoje).sum()) if _total else 0
    _pneu = int((_h["Resultado do Modelo"] == "Pneumonia").sum()) if _total else 0
    _norm = _total - _pneu
    _pct = (_pneu / _total * 100) if _total else 0

    st.markdown("<div class='side-h'>Resumo</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='side-row'><span>Laudos salvos</span><b>{_total}</b></div>"
        f"<div class='side-row'><span>Laudos hoje</span><b>{_hoje_n}</b></div>",
        unsafe_allow_html=True,
    )
    if _total:
        st.markdown(
            f"<div style='margin-top:10px'><div class='mini-bar'>"
            f"<div style='width:{_pct:.1f}%;background:#E0675F'></div>"
            f"<div style='width:{100 - _pct:.1f}%;background:#3FB68B'></div></div>"
            f"<div class='mini-leg'><span>Pneumonia: {_pneu}</span><span>Normal: {_norm}</span></div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='side-h'>Últimos laudos</div>", unsafe_allow_html=True)
        _linhas = ""
        for _, r in _h.tail(5).iloc[::-1].iterrows():
            _nome = escape(str(r["Nome do Paciente"]))
            _nome = _nome if len(_nome) <= 22 else _nome[:21] + "…"
            _cls = "badge-pneu" if r["Resultado do Modelo"] == "Pneumonia" else "badge-norm"
            _linhas += (
                f"<div class='side-last'><div><b>{escape(str(r['ID do Diagnóstico']))}</b>"
                f"<small>{_nome}</small></div>"
                f"<span class='badge {_cls}'>{escape(str(r['Resultado do Modelo']))}</span></div>"
            )
        st.markdown(_linhas, unsafe_allow_html=True)
    else:
        st.caption("Nenhum laudo salvo ainda.")

    st.markdown("---")
    st.caption("Protótipo acadêmico (TCC). Não aprovado como dispositivo médico e não "
               "substitui a avaliação de um profissional de saúde.")
