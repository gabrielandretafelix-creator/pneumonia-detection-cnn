import os
import numpy as np
import tensorflow as tf
import medmnist
from medmnist import INFO

from sklearn.utils import class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

IMAGE_SIZE = 128

# Resolução NATIVA baixada do MedMNIST: 28, 64, 128 ou 224.
# O pneumoniamnist "padrão" vem em 28x28 - fazer resize disso para 128
# só interpola pixels, não recupera detalhe real do raio-x.
# Testado: 128 deu 91% de acurácia, 224 deu 90% (dentro da margem de
# ruído normal entre execuções, mas 128 também é mais rápido de treinar).
DATASET_SIZE = 128

BATCH_SIZE = 32

EPOCHS_HEAD = 15          # fase 1: treina apenas o classificador (backbone congelado)
EPOCHS_FINE = 40          # fase 2: fine-tuning do backbone

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5   # LR bem menor para não destruir os pesos pré-treinados

# Troque aqui para testar backbones diferentes: "densenet121" ou "efficientnetb0"
BACKBONE_NAME = "efficientnetb0"

# Quanto MENOR esse número, MAIS camadas do backbone ficam livres para
# fine-tuning (ou seja, "menos congelado"). Antes estava 300 (DenseNet121
# tem ~427 camadas), agora liberamos bem mais camadas.
FINE_TUNE_AT = 100

MODEL_PATH = "Pneumonia-Model-melhor.keras"

TREINAR_NOVAMENTE = False  # False = reaproveita o modelo já salvo (não retreina)

# SEED: fixe um valor (ex: 42) para reprodutibilidade, ou rode o script
# várias vezes trocando esse número (42, 43, 44...) para reportar no TCC
# a média e o desvio padrão da acurácia entre execuções - mais robusto
# cientificamente do que reportar um único resultado isolado.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# CARREGANDO DATASET
# ==========================================================

info = INFO["pneumoniamnist"]

DataClass = getattr(medmnist, info["python_class"])

train_dataset_med = DataClass(split="train", download=True, size=DATASET_SIZE)
val_dataset_med = DataClass(split="val", download=True, size=DATASET_SIZE)
test_dataset_med = DataClass(split="test", download=True, size=DATASET_SIZE)

# Mantemos as imagens em uint8 (0-255) e SEM redimensionar aqui.
# O resize, conversão para RGB e a normalização específica do backbone
# são feitos dentro do pipeline tf.data (mais rápido e usa muito menos RAM).

train_images = train_dataset_med.imgs
train_labels = train_dataset_med.labels.squeeze().astype("float32")

val_images = val_dataset_med.imgs
val_labels = val_dataset_med.labels.squeeze().astype("float32")

test_images = test_dataset_med.imgs
test_labels = test_dataset_med.labels.squeeze().astype("float32")

# ==========================================================
# BACKBONES DISPONÍVEIS
# ==========================================================

BACKBONES = {
    "densenet121": {
        "builder": tf.keras.applications.DenseNet121,
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "efficientnetb0": {
        "builder": tf.keras.applications.EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
}

backbone_config = BACKBONES[BACKBONE_NAME]

# ==========================================================
# PRÉ-PROCESSAMENTO (dentro do pipeline tf.data)
# ==========================================================

def preprocess(image, label):
    image = tf.expand_dims(image, axis=-1)            # (H, W) -> (H, W, 1)
    image = tf.image.grayscale_to_rgb(image)           # (H, W, 3) -> exigido pelo backbone pré-treinado
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    image = backbone_config["preprocess"](image)
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def make_dataset(images, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(5000)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = make_dataset(train_images, train_labels, training=True)
val_dataset = make_dataset(val_images, val_labels, training=False)
test_dataset = make_dataset(test_images, test_labels, training=False)

# ==========================================================
# TEST-TIME AUGMENTATION (TTA)
# ==========================================================
# Em vez de prever cada imagem uma única vez, geramos algumas
# "visões" diferentes dela (original, espelhada horizontalmente,
# brilho levemente ajustado) e tiramos a MÉDIA das probabilidades.
# Isso reduz o ruído da previsão individual e costuma dar um
# pequeno ganho de acurácia sem precisar retreinar nada.

def tta_preprocess(image, label, flip=False, brightness_delta=0.0):
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    if flip:
        image = tf.image.flip_left_right(image)
    if brightness_delta != 0.0:
        image = tf.image.adjust_brightness(image, brightness_delta)
    image = backbone_config["preprocess"](image)
    return image, label


def make_tta_dataset(images, labels, flip=False, brightness_delta=0.0):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(
        lambda x, y: tta_preprocess(x, y, flip=flip, brightness_delta=brightness_delta),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# Cada tupla é uma "visão" diferente da mesma imagem: (flip, delta_de_brilho)
TTA_VIEWS = [
    (False, 0.0),   # imagem original
    (True, 0.0),    # espelhada horizontalmente
    (False, 0.05),  # um pouco mais clara
    (False, -0.05), # um pouco mais escura
]


def tta_predict(trained_model, images, labels):
    all_probs = []
    for flip, delta in TTA_VIEWS:
        ds = make_tta_dataset(images, labels, flip=flip, brightness_delta=delta)
        probs = trained_model.predict(ds, verbose=0).ravel()
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)

# ==========================================================
# CLASS WEIGHT AUTOMÁTICO
# ==========================================================

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print(class_weights)

# ==========================================================
# CALLBACKS
# (usados nas DUAS fases de treino, por isso o "best" fica
#  preservado entre elas -> só é salvo o modelo com a maior
#  val_accuracy de todo o treinamento, seja na fase 1 ou 2)
# ==========================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ==========================================================
# CRIANDO O MODELO (Transfer Learning)
# ==========================================================

def criar_modelo():
    base_model = backbone_config["builder"](
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    base_model.trainable = False  # fase 1: backbone congelado

    print(f"\nBackbone '{BACKBONE_NAME}' tem {len(base_model.layers)} camadas.\n")

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = base_model(inputs, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    return model, base_model


model, base_model = criar_modelo()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)

# ==========================================================
# FASE 1: TREINO DO CLASSIFICADOR (BACKBONE CONGELADO)
# ==========================================================

history1 = None
history = None

if TREINAR_NOVAMENTE or not os.path.exists(MODEL_PATH):

    print("\n=== FASE 1: treinando cabeça do modelo (backbone congelado) ===\n")

    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS_HEAD,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    # ======================================================
    # FASE 2: FINE-TUNING (descongela parte do backbone)
    # ======================================================

    print("\n=== FASE 2: fine-tuning do backbone ===\n")

    base_model.trainable = True

    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    # IMPORTANTE: mesmo nas camadas "descongeladas", mantemos as
    # BatchNormalization travadas (trainable=False). Com batch pequeno
    # e poucos dados, liberar o BN para treino faz as estatísticas de
    # normalização ficarem instáveis e o modelo "quebra" logo nas
    # primeiras épocas do fine-tuning (loss dispara, accuracy despenca).
    n_bn_frozen = 0
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            n_bn_frozen += 1

    n_trainable = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"Camadas do backbone liberadas para treino: {n_trainable}/{len(base_model.layers)}")
    print(f"Camadas de BatchNormalization mantidas congeladas: {n_bn_frozen}\n")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )

    initial_epoch = history1.epoch[-1] + 1

    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=initial_epoch + EPOCHS_FINE,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    # Junta as duas fases para os gráficos ficarem contínuos
    history = {}
    for key in history1.history:
        history[key] = history1.history[key] + history2.history[key]

else:

    print("\nCarregando modelo salvo...\n")

    model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================================
# CARREGA O MELHOR MODELO SALVO (maior val_accuracy) PARA AVALIAR
# ==========================================================

print("\nCarregando o melhor checkpoint salvo em disco para avaliação final...\n")

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================================
# AVALIAÇÃO
# ==========================================================

print("\n==============================")
print("AVALIANDO MODELO")
print("==============================")

results = model.evaluate(test_dataset, verbose=0)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# ==========================================================
# BUSCA DO MELHOR THRESHOLD (usando o conjunto de VALIDAÇÃO)
# ==========================================================
# Em vez de usar 0.5 fixo, testamos vários thresholds na validação
# e escolhemos o que maximiza a balanced accuracy (média entre o
# recall de Normal e o recall de Pneumonia). Isso ajuda a corrigir
# o desbalanceamento que aparece no classification report
# (ex.: recall baixo de "Normal" mesmo com class_weight).

val_predictions = tta_predict(model, val_images, val_labels)

best_threshold = 0.5
best_score = -1

for t in np.arange(0.30, 0.71, 0.01):
    y_val_pred = (val_predictions > t).astype(int)
    score = balanced_accuracy_score(val_labels.astype(int), y_val_pred)
    if score > best_score:
        best_score = score
        best_threshold = t

print(f"\nMelhor threshold encontrado na validação (com TTA): {best_threshold:.2f} "
      f"(balanced accuracy = {best_score:.4f})\n")

# ==========================================================
# PREVISÕES (com Test-Time Augmentation)
# ==========================================================

predictions = tta_predict(model, test_images, test_labels)

y_pred = (predictions > best_threshold).astype(int)

y_true = test_labels.astype(int)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Pneumonia"]
    )
)

# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Pneumonia"],
    yticklabels=["Normal", "Pneumonia"]
)

plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# ==========================================================
# CURVA ROC E PRECISION-RECALL
# ==========================================================
# 'predictions' aqui são as probabilidades (0 a 1) de Pneumonia,
# já calculadas com TTA. Usamos elas (não o y_pred binário) para
# construir as curvas.

fpr, tpr, _ = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

precision_vals, recall_vals, _ = precision_recall_curve(y_true, predictions)
avg_precision = average_precision_score(y_true, predictions)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="steelblue", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classificador aleatório")
plt.xlabel("Taxa de Falso Positivo (1 - Especificidade)")
plt.ylabel("Taxa de Verdadeiro Positivo (Recall)")
plt.title("Curva ROC")
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(recall_vals, precision_vals, label=f"AP = {avg_precision:.4f}", color="darkorange", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.legend(loc="lower left")

plt.tight_layout()
plt.savefig("curva_roc_pr.png", dpi=150)
plt.show()

print(f"\nAUC (ROC): {roc_auc:.4f}")
print(f"Average Precision (PR-AUC): {avg_precision:.4f}\n")

# ==========================================================
# GRAD-CAM (interpretabilidade)
# ==========================================================
# Mostra, para algumas imagens de teste, um mapa de calor indicando
# quais regiões da imagem mais influenciaram a decisão do modelo.
# Útil para verificar se o modelo está "olhando" para os pulmões
# (esperado) e não para artefatos irrelevantes da imagem.

def gerar_grad_cam(keras_model, image_batch):
    """
    image_batch: array já pré-processado, shape (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    Retorna o heatmap normalizado (0 a 1).

    Observação técnica: em vez de tentar reaproveitar os tensores internos
    do modelo carregado do disco (o que pode gerar erro de "grafo
    desconectado" no Keras 3, especialmente após load_model), reconstruímos
    manualmente o caminho: chamamos o backbone e depois cada camada da
    "cabeça" (GAP, BatchNorm, Dropout, Dense...) em sequência, a partir do
    mesmo tensor de entrada. Isso garante que o grafo fique sempre
    conectado corretamente.
    """
    backbone = keras_model.get_layer(index=1)

    inputs = keras_model.input

    conv_output = backbone(inputs, training=False)

    x = conv_output
    for layer in keras_model.layers[2:]:
        x = layer(x, training=False)

    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_output, x])

    with tf.GradientTape() as tape:
        conv_output_val, predictions_gc = grad_model(image_batch, training=False)
        loss = predictions_gc[:, 0]  # saída sigmoide única (prob. de Pneumonia)

    grads = tape.gradient(loss, conv_output_val)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output_val = conv_output_val[0]
    heatmap = conv_output_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def mostrar_grad_cam(n_exemplos=6):
    # Pega algumas imagens de teste (uma mistura de acertos/erros, se possível)
    indices = np.random.choice(len(test_images), size=n_exemplos, replace=False)

    plt.figure(figsize=(15, 5 * ((n_exemplos + 2) // 3)))

    for i, idx in enumerate(indices):
        img_raw = test_images[idx]
        img_batch, _ = tta_preprocess(
            tf.convert_to_tensor(img_raw), tf.constant(0.0), flip=False, brightness_delta=0.0
        )
        img_batch = tf.expand_dims(img_batch, axis=0)

        heatmap = gerar_grad_cam(model, img_batch)
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis], (IMAGE_SIZE, IMAGE_SIZE)
        ).numpy().squeeze()

        prob = float(model.predict(img_batch, verbose=0)[0][0])
        pred_label = "Pneumonia" if prob > best_threshold else "Normal"
        real_label = "Pneumonia" if y_true[idx] == 1 else "Normal"

        plt.subplot((n_exemplos + 2) // 3, 3, i + 1)
        plt.imshow(img_raw, cmap="gray")
        plt.imshow(heatmap_resized, cmap="jet", alpha=0.4)
        acerto = "✓" if pred_label == real_label else "✗"
        plt.title(f"{acerto} Real: {real_label} | Previsto: {pred_label} ({prob:.2f})", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("grad_cam_exemplos.png", dpi=150)
    plt.show()


print("\nGerando visualizações Grad-CAM...\n")
mostrar_grad_cam(n_exemplos=6)

# ==========================================================
# GRÁFICOS
# ==========================================================

if history is not None:

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Treino")
    plt.plot(history["val_accuracy"], label="Validação")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Treino")
    plt.plot(history["val_loss"], label="Validação")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
