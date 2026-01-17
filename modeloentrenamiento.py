# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 16:53:13 2026

@author: Lucía
"""

# ============================================================
# CNN2 (estilo VT-CNN2) con TensorFlow/Keras (TF2) usando channels_last (NHWC)
# Objetivo:
#  1) Cargar el dataset RML2016.10a (diccionario con señales IQ)
#  2) Preparar X (señales) e Y (etiquetas)
#  3) Entrenar una CNN para clasificar modulaciones
#  4) Evaluar y guardar el modelo entrenado
# ============================================================



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# -----------------------------
# 1) Carga del dataset (pickle)
# -----------------------------
import pickle

DATASET_PATH = "RML2016.10a_dict_v1.dat"

print("Cargando dataset:", DATASET_PATH)

# Abrimos el archivo binario y lo cargamos como diccionario (Xd)
# encoding="latin1" se usa porque el dataset original fue generado en Python2
with open(DATASET_PATH, "rb") as f:
    Xd = pickle.load(f, encoding="latin1")

# Extraer lista de SNRs y modulaciones a partir de las claves del diccionario
# Cada clave en Xd es: (modulación, snr)
snrs = sorted(list(set([k[1] for k in Xd.keys()])))  # lista de SNRs (ej: -20, -18, ..., 18)
mods = sorted(list(set([k[0] for k in Xd.keys()])))  # lista de modulaciones (ej: BPSK, QPSK, ...)
classes = mods  # “classes” será la lista de clases que queremos predecir

print("Num mods:", len(mods))
print("Num snrs:", len(snrs))

# -----------------------------
# Construir X (señales) y lbl (etiquetas)
# -----------------------------

# X_list será una lista con bloques de señales (cada bloque viene de Xd[(mod,snr)])
# lbl guardará una etiqueta (mod,snr) por cada ejemplo individual
X_list = []
lbl = []

# Recorremos todas las modulaciones y SNRs disponibles
for mod in mods:
    for snr in snrs:
        # Extrae un bloque de señales para esa modulación y SNR
        # Normalmente shape: (n, 2, 128)  -> n ejemplos, 2 canales (I,Q), 128 muestras
        x_block = Xd[(mod, snr)]
        X_list.append(x_block)

        # Añadimos una etiqueta (mod, snr) por cada ejemplo dentro del bloque
        for _ in range(x_block.shape[0]):
            lbl.append((mod, snr))

# Unimos todos los bloques para obtener un array grande con TODAS las señales
# Resultado: X.shape = (N_total, 2, 128)
X = np.vstack(X_list)

# Convertimos lbl a array para poder indexar fácilmente
lbl = np.array(lbl, dtype=object)

print("X.shape total:", X.shape)

# -----------------------------
# 2) Separación Train / Test
# -----------------------------

# Semilla para que el “shuffle” sea reproducible
np.random.seed(2016)

# Número total de ejemplos
n_examples = X.shape[0]

# Escogemos 50% para entrenamiento
n_train = int(n_examples * 0.5)

# train_idx: índices aleatorios para train
# test_idx: el resto de índices para test
train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
test_idx = np.array(list(set(range(0, n_examples)) - set(train_idx)))

# Construimos X_train y X_test seleccionando filas de X por índices
X_train = X[train_idx].astype(np.float32)  # float32 recomendado para TF
X_test  = X[test_idx].astype(np.float32)

# -----------------------------
# 3) Normalización RMS 
# -----------------------------
# Normaliza cada ejemplo para que tenga potencia "similar" y el modelo no dependa tanto de amplitud
# RMS = sqrt(mean(x^2))
def rms_normalize(x):
    p = np.sqrt(np.mean(x**2, axis=(1, 2), keepdims=True) + 1e-12)  # +1e-12 evita división por 0
    return x / p

X_train = rms_normalize(X_train)
X_test  = rms_normalize(X_test)

# -----------------------------
# 4) Convertir a channels_last 
# -----------------------------
# Pasamos de (N, 2, 128) a (N, 2, 128, 1)
# El último "1" es el canal (como si fuese una imagen de 1 canal), es necesario porque asi trabaja CNN
X_train = X_train[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

print("X_train (NHWC) shape:", X_train.shape)  # (N, 2, 128, 1)

# -----------------------------
# 5) Crear etiquetas one-hot (Y_train y Y_test)
# -----------------------------
# One-hot:
# si hay 8 clases y la clase correcta es la 2 -> [0,0,1,0,0,0,0,0]
def to_onehot(indices, n_classes):
    indices = np.array(indices, dtype=np.int64)
    y = np.zeros((len(indices), n_classes), dtype=np.float32)
    y[np.arange(len(indices)), indices] = 1.0
    return y

# Convertimos cada etiqueta lbl[i] = (mod, snr) a un índice de clase (según posición en mods)
Y_train_idx = [mods.index(lbl[i][0]) for i in train_idx]  # lbl[i][0] es la modulación
Y_test_idx  = [mods.index(lbl[i][0]) for i in test_idx]

# Creamos Y_train y Y_test como one-hot
Y_train = to_onehot(Y_train_idx, len(mods))
Y_test  = to_onehot(Y_test_idx, len(mods))

print("Y_train shape:", Y_train.shape, "Num classes:", len(classes))
print("Classes:", classes)

# -----------------------------
# 6) Definir el modelo CNN2 (channels_last)
# -----------------------------
dr = 0.5  # dropout rate: apaga neuronas al entrenar para evitar sobreajuste

# Modelo secuencial: capas en orden
model = keras.Sequential([
    # Entrada: “imagen” de tamaño (alto=2, ancho=128, canales=1)
    keras.Input(shape=(2, 128, 1)),

    # Añade ceros en el ancho (W): (0 arriba/abajo, 2 izquierda/derecha)
    keras.layers.ZeroPadding2D((0, 2)),

    # Conv1: 256 filtros, kernel 1x3
    # Busca patrones pequeños a lo largo del tiempo (ancho)
    keras.layers.Conv2D(
        filters=256,
        kernel_size=(1, 3),
        padding="valid",
        activation="relu",
        kernel_initializer="glorot_uniform",
        name="conv1"
    ),
    keras.layers.Dropout(dr),

    # Padding de nuevo y segunda convolución
    keras.layers.ZeroPadding2D((0, 2)),

    # Conv2: 80 filtros, kernel 2x3
    # Aquí ya combina información de I y Q (alto=2)
    keras.layers.Conv2D(
        filters=80,
        kernel_size=(2, 3),
        padding="valid",
        activation="relu",
        kernel_initializer="glorot_uniform",
        name="conv2"
    ),
    keras.layers.Dropout(dr),

    # Flatten: convierte mapas 2D a un vector 1D
    keras.layers.Flatten(),

    # Capa densa: combina las características aprendidas
    keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal", name="dense1"),
    keras.layers.Dropout(dr),

    # Salida: tantas neuronas como clases, con softmax para probabilidades
    keras.layers.Dense(len(classes), activation="softmax", kernel_initializer="he_normal", name="output_softmax"),
])

# Compilación: definimos cómo aprende
# - categorical_crossentropy: para clasificación multiclase con one-hot
# - Adam: optimizador estándar y estable
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# Muestra resumen de la red (capas, tamaños, parámetros)
model.summary()

# -----------------------------
# 7) Entrenamiento
# -----------------------------
nb_epoch = 100
batch_size = 1024

# Guardar “mejores pesos” durante entrenamiento
BEST_WEIGHTS_PATH = "cnn2_best_weights.h5"

# Guardar el modelo final completo (arquitectura + pesos), útil para hls4ml
FINAL_MODEL_PATH  = "modelo_final_pynq.h5"

# Callbacks:
# - ModelCheckpoint: guarda el modelo cuando mejora val_loss
# - EarlyStopping: para si no mejora tras X épocas y restaura lo mejor
callbacks = [
    keras.callbacks.ModelCheckpoint(
        BEST_WEIGHTS_PATH, monitor="val_loss", verbose=1, save_best_only=True, mode="auto"
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
    )
]

# Entrenamiento
history = model.fit(
    X_train, Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks=callbacks
)

# Guardar modelo completo (ideal para hls4ml)
# include_optimizer=False: no guardamos el estado del optimizador (no hace falta para inferencia)
model.save(FINAL_MODEL_PATH, include_optimizer=False)
print("Modelo completo guardado en:", FINAL_MODEL_PATH)

# -----------------------------
# 8) Evaluación final en test
# -----------------------------
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# -----------------------------
# 9) Gráfica de pérdida (loss) en train y validación
# -----------------------------
plt.figure()
plt.title("Training performance")
plt.plot(history.epoch, history.history["loss"], label="train loss")
plt.plot(history.epoch, history.history["val_loss"], label="val_loss")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 10) Matriz de confusión (global)
# -----------------------------
def plot_confusion_matrix(cm, title="Confusion matrix", cmap=plt.cm.Blues, labels=None):
    if labels is None:
        labels = []
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

# Predicciones del modelo sobre X_test
test_Y_hat = model.predict(X_test, batch_size=batch_size, verbose=0)

# conf[j,k] cuenta cuántas veces la clase real j se predijo como k
conf = np.zeros((len(classes), len(classes)), dtype=np.float64)
for i in range(X_test.shape[0]):
    j = int(np.argmax(Y_test[i, :]))      # clase real
    k = int(np.argmax(test_Y_hat[i, :]))  # clase predicha
    conf[j, k] += 1

# Normalizar por filas (para ver porcentajes)
confnorm = np.zeros_like(conf)
for i in range(len(classes)):
    row_sum = np.sum(conf[i, :])
    if row_sum > 0:
        confnorm[i, :] = conf[i, :] / row_sum

plt.figure()
plot_confusion_matrix(confnorm, labels=classes, title="Confusion Matrix (Overall)")
plt.show()

# -----------------------------
# 11) Accuracy por SNR
# -----------------------------
acc = {}

# Extraemos los SNRs de los ejemplos de test
test_SNRs = np.array([lbl[i][1] for i in test_idx])

# Repetimos evaluación por cada SNR
for snr in snrs:
    idx = np.where(test_SNRs == snr)[0]
    if len(idx) == 0:
        continue

    test_X_i = X_test[idx]
    test_Y_i = Y_test[idx]

    test_Y_i_hat = model.predict(test_X_i, batch_size=batch_size, verbose=0)

    # Confusión para este SNR
    conf_snr = np.zeros((len(classes), len(classes)), dtype=np.float64)
    for i in range(test_X_i.shape[0]):
        j = int(np.argmax(test_Y_i[i, :]))      # real
        k = int(np.argmax(test_Y_i_hat[i, :]))  # predicha
        conf_snr[j, k] += 1

    # Accuracy global para ese SNR
    cor = np.sum(np.diag(conf_snr))  # aciertos (diagonal)
    tot = np.sum(conf_snr)           # total
    overall_acc = (cor / tot) if tot > 0 else 0.0

    acc[snr] = float(overall_acc)
    print(f"SNR {snr:>3} -> Overall Accuracy: {overall_acc:.4f}")

# Curva accuracy vs SNR
plt.figure()
plt.plot(snrs, [acc.get(x, 0.0) for x in snrs])
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
plt.grid(True)
plt.show()

# -----------------------------
# 12) Guardar resultados (accuracy por SNR) en un archivo
# -----------------------------
with open("results_cnn2_d0.5.dat", "wb") as fd:
    pickle.dump(("CNN2", 0.5, acc), fd)

print("Resultados guardados en: results_cnn2_d0.5.dat")
