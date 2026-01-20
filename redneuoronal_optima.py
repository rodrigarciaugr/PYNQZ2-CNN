import os
import sys
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ============ DATASET ============
DATASET_PATH = "./RML2016.10a_dict_v1.dat"

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: No se encuentra el dataset en: {DATASET_PATH}")
    sys.exit(1)

mods_subset = ['BPSK', 'QPSK', 'PAM4', 'GFSK']
snrs_train = [10, 12, 14, 16, 18]

print("Cargando dataset...")
with open(DATASET_PATH, "rb") as f:
    Xd = pickle.load(f, encoding="latin1")

X_list, lbl = [], []
for mod in mods_subset:
    for snr in snrs_train:
        x_block = Xd[(mod, snr)].astype(np.float32)  # (N, 2, 128)
        X_list.append(x_block)
        lbl += [mod] * x_block.shape[0]

X = np.vstack(X_list)
print(f"Datos cargados: X.shape = {X.shape}")

# Normalización global (tal como estabas haciendo)
X = (X - np.mean(X)) / (np.std(X) + 1e-12)

def to_onehot(mod_list):
    idx = np.array([mods_subset.index(m) for m in mod_list], dtype=np.int64)
    y = np.zeros((len(idx), len(mods_subset)), dtype=np.float32)
    y[np.arange(len(idx)), idx] = 1.0
    return y

Y = to_onehot(lbl)

# Split
np.random.seed(2016)
n = X.shape[0]
train_idx = np.random.choice(range(n), size=int(0.8*n), replace=False)
test_idx = np.array(list(set(range(n)) - set(train_idx)))

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ============ MODELO ============
inputs = keras.Input(shape=(2, 128), name="input_1")
x = layers.Reshape((2, 128, 1), name="reshape")(inputs)

x = layers.Conv2D(4, (1, 3), padding="valid", activation="relu",
                  kernel_initializer="glorot_uniform", name="conv2d_1")(x)
x = layers.MaxPooling2D((1, 2), name="maxpool1")(x)
x = layers.Conv2D(4, (2, 3), padding="valid", activation="relu",
                  kernel_initializer="glorot_uniform", name="conv2d_2")(x)

x = layers.Flatten(name="flatten")(x)
x = layers.Dense(8, activation="relu", name="dense_1")(x)
outputs = layers.Dense(len(mods_subset), activation="softmax", name="output_softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ============ TRAIN ============
cb = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
]

history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_test, Y_test),
    callbacks=cb,
    verbose=2
)

loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Accuracy test: {acc*100:.2f}%")

# Guardar
model.save("model_fpga.h5", include_optimizer=False)
print("Guardado: model_fpga.h5")