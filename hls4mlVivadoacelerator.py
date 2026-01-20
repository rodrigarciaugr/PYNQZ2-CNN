#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import time
import pickle
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
import hls4ml


def to_onehot(mod_list, mods_subset):
    idx = np.array([mods_subset.index(m) for m in mod_list], dtype=np.int64)
    y = np.zeros((len(idx), len(mods_subset)), dtype=np.float32)
    y[np.arange(len(idx)), idx] = 1.0
    return y


def main():
    # =========================
    # 1) CARGA Y PREPARACIÓN DE DATOS
    # =========================
    DATASET_PATH = "./RML2016.10a_dict_v1.dat"
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"No existe el dataset en: {DATASET_PATH}")

    print("Cargando dataset...")
    with open(DATASET_PATH, "rb") as f:
        Xd = pickle.load(f, encoding="latin1")

    mods_subset = ["BPSK", "QPSK", "PAM4", "GFSK"]
    snrs_train = [10, 12, 14, 16, 18]

    X_list, lbl = [], []
    for mod in mods_subset:
        for snr in snrs_train:
            x_block = Xd[(mod, snr)].astype(np.float32)  # (1000, 2, 128)
            X_list.append(x_block)
            lbl += [mod] * x_block.shape[0]

    X = np.vstack(X_list).astype(np.float32)
    # Normalización global
    X = (X - np.mean(X)) / (np.std(X) + 1e-12)
    Y = to_onehot(lbl, mods_subset)

    # Split train/test
    np.random.seed(2016)
    n = X.shape[0]
    train_idx = np.random.choice(range(n), size=int(0.8 * n), replace=False)
    test_idx = np.array(list(set(range(n)) - set(train_idx)))

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_test : {X_test.shape},  Y_test : {Y_test.shape}")

    # =========================
    # 2) MODELO KERAS (CNN)
    # =========================
    inputs = keras.Input(shape=(2, 128), name="input_1")
    x = layers.Reshape((2, 128, 1), name="reshape_1")(inputs)
    x = layers.Conv2D(
        4, (1, 3),
        padding="valid",
        activation="relu",
        kernel_initializer="glorot_uniform",
        name="conv2d_1"
    )(x)
    x = layers.MaxPooling2D((1, 2), name="maxpool1")(x)
    x = layers.Conv2D(
        4, (2, 3),
        padding="valid",
        activation="relu",
        kernel_initializer="glorot_uniform",
        name="conv2d_2"
    )(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(8, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(len(mods_subset), activation="softmax", name="output_softmax")(x)

    model = keras.Model(inputs, outputs, name="cnn_rml")
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("\nEntrenando modelo...")
    history = model.fit(
        X_train, Y_train,
        epochs=20,
        batch_size=128,
        validation_data=(X_test, Y_test),
        verbose=1
    )
    
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\n=== Accuracy final: {acc * 100:.2f}% ===")

    # Guardar modelo
    model.save("model_oficial.h5", include_optimizer=False)
    print("Modelo guardado: model_oficial.h5")

    # =========================
    # 3) CONVERSIÓN A HLS4ML - MÉTODO OFICIAL DE EJEMPLOS
    # =========================
    OUTPUT_DIR = "hls_PYNQ_AXI_STREAM"
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    print("\n=== Configurando conversión a HLS4ML (método oficial) ===")
    
    # Paso 1: Crear configuración base del modelo (SIN default_precision aquí)
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    
    # Paso 2: Configurar precisión y optimizaciones DESPUÉS
    # IMPORTANTE: No usar default_precision en config_from_keras_model
    # Se configura aquí en el diccionario de configuración
    config['Model']['Precision'] = 'ap_fixed<16,6>'
    config['Model']['ReuseFactor'] = 64  # CRÍTICO para PYNQ-Z2
    config['Model']['Strategy'] = 'Latency'
    
    # Paso 3: Configuración específica de softmax
    config['LayerName']['output_softmax']['exp_table_t'] = 'ap_fixed<18,8>'
    config['LayerName']['output_softmax']['inv_table_t'] = 'ap_fixed<18,4>'
    
    # Paso 4: Ajustar ReuseFactor por capa si es necesario
    # Para dispositivos pequeños como PYNQ-Z2, aumentar ReuseFactor en capas Dense
    config['LayerName']['dense_1']['ReuseFactor'] = 64
    
    print("Configuración del modelo:")
    print(f"  - Precision: {config['Model']['Precision']}")
    print(f"  - ReuseFactor: {config['Model']['ReuseFactor']}")
    print(f"  - Strategy: {config['Model']['Strategy']}")

    # NOTA: El AcceleratorConfig se crea AUTOMÁTICAMENTE cuando usas
    # backend='VivadoAccelerator' en convert_from_keras_model()
    # No es necesario crearlo manualmente como en tu código original
    
    print("\n=== Configuración completa preparada ===")

    # =========================
    # 5) CONVERTIR MODELO A HLS4ML
    # =========================
    print("\nConvirtiendo modelo a HLS4ML con backend VivadoAccelerator...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=OUTPUT_DIR,
        backend='VivadoAccelerator',
        board='pynq-z2'
    )
    
    print("✓ Conversión completada")
    
    # Compilar el modelo HLS
    hls_model.compile()
    print("✓ Modelo HLS compilado")

    # =========================
    # 6) BUILD (SÍNTESIS + EXPORT + BITFILE)
    # =========================
    print("\n" + "="*70)
    print("INICIANDO BUILD - Este proceso tomará 15-45 minutos")
    print("="*70)
    print("Etapas:")
    print("  1. C Synthesis (HLS) - ~5-10 min")
    print("  2. Export RTL - ~1 min")
    print("  3. Create Block Design (Vivado) - ~2-5 min")
    print("  4. Generate Bitstream - ~10-30 min")
    print("="*70 + "\n")
    
    start = time.time()
    
    try:
        hls_model.build(
            reset=True,      # Limpia builds anteriores
            csim=False,      # Skip C simulation (ahorra tiempo)
            synth=True,      # Síntesis HLS (REQUERIDO)
            cosim=False,     # Skip co-simulation (ahorra tiempo)
            validation=False,
            export=True,     # Exportar IP RTL (REQUERIDO)
            vsynth=False,    # Skip logic synthesis
            bitfile=True     # Generar bitstream .bit y .hwh (REQUERIDO)
        )
        
        elapsed = time.time() - start
        print("\n" + "="*70)
        print(f"✓✓✓ BUILD COMPLETADO EXITOSAMENTE en {elapsed/60:.1f} minutos ✓✓✓")
        print("="*70)
        
    except Exception as e:
        elapsed = time.time() - start
        print("\n" + "="*70)
        print(f"✗✗✗ ERROR EN BUILD después de {elapsed/60:.1f} minutos ✗✗✗")
        print("="*70)
        print(f"\nError: {str(e)}")
        print("\nPosibles causas:")
        print("  - Vivado no está en PATH")
        print("  - Versión incorrecta de Vivado (usa 2019.2 o 2020.1)")
        print("  - Problemas de licencia")
        print("  - Recursos insuficientes para PYNQ-Z2")
        raise

    # =========================
    # 7) VERIFICACIÓN DE ARCHIVOS GENERADOS
    # =========================
    print("\n=== VERIFICANDO ARCHIVOS GENERADOS ===\n")
    
    # Estructura esperada
    vivado_dir = os.path.join(OUTPUT_DIR, "myproject_vivado_accelerator")
    firmware_dir = os.path.join(OUTPUT_DIR, "firmware")
    
    # Buscar bitstream y hardware handoff
    bitfile_found = False
    hwh_found = False
    
    if os.path.isdir(vivado_dir):
        print(f"✓ Directorio Vivado: {vivado_dir}")
        for root, dirs, files in os.walk(vivado_dir):
            for f in files:
                if f.endswith('.bit'):
                    bitfile_path = os.path.join(root, f)
                    print(f"  ✓✓ BITSTREAM: {bitfile_path}")
                    bitfile_found = True
                if f.endswith('.hwh'):
                    hwh_path = os.path.join(root, f)
                    print(f"  ✓✓ HWH: {hwh_path}")
                    hwh_found = True
    
    if firmware_dir and os.path.isdir(firmware_dir):
        print(f"✓ Directorio firmware: {firmware_dir}")
    
    # Verificar interfaz AXI-Stream en código fuente
    print("\n=== VERIFICANDO INTERFAZ AXI-STREAM ===\n")
    if os.path.isdir(firmware_dir):
        try:
            # Buscar archivos .cpp y .h
            found_axis = False
            for fname in os.listdir(firmware_dir):
                if fname.endswith(('.cpp', '.h')):
                    fpath = os.path.join(firmware_dir, fname)
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if 'axis' in content.lower() or 'hls::stream' in content:
                            print(f"✓ Interfaz stream encontrada en: {fname}")
                            found_axis = True
                            # Mostrar algunas líneas relevantes
                            for i, line in enumerate(content.splitlines()):
                                if 'axis' in line.lower() or 'hls::stream' in line:
                                    print(f"    Línea {i+1}: {line.strip()[:80]}")
                                    if i > 5:  # Solo primeras ocurrencias
                                        break
                            break
            
            if not found_axis:
                print("⚠ No se detectó 'axis' explícitamente en firmware/")
                print("  (Esto puede ser normal, VivadoAccelerator usa wrappers)")
        
        except Exception as e:
            print(f"⚠ No se pudo verificar código fuente: {e}")
    
    # =========================
    # 8) RESUMEN FINAL
    # =========================
    print("\n" + "="*70)
    print("RESUMEN DE LA CONVERSIÓN")
    print("="*70)
    print(f"Modelo: {model.name}")
    print(f"Parámetros: {model.count_params():,}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Backend: VivadoAccelerator")
    print(f"Board: pynq-z2")
    print(f"ReuseFactor: {config['Model']['ReuseFactor']}")
    print(f"Precisión: {config['Model']['Precision']}")
    
    if bitfile_found and hwh_found:
        print(f"\n✓✓✓ ARCHIVOS PARA PYNQ GENERADOS CORRECTAMENTE ✓✓✓")
    else:
        print(f"\n⚠⚠⚠ FALTAN ARCHIVOS (.bit o .hwh) ⚠⚠⚠")
    
    print("\n" + "="*70)
    print("PRÓXIMOS PASOS PARA DEPLOYMENT EN PYNQ-Z2")
    print("="*70)
    print("""
1. Conectar a tu PYNQ-Z2 vía SSH o Jupyter:
   ssh xilinx@192.168.2.99  (password: xilinx)

2. Copiar archivos .bit y .hwh a PYNQ:
   scp *.bit *.hwh xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/

3. En Jupyter Notebook de PYNQ, ejecutar:
   
   from pynq import Overlay
   import numpy as np
   
   # Cargar overlay
   overlay = Overlay('tu_archivo.bit')
   
   # Acceder al DMA
   dma = overlay.axi_dma_0
   
   # Preparar buffers
   from pynq import allocate
   input_buffer = allocate(shape=(2, 128), dtype=np.float32)
   output_buffer = allocate(shape=(4,), dtype=np.float32)
   
   # Llenar input con tus datos IQ
   input_buffer[:] = tus_datos_normalizados
   
   # Enviar datos y recibir predicción
   dma.sendchannel.transfer(input_buffer)
   dma.recvchannel.transfer(output_buffer)
   dma.sendchannel.wait()
   dma.recvchannel.wait()
   
   # Resultado
   prediccion = np.argmax(output_buffer)
   print(f"Modulación detectada: {mods_subset[prediccion]}")

4. Documentación:
   - PYNQ: https://pynq.readthedocs.io/
   - hls4ml: https://fastmachinelearning.org/hls4ml/
""")
    print("="*70)


if __name__ == "__main__":
    main()