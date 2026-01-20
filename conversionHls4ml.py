#!/usr/bin/env python3

import numpy as np
import hls4ml
import os
import sys
import shutil
from tensorflow import keras
import pickle


def main():
    if not os.path.exists("model_fpga.h5"):
        sys.exit(1)

    model = keras.models.load_model("model_fpga.h5", compile=False)
    model.summary()

    DATASET_PATH = "./RML2016.10a_dict_v1.dat"
    if not os.path.exists(DATASET_PATH):
        sys.exit(1)

    with open(DATASET_PATH, "rb") as f:
        Xd = pickle.load(f, encoding="latin1")

    mods_subset = ['BPSK', 'QPSK', 'PAM4', 'GFSK']
    snrs_train = [10, 12, 14, 16, 18]

    X_list = []
    for mod in mods_subset:
        for snr in snrs_train:
            X_list.append(Xd[(mod, snr)].astype(np.float32))

    X = np.vstack(X_list)
    X = (X - np.mean(X)) / (np.std(X) + 1e-12)

    np.random.seed(2016)
    n = X.shape[0]
    train_idx = np.random.choice(range(n), size=int(0.8 * n), replace=False)
    test_idx = np.array(list(set(range(n)) - set(train_idx)))
    X_test = X[test_idx]

    OUTPUT_DIR = "hls4ml_prj"
    FPGA_PART = "xc7z020clg400-1"

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')

    config['Model'] = {
        'Precision': 'ap_fixed<16,6>',
        'ReuseFactor': 1,
        'Strategy': 'Latency'
    }

    for layer_name in config['LayerName'].keys():
        config['LayerName'][layer_name]['Precision'] = 'ap_fixed<16,6>'
        config['LayerName'][layer_name]['Strategy'] = 'Latency'

    try:
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=OUTPUT_DIR,
            part=FPGA_PART,
            clock_period=10,
            io_type='io_stream',
            backend='vivado'
        )
    except Exception:
        sys.exit(1)

    try:
        hls_model.write()
    except Exception:
        sys.exit(1)

    firmware_dir = os.path.join(OUTPUT_DIR, "firmware")
    if not os.path.exists(firmware_dir):
        sys.exit(1)

    try:
        hls_model.build(
            reset=True,
            csim=False,
            synth=True,
            cosim=False,
            validation=False,
            export=True,
            vsynth=False
        )
    except Exception:
        sys.exit(1)

    ip_path = os.path.join(OUTPUT_DIR, "myproject_prj", "solution1", "impl", "ip")
    if not (os.path.exists(ip_path) and os.listdir(ip_path)):
        sys.exit(1)

    report_file = os.path.join(
        OUTPUT_DIR, "myproject_prj", "solution1", "syn", "report", "myproject_csynth.rpt"
    )

    if os.path.exists(report_file):
        with open(report_file, 'r') as f:
            lines = f.readlines()
        with open("synthesis_report.txt", "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    main()

