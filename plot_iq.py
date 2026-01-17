import pickle

import numpy as np

# ==============================================================================
# EXPLICACIÓN DEL CÓDIGO
# ==============================================================================
# Este script visualiza las componentes IQ (In-phase y Quadrature) de señales
# moduladas con diferentes esquemas de modulación a distintos niveles de ruido (SNR).
#
# El dataset contiene 1000 muestras por cada combinación de modulación y SNR.
# Cada muestra tiene 128 puntos de una señal compleja: [I, Q] donde:
#   I = componente en fase (real)
#   Q = componente en cuadratura (imaginaria)
#
# El código genera 2 gráficos interactivos con Plotly:
#   1. Diagrama de constelación: muestra todos los puntos IQ (nube de puntos)
#   2. Espectro de potencia: análisis en frecuencia de la señal
# ==============================================================================

# Carga el dataset (diccionario pickle con todas las modulaciones y SNRs)
with open('RML2016.10a_dict.dat', 'rb') as f:
    dataset = pickle.load(f)

# ==============================================================================
# SELECCIONA MODULACIÓN Y SNR
# ==============================================================================
# Modulaciones disponibles:
#   - BPSK        (Binary Phase Shift Keying) - 2 símbolos
#   - QPSK        (Quadrature PSK) - 4 símbolos
#   - 8PSK        (8-ary PSK) - 8 símbolos
#   - PAM4        (Pulse Amplitude Modulation) - 4 símbolos
#   - QAM16       (16-ary QAM) - 16 símbolos
#   - QAM64       (64-ary QAM) - 64 símbolos
#   - CPFSK       (Continuous Phase FSK) - cambios de frecuencia
#   - GFSK        (Gaussian FSK) - FSK suavizado
#
# SNR disponibles: -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
# SNR bajo = mucho ruido (señal degradada)
# SNR alto = poco ruido (señal limpia)

mod_type = 'QAM16'  # Cambiar a: BPSK, QPSK, 8PSK, PAM4, QAM64, CPFSK, GFSK
snr = 18            # Cambiar a: -20 a 18 (en pasos de 2)

# ==============================================================================
# EXTRAE Y PREPARA LOS DATOS
# ==============================================================================
# Obtiene 1000 muestras para la modulación y SNR seleccionados
samples = dataset[(mod_type, snr)]  # Shape: (1000, 2, 128)
#   - 1000: número de muestras
#   - 2: [componente I, componente Q]
#   - 128: puntos por muestra

# Separa componentes I (parte real) y Q (parte imaginaria)
i_component = samples[:, 0, :]  # Extrae todos los valores de I de todas las muestras
q_component = samples[:, 1, :]  # Extrae todos los valores de Q de todas las muestras

# Para encontrar el instante óptimo de muestreo, probamos los 8 posibles offsets.
# El offset correcto será aquel que maximice la potencia promedio de los símbolos,
# ya que la energía es máxima en el centro del símbolo.
best_offset = -1
max_power = -1
for offset in range(8):
    # Downsample con el offset actual
    i_test = i_component[:, offset::8]
    q_test = q_component[:, offset::8]
    # Calcula la potencia promedio
    power = np.mean(i_test**2 + q_test**2)
    if power > max_power:
        max_power = power
        best_offset = offset

print(f"Offset óptimo encontrado: {best_offset}")

# Usa el mejor offset para obtener los puntos de la constelación
i_symbols = i_component[:, best_offset::8].flatten()
q_symbols = q_component[:, best_offset::8].flatten()
import matplotlib.pyplot as plt

# ==============================================================================
# CREA LOS 2 GRÁFICOS CON MATPLOTLIB
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ==============================================================================
# GRÁFICO 1: DIAGRAMA DE CONSTELACIÓN
# ==============================================================================
ax1.scatter(i_symbols, q_symbols, s=4, alpha=0.4)
ax1.set_title(f'Constelación: {mod_type} @ {snr} dB SNR')
ax1.set_xlabel('I (Componente Real)')
ax1.set_ylabel('Q (Componente Imaginaria)')
ax1.set_aspect('equal', 'box')
ax1.grid(True)

# ==============================================================================
# GRÁFICO 2: ESPECTRO DE POTENCIA
# ==============================================================================


# Aplica ventana de Hann para reducir lóbulos secundarios
window = np.hanning(i_component.shape[1])
i_windowed = i_component * window
q_windowed = q_component * window

# Calcula FFT
fft_i = np.fft.fft(i_windowed, axis=1)
fft_q = np.fft.fft(q_windowed, axis=1)

# Calcula potencia promedio y la convierte a dB
power = np.mean(np.abs(fft_i)**2 + np.abs(fft_q)**2, axis=0)
power_db = 10 * np.log10(power)
power_shifted = np.fft.fftshift(power_db)

# Frecuencias normalizadas
freq = np.fft.fftfreq(i_component.shape[1])
freq_shifted = np.fft.fftshift(freq)

ax2.plot(freq_shifted, power_shifted, color='r')
ax2.set_title(f'Espectro de Potencia: {mod_type} @ {snr} dB SNR')
ax2.set_xlabel('Frecuencia Normalizada')
ax2.set_ylabel('Potencia (dB)')
ax2.grid(True)

# ==============================================================================
# ACTUALIZA LAYOUT GENERAL Y MUESTRA
# ==============================================================================
fig.suptitle(f'Análisis IQ: {mod_type} @ {snr} dB SNR', fontsize=16)
plt.tight_layout(rect=(0, 0, 1, 0.96))

# ==============================================================================
# GUARDA Y MUESTRA
# ==============================================================================
# plt.savefig('iq_constellation.png')
print(f"Plot saved as 'iq_constellation.png'")
print(f"\nDataset info:")
print(f"Modulation: {mod_type}")
print(f"SNR: {snr} dB")
print(f"Total points plotted: {len(i_symbols)}")
print(f"Available modulations: {sorted(set(key[0] for key in dataset.keys()))}")
print(f"Available SNR values: {sorted(set(key[1] for key in dataset.keys()))}")

plt.show()

