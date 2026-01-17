import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================
# Carga el dataset (diccionario pickle con todas las modulaciones y SNRs)
with open('RML2016.10a_dict.dat', 'rb') as f:
    dataset = pickle.load(f)

# Obtiene la lista de modulaciones únicas disponibles en el dataset
modulations = sorted(list(set(key[0] for key in dataset.keys())))
snr_to_plot = 18

print(f"Modulaciones a plotear: {modulations}")
print(f"SNR seleccionado: {snr_to_plot} dB")

# ==============================================================================
# CREACIÓN DE GRÁFICOS
# ==============================================================================
# Crea una figura con un subplot para cada modulación
num_modulations = len(modulations)
fig, axes = plt.subplots(num_modulations, 1, figsize=(12, 2 * num_modulations), sharex=True)
fig.suptitle(f'Formas de Onda de 10 Vectores (1280 muestras) para SNR={snr_to_plot} dB', fontsize=16)

for i, mod_type in enumerate(modulations):
    ax = axes[i]
    
    # Obtiene los datos para la modulación y SNR seleccionados
    samples = dataset[(mod_type, snr_to_plot)]  # Shape: (1000, 2, 128)
    
    # Toma los primeros 10 vectores
    ten_vectors = samples[90:92, :, :]  # Shape: (10, 2, 128)
    
    # Separa y concatena los componentes I y Q para tener una forma de onda continua
    i_waveform = ten_vectors[:, 0, :].flatten()  # 10 * 128 = 1280 muestras
    q_waveform = ten_vectors[:, 1, :].flatten()  # 10 * 128 = 1280 muestras
    
    # Crea un eje de tiempo en número de muestras
    time_axis = np.arange(len(i_waveform))
    
    # Plotea las formas de onda I y Q
    ax.plot(time_axis, i_waveform, color='blue', label='I component', alpha=0.7)
    ax.plot(time_axis, q_waveform, color='red', label='Q component', alpha=0.7)
    
    # Calcula y plotea la magnitud
    magnitude = np.sqrt(i_waveform**2 + q_waveform**2)
    ax.plot(time_axis, magnitude, color='green', label='Magnitud', linewidth=1.5)
    
    ax.set_ylabel(mod_type)
    ax.legend(loc='upper right')
    ax.grid(True)

axes[-1].set_xlabel('Número de Muestra')
plt.tight_layout(rect=(0, 0.03, 1, 0.96))

# ==============================================================================
# GUARDA Y MUESTRA
# ==============================================================================
#plt.savefig('waveforms.png')
print("\nPlot guardado como 'waveforms.png'")
plt.show()
