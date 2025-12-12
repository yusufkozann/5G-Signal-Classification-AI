import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

filename = 'RML2016.10a_dict.pkl'

print(f"Loading data from: {filename} ...")

try:
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        Xd = u.load()
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# --- DIAGNOSTIC STEP: Let's see what keys are actually inside ---
all_keys = list(Xd.keys())
# Extract unique modulation names and SNRs
unique_mods = sorted(list(set([k[0] for k in all_keys])))
unique_snrs = sorted(list(set([k[1] for k in all_keys])))

print("\n--- DATASET INVENTORY ---")
print(f"Available Modulations: {unique_mods}")
print(f"Available SNRs: {unique_snrs}")
print("-------------------------\n")

# --- PARAMETERS ---
# Let's try 'QAM16' instead of '16QAM'. If not found, we pick the first one available.
target_mod = 'QAM16' 
target_snr = 18

# Fallback mechanism
if (target_mod, target_snr) not in Xd:
    print(f"Warning: {target_mod} at {target_snr}dB not found!")
    print("Switching to the first available signal in the dataset...")
    target_mod = unique_mods[0] # Pick the first available modulation
    target_snr = unique_snrs[-1] # Pick the highest SNR available
    print(f"New Target: {target_mod} at {target_snr}dB")

# Fetch the data
signal_data = Xd[(target_mod, target_snr)][55] # 55th sample
I = signal_data[0]
Q = signal_data[1]

# --- PLOTTING ---
plt.style.use('dark_background')
plt.figure(figsize=(12, 6))

plt.plot(I, color='#00ffcc', label='In-Phase (I)', linewidth=2)
plt.plot(Q, color='#ff0066', label='Quadrature (Q)', linewidth=2, alpha=0.8)

plt.title(f"5G Signal Analysis: {target_mod} at {target_snr}dB SNR", 
          fontsize=14, fontweight='bold', color='white')
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.2, linestyle='--')

plt.figtext(0.5, 0.02, "Project: Deep Learning Based AMC | Platform: Python & TensorFlow", 
            ha="center", fontsize=10, color='gray')

output_filename = "signals_output.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nSUCCESS! Plot saved as: {output_filename}")
plt.show()