# Deep Learning Based Automatic Modulation Classification (AMC)

This project implements a Convolutional Neural Network (CNN) architecture to classify modulation types of radio signals, focusing on high accuracy even in noisy environments. The study is based on the widely used **RadioML 2016.10a** dataset.

## 🚀 Project Overview

Automatic Modulation Classification (AMC) is a critical component for next-generation wireless communication systems (5G/6G) and cognitive radio. This project demonstrates:

- Preprocessing of raw **I/Q radio signals**.
- Implementation of a **CNN model** using TensorFlow/Keras.
- Performance evaluation using **Confusion Matrices** and **Accuracy vs. SNR** graphs.

## 📊 Key Results

The trained model achieved promising results:

- **Peak Accuracy:** >85% at 18dB SNR.
- **Overall Accuracy:** ~66% (across all SNR levels from -20dB to 18dB).
- **Observation:** The model successfully distinguishes digital modulations (like BPSK, GFSK) but shows expected confusion between QAM16 and QAM64 due to constellation similarities in noisy channels.

## 🛠️ Technologies

- **Language:** Python 3.11
- **Deep Learning:** TensorFlow, Keras
- **Data Science:** NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

## 📂 Dataset Information

This project uses the **RadioML2016.10a** dataset provided by **DeepSig Inc.**

Due to licensing and size constraints, the dataset is **not included** in this repository.

- **Source:** [DeepSig Datasets](https://www.deepsig.ai/datasets)
- **Reference:** T. J. O'Shea, N. West, "Radio Machine Learning Dataset Generation with GNU Radio", 2016.

## ⚙️ How to Run

1. Clone this repository.
2. Download `RML2016.10a_dict.pkl` from the source above and place it in the main folder.
3. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn
4.Run the training script:
```bash
   python train_model.py
   python evaluate_model.py




