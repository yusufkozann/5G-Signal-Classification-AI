import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns # Bunu yüklememiz gerekebilir, aşağıda anlatacağım

# 1. VERİYİ VE MODELİ YÜKLE
print("Veri ve Model yükleniyor...")
filename = 'RML2016.10a_dict.pkl'
with open(filename, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    Xd = u.load()

# Eğittiğimiz modeli geri çağırıyoruz
model = load_model('amc_model_v1.h5')

# Parametreleri alalım
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

# --- GRAFİK 1: SNR'a Göre Doğruluk (Accuracy vs SNR) ---
print("SNR analizi yapılıyor...")
acc_scores = []

for snr in snrs:
    # Sadece o SNR değerine ait verileri topla
    test_X = []
    test_Y = []
    
    for mod in mods:
        # Veriyi çek
        t_data = Xd[(mod, snr)]
        test_X.extend(t_data)
        
        # Etiketi oluştur (One-Hot değil, direkt sayı olarak: 0, 1, 2...)
        # Çünkü confusion matrix için düz numara lazım
        test_Y.extend([mods.index(mod)] * t_data.shape[0])
        
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    
    # Modelin anlaması için şekli düzelt: (N, 2, 128, 1)
    test_X = np.expand_dims(test_X, axis=3)
    
    # Tahmin yap
    # Model çıktıları olasılık verir (Softmax), en yüksek olasılığın indexini alıyoruz (argmax)
    preds = model.predict(test_X, verbose=0)
    y_pred_label = np.argmax(preds, axis=1)
    
    # Doğruluğu hesapla
    # Doğru tahmin sayısı / Toplam veri
    accuracy = np.mean(y_pred_label == test_Y)
    acc_scores.append(accuracy)
    print(f"SNR: {snr}dB -> Accuracy: %{accuracy*100:.2f}")

# Grafiği Çiz
plt.style.use('ggplot') # Güzel bir tema
plt.figure(figsize=(10, 6))
plt.plot(snrs, acc_scores, marker='o', linewidth=2, color='blue')
plt.title("Sinyal Kalitesine (SNR) Göre Model Başarısı")
plt.xlabel("Sinyal-Gürültü Oranı (SNR) [dB]")
plt.ylabel("Doğruluk (Accuracy)")
plt.grid(True)
plt.savefig("sonuc_snr_grafigi.png")
print("Grafik 1 Kaydedildi: sonuc_snr_grafigi.png")

# --- GRAFİK 2: Confusion Matrix (Sadece Yüksek SNR için) ---
# Düşük SNR'da her şey karışır, o yüzden sadece 18dB'ye bakalım, modelin gerçek zekasını görelim.
print("\nConfusion Matrix (18dB) oluşturuluyor...")

target_snr = 18
test_X_cm = []
test_Y_cm = []

for mod in mods:
    t_data = Xd[(mod, target_snr)]
    test_X_cm.extend(t_data)
    test_Y_cm.extend([mods.index(mod)] * t_data.shape[0])

test_X_cm = np.expand_dims(np.array(test_X_cm), axis=3)
test_Y_cm = np.array(test_Y_cm)

preds = model.predict(test_X_cm, verbose=0)
y_pred_cm = np.argmax(preds, axis=1)

# Matrisi hesapla
cm = confusion_matrix(test_Y_cm, y_pred_cm)
# Normalize et (Yüzdelik olarak görmek daha iyidir)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Çiz
plt.figure(figsize=(12, 10))
# Seaborn kütüphanesi ile ısı haritası çiziyoruz
try:
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=mods, yticklabels=mods)
except:
    # Seaborn yoksa basit çizim
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

plt.title(f"Confusion Matrix (SNR={target_snr}dB)")
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Tahmin Edilen Sınıf')
plt.savefig("sonuc_confusion_matrix.png")
print("Grafik 2 Kaydedildi: sonuc_confusion_matrix.png")
plt.show()