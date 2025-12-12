import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 1. VERİYİ YÜKLEME
filename = 'RML2016.10a_dict.pkl'
print(f"Veri yükleniyor: {filename} ...")

try:
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        Xd = u.load()
except Exception as e:
    print(f"Hata: {e}")
    exit()

# 2. VERİYİ DÜZENLEME (PREPROCESSING)
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = [] 
lbl = []

print("Veriler birleştiriliyor...")
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))

X = np.vstack(X) # Veriyi (220000, 2, 128) formatına getir
print(f"Toplam Veri Şekli: {X.shape}")

# Etiketleri Ayırma (Sadece Modülasyon Türünü alıyoruz)
# lbl = [('QPSK', 10), ('BPSK', -20)...] -> sadece ['QPSK', 'BPSK'...] lazım
labels = np.array([x[0] for x in lbl])

# One-Hot Encoding (Bilgisayarın anlaması için: QPSK -> [0,0,1,0...])
print("Etiketler kodlanıyor (One-Hot Encoding)...")
lb = LabelBinarizer()
y = lb.fit_transform(labels)

# Veriyi Eğitim (%80) ve Test (%20) olarak bölme
print("Veri Eğitim ve Test olarak ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN için şekil değiştirme (Reshape): (N, 2, 128) -> (N, 2, 128, 1)
# Keras Conv2D katmanı 4 boyutlu veri ister (Batch, Height, Width, Channels)
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

print(f"Eğitim Verisi: {X_train.shape}")
print(f"Test Verisi: {X_test.shape}")

# 3. MODELİ KURMA (CNN Mimarisi)
# Bu mimari, bu iş için standart kabul edilen basit ama etkili bir yapıdır.
model = Sequential()

# Giriş Katmanı (Reshape gerekebilir demiştik ama veriyi zaten hazırladık)
# Input shape: (2, 128, 1)
model.add(Conv2D(64, (1, 3), activation='relu', input_shape=(2, 128, 1), padding='same'))
model.add(Dropout(0.5)) # Aşırı öğrenmeyi (Overfitting) engellemek için

model.add(Conv2D(16, (2, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))

model.add(Flatten()) # Düzleştirme (Matris -> Vektör)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(mods), activation='softmax')) # Çıkış katmanı (11 Sınıf)

# Modeli Derle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary() # Modelin özetini ekrana bas

# 4. EĞİTİMİ BAŞLAT (TRAINING)
# Epoch: Veri setinin üzerinden kaç kere geçileceği (Şimdilik 10 yeterli, sonra artırırız)
# Batch_size: Her adımda kaç veri işleneceği
print("\n--- EĞİTİM BAŞLIYOR (Bu işlem biraz sürebilir) ---")
history = model.fit(
    X_train, y_train,
    batch_size=1024,
    epochs=15,
    verbose=1,
    validation_data=(X_test, y_test)
)

# 5. SONUCU KAYDETME
print("Model kaydediliyor...")
model.save("amc_model_v1.h5") # Modeli dosyaya kaydet (Sonra tekrar kullanmak için)
print("BAŞARILI! Model 'amc_model_v1.h5' olarak kaydedildi.")

# Başarıyı göster
score = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Doğruluğu (Accuracy): %{score[1]*100:.2f}")