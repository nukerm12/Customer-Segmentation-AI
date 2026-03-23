import pandas as pd

# Dosya yolunu belirt
file_path = "marketing_campaign.csv"

# Noktalı virgülü ayırıcı olarak kullan
not_balanced_data = pd.read_csv(file_path, sep=';')
print(not_balanced_data['Response'].value_counts())
from sklearn.utils import resample

# Azınlık ve çoğunluk sınıflarını ayır
majority = not_balanced_data[not_balanced_data['Response'] == 0]
minority = not_balanced_data[not_balanced_data['Response'] == 1]

# Çoğunluk sınıfından örnek azalt (undersampling)
majority_downsampled = resample(majority,
                                replace=False,
                                n_samples=len(minority),
                                random_state=42)

# Azınlık ve dengeye getirilmiş çoğunluk sınıflarını birleştir
data = pd.concat([majority_downsampled, minority])
print(data['Response'].value_counts())
# İlk birkaç satırı ve veri seti bilgilerini kontrol et
print(data.head())
print(data.info())
# Eksik değerlerin kontrolü
print(data.isnull().sum())
# Education sütunundaki kategorik değerleri dönüştür
education_mapping = {
    "Basic": 0,          # En düşük seviye
    "2n Cycle": 1,       # İkinci kademe
    "Graduation": 2,     # Lisans
    "Master": 3,         # Yüksek lisans
    "PhD": 4             # Doktora
}

# Mapping işlemini uygulayın
data['Education'] = data['Education'].map(education_mapping)

# Dönüştürme işlemini kontrol edin
print(data['Education'].value_counts())
# Marital_Status sütunundaki kategorik değerleri dönüştür
status_mapping = {
    "Married": 1,       # Evli olanları 1 yap
    "Together": 1,      # Birlikte olanları da 1 yap
    "Single": 0,        # Bekar olanları 0 yap
    "Divorced": 0,      # Boşanmış olanları 0 yap
    "Widow": 0,         # Dul olanları 0 yap
    "Alone": 0,         # Yalnız olanları 0 yap
    "Absurd": 0,        # Geçersiz olanları 0 yap
    "YOLO": 0           # "YOLO" gibi geçersiz verileri de 0 yap
}

data['Marital_Status'] = data['Marital_Status'].map(status_mapping)

# Dönüştürme işlemini kontrol edin
print(data['Marital_Status'].value_counts())

# Eksik verileri doldur veya sil
data['Income'] = data['Income'].fillna(data['Income'].median())
data['Marital_Status'] = data['Marital_Status'].fillna(data['Marital_Status'].mode()[0])# Eksik gelirleri medyan ile doldur
data = data.dropna()  # Tüm eksik verileri içeren satırları sil
from datetime import datetime

# Yaş hesapla (2024 yılı örneği)
data['Age'] = 2024 - data['Year_Birth']

# Mantıksız yaşları kaldır (örneğin, yaş > 100)
data = data[data['Age'] < 100]
# Bins ve Labels tanımları
bins = [30, 35, 40, 45, 50, 55, 60, 65, float('inf')]  # Yaş grupları sınırları
labels = ['30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65+']  # Gruplara verilecek isimler

# pd.cut ile yaş gruplarını oluşturma
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

# Yeni sütunu kontrol etmek için ilk birkaç satırı görüntüleyin
print(data[['Age', 'Age_Group']].head())

# Grupların dağılımını kontrol etmek için (opsiyonel)
print(data['Age_Group'].value_counts())
# Tüm ürün kategorilerindeki harcamaların toplamı
data['TotalSpent'] = data[['MntFishProducts', 'MntMeatProducts', 'MntSweetProducts',
                           'MntWines', 'MntFruits', 'MntGoldProds']].sum(axis=1)
print(data[['TotalSpent']].head())
import seaborn as sns
import matplotlib.pyplot as plt

# Kampanya yanıt oranlarını görselleştir
sns.countplot(x='Response', data=not_balanced_data)
plt.title('Kampanya Yanıt Dağılımı')
plt.close()
sns.countplot(x='Response', data=data)
plt.title('Kampanya Yanıt Dağılımı')
plt.close()

# Yanıt verenlerin (Response = 1) verilerini filtrele
response_yes = data[data['Response'] == 1]

# Gelir, yaş ve toplam harcama verilerini al
response_yes_income = response_yes['Income']
response_yes_age = response_yes['Age']
response_yes_spent = response_yes['TotalSpent']

# Grafik alanı oluştur (3 alt grafik, 1 satır ve 3 sütun)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Grafik: Gelir Dağılımı
sns.histplot(response_yes_income, bins=20, kde=True, ax=axes[0])
axes[0].set_title('Gelir Dağılımı')
axes[0].set_xlabel('Gelir')
axes[0].set_ylabel('Frekans')

# 2. Grafik: Yaş Dağılımı
sns.histplot(response_yes_age, bins=20, kde=True, ax=axes[1])
axes[1].set_title('Yaş Dağılımı')
axes[1].set_xlabel('Yaş')
axes[1].set_ylabel('Frekans')

# 3. Grafik: Toplam Harcama Dağılımı
sns.histplot(response_yes_spent, bins=20, kde=True, ax=axes[2])
axes[2].set_title('Toplam Harcama Dağılımı')
axes[2].set_xlabel('Toplam Harcama')
axes[2].set_ylabel('Frekans')

# Figürü göster
plt.tight_layout()
plt.close()
X = data.drop(columns=['Response'])  # Hedef değişken hariç diğer sütunlar
y = data['Response']  # Hedef değişken
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Sadece sayısal sütunları seç
import numpy as np
numeric_columns = X_train.select_dtypes(include=[np.number]).columns
X_train = X_train[numeric_columns]
X_test = X_test[numeric_columns]
# Sabit sütunları bul ve çıkar
constant_columns = [col for col in X_train.columns if X_train[col].nunique() <= 1]
if constant_columns:
    print(f"Sabit sütunlar kaldırılıyor: {constant_columns}")
    X_train = X_train.drop(columns=constant_columns)
    X_test = X_test.drop(columns=constant_columns)
# Spearman korelasyonu ile özellik seçimi
from scipy.stats import spearmanr

# Eğitim setinde hedef değişken ile özelliklerin korelasyonunu hesapla
correlations = {}
for column in X_train.columns:
    corr, _ = spearmanr(X_train[column], y_train)
    correlations[column] = corr

# Belirli bir eşik değeri üzerinde korelasyona sahip özellikleri seç
correlation_threshold = 0.2  # Eşik değeri
selected_features = [k for k, v in correlations.items() if abs(v) > correlation_threshold]

# Sadece seçilen özellikleri kullan
X_train = X_train[selected_features]
X_test = X_test[selected_features]

import numpy as np
# Numerik sütunları belirle
numerik_sütunlar = [col for col in data.select_dtypes(include=[np.number]).columns]
numerik_veri = data[numerik_sütunlar]
# Numerik veriyi kontrol edin
print(numerik_veri.head())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerik_veri_scaled = scaler.fit_transform(numerik_veri)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Modeli oluştur
model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)

# Modeli eğit
model.fit(X_train_scaled, y_train)

# Test seti üzerinde tahmin yap
y_pred = model.predict(X_test_scaled)

# Performansı değerlendirme
print(f"Model Doğruluk Oranı: {accuracy_score(y_test, y_pred):.2f}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi')
plt.close()
print(f"Sütun isimleri sayısı: {len(X.columns)}")
print(f"Model katsayıları sayısı: {len(model.coef_[0])}")
# Standartlaştırmadan önce sütun isimlerini kaydedin
feature_names = X_train.columns

# Standartlaştırma işlemleri
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.coef_[0]
}).sort_values(by='Importance', ascending=False)

print(feature_importance)
print("Tüm sütun isimleri:")
print(feature_names)

print("Model katsayılarının sayısı:")
print(len(model.coef_[0]))
# Model katsayılarının ve sütun isimlerinin uzunluklarını kontrol et
print(f"Model katsayılarının sayısı: {len(model.coef_[0])}")
print(f"Sütun isimlerinin sayısı: {len(feature_names)}")

# Katsayı ve sütun isimlerini eşleştir
if len(feature_names) != len(model.coef_[0]):
    print("Eksik veya fazla sütunlar olabilir.")
    print(f"Sütun isimleri: {len(feature_names)}")
    print(f"Model katsayıları: {len(model.coef_[0])}")

    # Fazla sütunları bul
    if len(feature_names) > len(model.coef_[0]):
        print("Fazla sütunlar:")
        extra_columns = feature_names[len(model.coef_[0]):]
        print(extra_columns)
        # Fazla sütunları çıkar
        feature_names = feature_names[:len(model.coef_[0])]
        print(f"Güncellenmiş sütun isimleri: {feature_names}")

    # Eksik sütunları bul
    elif len(feature_names) < len(model.coef_[0]):
        print("Eksik sütunlar:")
        missing_columns = model.coef_[0][len(feature_names):]
        print(missing_columns)
        # Eksik sütunları tamamlayın (isteğe bağlı)
        print("Eksik sütunları analiz edin ve yeniden kontrol edin.")

# Özellik etkisi analizi
import pandas as pd
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.coef_[0]
}).sort_values(by='Importance', ascending=False)
# Özellik önem sıralamasını görselleştirin
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Özelliklerin Önem Sıralaması')
plt.xlabel('Önem Derecesi')
plt.ylabel('Özellikler')
plt.close()
# Yaş aralıklarını tanımla
age_ranges = [(32, 33), (33, 34), (34, 35), (35, 36),
              (36, 37), (37, 38), (38, 39), (39, 40), (41, 42), (42, 43),
              (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49),
              (49, 50), (50, 51), (51, 52), (53, 54), (54, 55), (55, 56),
              (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62),
              (62, 63), (63, 64), (64, 65), (65, float('inf'))]

# Tüm yaş aralıkları için analiz yap
importance_results = []

for age_range in age_ranges:
    # 1. Belirtilen yaş aralığına göre veriyi filtrele
    filtered_data = data[(data['Age'] >= age_range[0]) & (data['Age'] < age_range[1])]

    # Verinin boş olup olmadığını kontrol et
    if filtered_data.empty:
        print(f"Yaş aralığı {age_range} için veri bulunamadı.")
        continue

    # 2. Veri setini hazırlama
    X_filtered = filtered_data.drop(columns=['Response', 'Age_Group'])  # 'Response' hedef değişken
    y_filtered = filtered_data['Response']

    # Sadece sayısal sütunları seç
    X_filtered = X_filtered.select_dtypes(include=[np.number])

    # Veri kontrolü
    if X_filtered.empty or y_filtered.nunique() <= 1:
        print(f"{age_range} yaş aralığında yeterli veri yok.")
        continue

    # 3. Veriyi ölçeklendirme
    scaler = StandardScaler()
    X_filtered_scaled = scaler.fit_transform(X_filtered)

    # 4. Modeli eğitme
    model_filtered = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=42)
    try:
        model_filtered.fit(X_filtered_scaled, y_filtered)
    except Exception as e:
        print(f"Model eğitimi sırasında hata oluştu: {e}")
        continue

    # 5. Özellik önemlerini çıkarma
    if hasattr(model_filtered, 'coef_'):
        feature_importance_filtered = pd.DataFrame({
            'Feature': X_filtered.columns,
            'Importance': model_filtered.coef_[0]
        }).sort_values(by='Importance', ascending=False)
    else:
        print(f"Model, {age_range} yaş aralığı için eğitilemedi.")
        continue

    # Görselleştirme
    if not feature_importance_filtered.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_filtered)
        plt.title(f'Yaş Aralığı {age_range[0]}-{age_range[1]} Özelliklerin Önem Sıralaması')
        plt.xlabel('Önem Derecesi')
        plt.ylabel('Özellikler')
        plt.close()

# Tüm yaş aralıkları için sonuçları birleştir
if importance_results:
    # Boş DataFrame'leri filtrele
    filtered_results = [df for df in importance_results if not df.empty]
    if filtered_results:
        all_importances = pd.concat(filtered_results, ignore_index=True)
    else:
        print("Tüm DataFrame'ler boş, birleştirme işlemi yapılamıyor.")
else:
    print("importance_results listesi boş.")
# Genel model eğitimi (tüm yaş grupları için),
["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd","#8c564b"]
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def train_general_model(data):
    X = data.drop(columns=['Response', 'Age_Group'])  # Hedef değişken 'Response' hariç diğer sütunlar
    y = data['Response']

    # Sadece sayısal sütunları seç
    X = X.select_dtypes(include=[np.number])

    # Veriyi ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model eğitimi
    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    model.fit(X_scaled, y)

    # Özellik önemlerini çıkarma
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.coef_[0]
    }).sort_values(by='Importance', ascending=False)

    return feature_importance

def feature_importance_specific_age_range_with_tolerance(data, start_age, end_age, tolerance=5):
    """
    Girilen yaş aralığı için özellik önem sıralaması yapar, veri eksikse toleranslı analiz yapar.
    :param data: Veri seti
    :param start_age: Yaş aralığının başlangıcı
    :param end_age: Yaş aralığının sonu
    :param tolerance: Tolerans (varsayılan: 1)
    """
    # Yaş aralığını ve toleransı göz önünde bulundur
    filtered_data = data[
        ((data['Age'] >= start_age - tolerance) & (data['Age'] <= end_age + tolerance))
    ]

    # Eğer veri hala boşsa tahmini modele geç
    if filtered_data.empty:
        print(f"Yaş aralığı {start_age}-{end_age} (±{tolerance}) için yeterli veri bulunamadı! Genel model kullanılacak.")
        return train_general_model(data)

    # Hedef değişken ve özellikler
    X_filtered = filtered_data.drop(columns=['Response', 'Age_Group'])
    y_filtered = filtered_data['Response']

    # Sadece sayısal sütunları seç
    X_filtered = X_filtered.select_dtypes(include=[np.number])

    # Veriyi ölçeklendirme
    scaler = StandardScaler()
    X_filtered_scaled = scaler.fit_transform(X_filtered)

    # Model eğitme
    model_filtered = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    model_filtered.fit(X_filtered_scaled, y_filtered)

    # Özellik önemlerini çıkarma
    feature_importance_filtered = pd.DataFrame({
        'Feature': X_filtered.columns,
        'Importance': model_filtered.coef_[0]
    }).sort_values(by='Importance', ascending=False)

    # Özellik önemlerini görselleştirme
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=pd.concat([feature_importance_filtered.nlargest(3, 'Importance'),
                        feature_importance_filtered.nsmallest(3, 'Importance')]),
        palette='coolwarm'
    )
    plt.title(f"Yaş Aralığı ( {start_age}-{end_age} ): En Önemli ve En Önemsiz Özellikler")
    plt.xlabel("Özellik Önem Derecesi")
    plt.ylabel("Özellikler")
    plt.tight_layout()
    plt.show()

    return feature_importance_filtered


def feature_importance_ai_with_tolerance(data, tolerance=5):
    """
    Kullanıcı girdisine göre yaş aralığına özel analiz yapar, toleranslı tahmin ekler.
    :param data: Veri seti
    :param tolerance: Tolerans (varsayılan: 1)
    """
    while True:
        try:
            # Kullanıcıdan yaş aralığı girişini al
            start_age = int(input("Başlangıç yaşını girin (Çıkmak için -1 yazın): "))
            if start_age == -1:
                print("Programdan çıkılıyor. Görüşmek üzere!")
                break

            end_age = int(input("Bitiş yaşını girin: "))
            if end_age < start_age:
                print("Bitiş yaşı, başlangıç yaşından küçük olamaz. Tekrar deneyin.")
                continue
        except ValueError:
            print("Lütfen geçerli bir sayı girin!")
            continue

        # Yaş aralığı analizi yap
        specific_importance = feature_importance_specific_age_range_with_tolerance(
            data, start_age, end_age, tolerance
        )

        if specific_importance is not None:
            print(f"\nGirilen yaş aralığı: {start_age}-{end_age} ")
            print("En Önemli 3 Özellik:")
            print(specific_importance.nlargest(3, 'Importance'))
            print("\nEn Önemsiz 3 Özellik:")
            print(specific_importance.nsmallest(3, 'Importance'))


# Analizi başlat
feature_importance_ai_with_tolerance(data, tolerance=5)
