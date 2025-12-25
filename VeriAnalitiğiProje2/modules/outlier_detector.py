# modules/outlier_detector.py

# Veri işleme kütüphaneleri
import pandas as pd
import numpy as np

# Makine Öğrenmesi (Scikit-Learn) kütüphaneleri
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class OutlierDetector:
    """
    Bu sınıf;
    1. Çoklu Doğrusal Regresyon modeli kurar.
    2. Z-Score yöntemiyle Fırsat/Kazık araçları tespit eder.
    3. Kullanıcıdan gelen DETAYLI araç verisi için fiyat tahmini yapar.
    """

    def __init__(self, df):
        # Analiz edilecek ana veri seti
        self.df = df
        
        # Eğitilmiş modeli saklayacağımız değişken
        self.model = None
        
        # Sonuçları saklayacağımız değişkenler
        self.df_outliers = None # (Fırsat/Kazık etiketli tam liste)
        self.df_cleaned = None  # (Outlier'lardan arındırılmış temiz liste)
        
        # MODELDE KULLANILACAK ÖZELLİKLER (Features)
        # Sütun isimleri artık TAM TÜRKÇE ve boşluklu.
        # "Sahibinden" tarzı detaylı analiz için kategorik özellikleri artırdık.
        self.features = [
            # Sayısal Özellikler
            'Beygir Gücü',
            'Motor Hacmi',
            'Otoyol Yakıt',
            'Boş Ağırlık',
            
            # Kategorik Özellikler (Bunlar fiyata çok etki eder)
            'Marka',
            'Yakıt Tipi',      # Benzin / Dizel
            'Kasa Tipi',       # Sedan / Hatchback
            'Çekiş',           # Önden / Arkadan / 4x4
            'Hava Besleme'     # Turbo / Atmosferik
        ]
        
        # One-Hot Encoding yapılacak (Sayıya çevrilecek) kategorik sütunlar
        self.categorical_features = ['Marka', 'Yakıt Tipi', 'Kasa Tipi', 'Çekiş', 'Hava Besleme']

    def train_and_detect(self):
        """
        Modeli eğitir ve veri setindeki her araç için 
        'Bu fiyata değer mi?' analizi yapar.
        """
        # Eğer veri yoksa işlem yapma
        if self.df is None:
            return None, None

        # Verinin kopyasını alıyoruz
        data = self.df.copy()
        
        # Hedef Değişken (Y) -> Fiyat
        y = data['Fiyat']
        
        # Bağımsız Değişkenler (X) -> Seçtiğimiz özellikler
        X = data[self.features]

        # --- 1. MODEL HAZIRLIĞI (PIPELINE) ---
        
        # Kategorik verileri sayıya çevirme işlemi
        # handle_unknown='ignore': Eğitimde görmediği bir seçenek gelirse hata verme.
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='passthrough' # Sayısal sütunları olduğu gibi geçir
        )

        # Pipeline kuruyoruz: Önce Çevir -> Sonra Eğit
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # --- 2. MODEL EĞİTİMİ (FITTING) ---
        self.model.fit(X, y)
        
        # --- 3. TAHMİN VE ANALİZ ---
        predicted_prices = self.model.predict(X)

        # Residual (Hata) = Gerçek Fiyat - Tahmin
        residuals = y - predicted_prices
        
        # Standart Sapma ve Z-Skoru Hesaplama
        std_dev = np.std(residuals)
        z_scores = residuals / std_dev
        
        # Sonuçları tabloya ekleyelim
        data['Tahmini Değer'] = np.round(predicted_prices, 2)
        data['Fark'] = np.round(residuals, 2)
        data['Z_Skoru'] = np.round(z_scores, 2)

        # --- 4. ETİKETLEME (FIRSAT / KAZIK) ---
        threshold = 1.5
        
        def label_status(z):
            if z < -threshold:
                return "FIRSAT" # Tahminden çok daha ucuz
            elif z > threshold:
                return "KAZIK"  # Tahminden çok daha pahalı
            else:
                return "NORMAL"

        data['Durum'] = data['Z_Skoru'].apply(label_status)
        
        # 5. Çıktıları Hazırla
        self.df_outliers = data.copy()
        self.df_cleaned = data[data['Durum'] == "NORMAL"].copy()

        return self.df_outliers, self.df_cleaned

    def get_clean_metrics(self):
        """Modelin başarısını (R-Kare) temiz veri üzerinden ölçer."""
        if self.df_cleaned is None:
            return 0
        X_clean = self.df_cleaned[self.features]
        y_clean = self.df_cleaned['Fiyat']
        return self.model.score(X_clean, y_clean)

    def predict_single_car(self, user_data):
        """
        Kullanıcının formdan girdiği DETAYLI verileri alır ve tahmin yapar.
        user_data sözlüğü artık daha fazla detay içermeli.
        """
        if self.model is None:
            return None
            
        # Gelen sözlüğü DataFrame'e çevir
        input_df = pd.DataFrame([user_data])
        
        # Modeli kullanarak tahmin yap
        try:
            prediction = self.model.predict(input_df)
            return round(prediction[0], 2)
        except Exception as e:
            print(f"Tahmin Hatası: {e}")
            return 0