# modules/outlier_detector.py

import pandas as pd
import numpy as np

# Linear Regression yerine Random Forest kullanıyoruz.
# Bu model daha kararlıdır ve uçuk rakamlar (8.5e+30 gibi) üretmez.
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class OutlierDetector:
    """
    Random Forest kullanarak Fiyat Tahmini ve Fırsat Analizi yapar.
    Daha gerçekçi sonuçlar verir.
    """

    def __init__(self, df):
        self.df = df
        self.model = None
        self.df_outliers = None
        self.df_cleaned = None
        
        # Modelde kullanılacak özellikler
        self.features = [
            'Beygir Gücü', 'Motor Hacmi', 'Otoyol Yakıt', 'Boş Ağırlık',
            'Marka', 'Yakıt Tipi', 'Kasa Tipi', 'Çekiş', 'Hava Besleme'
        ]
        
        self.categorical_features = ['Marka', 'Yakıt Tipi', 'Kasa Tipi', 'Çekiş', 'Hava Besleme']

    def train_and_detect(self):
        if self.df is None: return None, None

        data = self.df.copy()
        y = data['Fiyat']
        X = data[self.features]

        # Pipeline Kurulumu
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='passthrough'
        )

        # GÜNCELLEME BURADA YAPILDI:
        # n_estimators=200 yaptık. (Eskiden 100'dü).
        # Bu, modelin arkada 200 farklı karar ağacı kurarak tahmin yapması demek.
        # Daha hassas ve sağlam sonuçlar verir.
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
        ])

        # Modeli Eğit
        self.model.fit(X, y)
        
        # Tahminleri Al
        predicted_prices = self.model.predict(X)

        # Residual ve Z-Score Hesabı
        residuals = y - predicted_prices
        std_dev = np.std(residuals)
        
        # Eğer standart sapma 0 ise (tek araç varsa) hata vermesin
        if std_dev == 0: std_dev = 1
            
        z_scores = residuals / std_dev
        
        data['Tahmini Değer'] = np.round(predicted_prices, 2)
        data['Fark'] = np.round(residuals, 2)
        data['Z_Skoru'] = np.round(z_scores, 2)

        # Etiketleme (Threshold = 1.5 Standart Sapma)
        threshold = 1.5 
        def label_status(z):
            if z < -threshold: return "FIRSAT"
            elif z > threshold: return "KAZIK"
            else: return "NORMAL"

        data['Durum'] = data['Z_Skoru'].apply(label_status)
        
        self.df_outliers = data.copy()
        self.df_cleaned = data[data['Durum'] == "NORMAL"].copy()

        return self.df_outliers, self.df_cleaned

    def get_clean_metrics(self):
        if self.df_cleaned is None: return 0
        X_clean = self.df_cleaned[self.features]
        y_clean = self.df_cleaned['Fiyat']
        return self.model.score(X_clean, y_clean)

    def predict_single_car(self, user_data):
        if self.model is None: return None
        input_df = pd.DataFrame([user_data])
        try:
            prediction = self.model.predict(input_df)
            return round(prediction[0], 2)
        except Exception as e:
            print(f"Tahmin Hatası: {e}")
            return 0