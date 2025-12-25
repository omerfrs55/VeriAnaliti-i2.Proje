# modules/data_loader.py

import pandas as pd
import numpy as np
import io
import requests

class DataLoader:
    """
    Veriyi çeken, temizleyen, model ayıklayan ve 
    sütun isimlerini TAM TÜRKÇE yapan modül.
    """
    
    def __init__(self):
        self.url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv"
        self.df = None
        self.df_clean = None

    def load_data(self):
        try:
            print(f"Bağlantı kuruluyor: {self.url}")
            response = requests.get(self.url)
            if response.status_code == 200:
                self.df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                print("Veri başarıyla çekildi.")
                return True
            else:
                return False
        except Exception as e:
            print(f"Hata: {e}")
            return False

    def preprocess_data(self):
        if self.df is None:
            return None

        data = self.df.copy()

        # 1. Gereksiz sütun atma
        if 'car_ID' in data.columns:
            data = data.drop(['car_ID'], axis=1)

        # 2. MARKA ve MODEL Ayrıştırma (Sahibinden Tarzı İçin)
        # Orijinal: 'alfa-romeo giulia'
        # Marka: 'alfa-romeo'
        # Model: 'giulia' (Geriye kalan her şey)
        
        # Önce Markayı alalım
        data['Marka'] = data['CarName'].apply(lambda x: x.split(' ')[0])
        
        # Marka düzeltmeleri
        corrections = {'maxda': 'mazda', 'porcshce': 'porsche', 'toyouta': 'toyota', 'vokswagen': 'volkswagen', 'vw': 'volkswagen'}
        data['Marka'] = data['Marka'].replace(corrections)
        
        # Şimdi 'Model' sütununu oluşturalım (Marka hariç ismin geri kalanı)
        # Örn: "audi 100 ls" -> Marka="audi", Model="100 ls"
        def get_model_name(full_name):
            parts = full_name.split(' ')
            if len(parts) > 1:
                return ' '.join(parts[1:]) # İlk kelime hariç gerisini birleştir
            else:
                return "Standart" # Eğer model adı yoksa
        
        data['Model'] = data['CarName'].apply(get_model_name)
        
        # Eski karmaşık ismi atıyoruz
        data = data.drop(['CarName'], axis=1)

        # 3. SÜTUN İSİMLERİNİ DÜZGÜN TÜRKÇE YAPMA
        # Artık boşluklu ve özel karakterli yazacağız, tabloda güzel dursun.
        rename_map = {
            'fueltype': 'Yakıt Tipi',
            'aspiration': 'Hava Besleme',
            'doornumber': 'Kapı Sayısı',
            'carbody': 'Kasa Tipi',
            'drivewheel': 'Çekiş',
            'enginelocation': 'Motor Yeri',
            'wheelbase': 'Dingil Mesafesi',
            'carlength': 'Uzunluk',
            'carwidth': 'Genişlik',
            'carheight': 'Yükseklik',
            'curbweight': 'Boş Ağırlık',
            'enginetype': 'Motor Tipi',
            'cylindernumber': 'Silindir Sayısı',
            'enginesize': 'Motor Hacmi',
            'fuelsystem': 'Yakıt Sistemi',
            'boreratio': 'Piston Çapı',
            'stroke': 'Piston Strok',
            'compressionratio': 'Sıkıştırma Oranı',
            'horsepower': 'Beygir Gücü',
            'peakrpm': 'Maksimum Devir',
            'citympg': 'Şehir İçi Yakıt',
            'highwaympg': 'Otoyol Yakıt',
            'price': 'Fiyat'
        }
        # Sütun 'brand' zaten yukarıda 'Marka' olmuştu, o yüzden map'e eklemedim.
        data = data.rename(columns=rename_map)

        # 4. İÇERİK TÜRKÇELEŞTİRME
        data['Yakıt Tipi'] = data['Yakıt Tipi'].map({'gas': 'Benzin', 'diesel': 'Dizel'})
        data['Kapı Sayısı'] = data['Kapı Sayısı'].map({'four': 4, 'two': 2})
        data['Kasa Tipi'] = data['Kasa Tipi'].replace({
            'convertible': 'Cabrio', 'hatchback': 'Hatchback', 
            'sedan': 'Sedan', 'wagon': 'Station Wagon', 'hardtop': 'Hardtop'
        })
        data['Çekiş'] = data['Çekiş'].replace({'rwd': 'Arkadan İtiş', 'fwd': 'Önden Çekiş', '4wd': '4x4'})
        data['Hava Besleme'] = data['Hava Besleme'].replace({'std': 'Atmosferik', 'turbo': 'Turbo'})

        self.df_clean = data
        return self.df_clean

    def get_numeric_data(self):
        if self.df_clean is not None:
            return self.df_clean.select_dtypes(include=[np.number])
        return None