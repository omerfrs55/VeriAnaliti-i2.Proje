# modules/stats_engine.py

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

class StatsEngine:
    """
    İstatistiksel analizleri ve faktör sıralamasını yapan modül.
    GÜNCELLEME: Fiyatı etkileyen en önemli faktörleri bulma özelliği eklendi.
    """
    
    def __init__(self, dataframe):
        self.df = dataframe

    def calculate_correlations(self):
        """Sayısal veriler için 3 farklı korelasyon türünü hesaplar."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        results = {
            "pearson": numeric_df.corr(method='pearson'),
            "spearman": numeric_df.corr(method='spearman'),
            "kendall": numeric_df.corr(method='kendall')
        }
        return results

    # --- YENİ ÖZELLİK: EN ETKİLİ FAKTÖRLERİ SIRALAMA ---
    def get_top_influencers(self, target_col='Fiyat'):
        """
        Hedef değişkeni (Fiyat) en çok etkileyen faktörleri bulur.
        1. En çok artıranlar (Pozitif Korelasyon)
        2. En çok düşürenler (Negatif Korelasyon)
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Sadece Fiyat sütunuyla olan korelasyonları al (Kendisi hariç)
        correlations = numeric_df.corr(method='pearson')[target_col].drop(target_col)
        
        # Değerlere göre sırala
        sorted_corr = correlations.sort_values(ascending=False)
        
        # İlk 3 (Fiyatı Artıranlar)
        top_3_pos = sorted_corr.head(3).to_dict()
        
        # Son 3 (Fiyatı Düşürenler - Negatif Korelasyonun en güçlüleri)
        # ascending=True yaparak en küçükleri (en negatifleri) alıyoruz.
        top_3_neg = correlations.sort_values(ascending=True).head(3).to_dict()
        
        return {
            'pozitif': top_3_pos,
            'negatif': top_3_neg
        }

    def compare_methods(self):
        """Pearson ve Spearman tutarlılık analizi."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        p = numeric_df.corr(method='pearson')
        s = numeric_df.corr(method='spearman')
        
        diff = (p - s).abs().mean().mean()
        consistency_score = round((1 - diff) * 100, 2)
        diff_score = round(diff, 4)
        
        if consistency_score > 95:
            yorum = "Mükemmel Tutarlılık. Veri seti oldukça temiz."
            renk = "success"
        elif consistency_score > 85:
            yorum = "Yüksek Tutarlılık. Pearson ve Spearman benzer sonuçlar veriyor."
            renk = "primary"
        else:
            yorum = "Düşük Tutarlılık. Veri setinde güçlü aykırı değerler (outlier) mevcut."
            renk = "warning"
            
        return {'fark': diff_score, 'yuzde': consistency_score, 'yorum': yorum, 'renk': renk}

    def cramers_v(self, x, y):
        """Kategorik ilişki hesaplama (Kodun geri kalanı aynı)"""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        with np.errstate(divide='ignore', invalid='ignore'):
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            result = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        return result

    def get_categorical_correlations(self, target_col='Fiyat'):
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        results = {}
        temp_df = self.df.copy()
        if target_col in temp_df.columns:
             temp_df['Fiyat_Kategorisi'] = pd.qcut(temp_df[target_col], q=4, labels=['Ekonomik', 'Orta', 'Lüks', 'Premium'])
             target_variable = temp_df['Fiyat_Kategorisi']
        else:
             return {}
        for col in cat_cols:
            if col != 'Fiyat_Kategorisi': 
                score = self.cramers_v(temp_df[col], target_variable)
                results[col] = round(score, 4)
        return results