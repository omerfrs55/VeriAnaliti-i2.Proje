# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

from modules.data_loader import DataLoader
from modules.stats_engine import StatsEngine
from modules.outlier_detector import OutlierDetector

app = Flask(__name__)

# --- GLOBAL DEĞİŞKENLER ---
loader = DataLoader()
data_loaded = False 
detector = None

def init_system():
    global data_loaded, loader, detector
    if not data_loaded:
        print("--- SİSTEM BAŞLATILIYOR ---")
        if loader.load_data():
            loader.preprocess_data()
            detector = OutlierDetector(loader.df_clean)
            detector.train_and_detect()
            data_loaded = True
            print("--- SİSTEM HAZIR ---")
        else:
            print("--- HATA: Veri Çekilemedi ---")

init_system()

# --- SAYFA 1: DASHBOARD ---
@app.route('/')
def dashboard():
    df_outliers = detector.df_outliers
    df_clean = detector.df_cleaned
    
    r_squared = detector.get_clean_metrics()
    opportunities = df_outliers[df_outliers['Durum'] == "FIRSAT"].sort_values(by='Z_Skoru').head(5)
    
    return render_template('dashboard.html', 
                           r_sq=round(r_squared, 3),
                           opportunities=opportunities.to_dict(orient='records'),
                           chart_data=df_clean[['Beygir Gücü', 'Fiyat']].values.tolist(),
                           outlier_data=opportunities[['Beygir Gücü', 'Fiyat']].values.tolist()
                           )

# --- SAYFA 2: ANALİZ ---
@app.route('/analysis')
def analysis():
    df = loader.df_clean
    stats = StatsEngine(df)
    
    corr_results = stats.calculate_correlations()
    cat_corr = stats.get_categorical_correlations(target_col='Fiyat')
    consistency = stats.compare_methods()
    
    # Isı Haritası Verisi
    corr_matrix = stats.calculate_correlations()['pearson']
    cols = [c for c in corr_matrix.columns if c != 'Fiyat'] + ['Fiyat']
    corr_matrix = corr_matrix[cols].reindex(cols)
    
    heatmap_data = {
        'z': corr_matrix.values.tolist(),
        'x': corr_matrix.columns.tolist(),
        'y': corr_matrix.columns.tolist()
    }
    
    influencers = stats.get_top_influencers(target_col='Fiyat')
    
    return render_template('analysis.html',
                           cat_corr=cat_corr,
                           consistency=consistency,
                           heatmap_data=heatmap_data,
                           influencers=influencers,
                           pearson_corr=corr_results['pearson'].to_html(classes='table table-sm table-bordered'),
                           spearman_corr=corr_results['spearman'].to_html(classes='table table-sm table-bordered'),
                           kendall_corr=corr_results['kendall'].to_html(classes='table table-sm table-bordered')
                           )

# --- SAYFA 3: HESAPLAMA ---
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    options = {
        'markalar': sorted(loader.df_clean['Marka'].unique()),
        'yakitlar': sorted(loader.df_clean['Yakıt Tipi'].unique()),
        'kasalar': sorted(loader.df_clean['Kasa Tipi'].unique()),
        'cekisler': sorted(loader.df_clean['Çekiş'].unique()),
        'beslemeler': sorted(loader.df_clean['Hava Besleme'].unique())
    }
    
    if request.method == 'POST':
        try:
            user_input = {
                'Marka': request.form['marka'],
                'Yakıt Tipi': request.form['yakit_tipi'],
                'Kasa Tipi': request.form['kasa_tipi'],
                'Çekiş': request.form['cekis'],
                'Hava Besleme': request.form['hava_besleme'],
                'Beygir Gücü': float(request.form['beygir']),
                'Motor Hacmi': float(request.form['motor']),
                'Otoyol Yakıt': float(request.form['yakit_tuketin']),
                'Boş Ağırlık': float(request.form['agirlik'])
            }
            
            predicted_price = detector.predict_single_car(user_input)
            
            prediction_result = {
                'tahmin': predicted_price,
                'marka': user_input['Marka'].upper()
            }
            
            if request.form.get('bulunan_fiyat'):
                found_price = float(request.form['bulunan_fiyat'])
                diff = found_price - predicted_price
                if diff < -2000:
                    prediction_result['analiz'] = "🔥 FIRSAT! (Piyasa altı)"
                    prediction_result['renk'] = "success"
                elif diff > 2000:
                    prediction_result['analiz'] = "⚠️ PAHALI! (Piyasa üstü)"
                    prediction_result['renk'] = "danger"
                else:
                    prediction_result['analiz'] = "✅ NORMAL Piyasa"
                    prediction_result['renk'] = "primary"
        
        except Exception as e:
            prediction_result = {'hata': f"Hata: {e}"}

    return render_template('predict.html', options=options, result=prediction_result)

# --- API ENDPOINTLERİ ---

@app.route('/api/get_models/<brand>')
def get_models(brand):
    df = loader.df_clean
    models = sorted(df[df['Marka'] == brand]['Model'].unique())
    return jsonify(models)

@app.route('/api/get_stats/<brand>/<model>')
def get_stats(brand, model):
    """
    GÜNCELLEME: Sadece HP ve Fiyat değil; Motor, Yakıt ve Ağırlık 
    için de doğrulama sınırlarını gönderiyoruz.
    """
    df = loader.df_clean
    if model == "Tümü":
        subset = df[df['Marka'] == brand]
    else:
        subset = df[(df['Marka'] == brand) & (df['Model'] == model)]
    
    if subset.empty:
        return jsonify({'error': 'Veri yok'})
    
    stats = {
        # Fiyat Bilgileri
        'fiyat_min': int(subset['Fiyat'].min()),
        'fiyat_max': int(subset['Fiyat'].max()),
        'fiyat_ort': int(subset['Fiyat'].mean()),
        
        # Beygir Bilgileri
        'hp_max': int(subset['Beygir Gücü'].max()),
        'hp_ort': int(subset['Beygir Gücü'].mean()),
        
        # Motor Hacmi Bilgileri
        'motor_max': int(subset['Motor Hacmi'].max()),
        'motor_ort': int(subset['Motor Hacmi'].mean()),
        
        # Yakıt (MPG) Bilgileri
        'yakit_max': int(subset['Otoyol Yakıt'].max()),
        'yakit_ort': int(subset['Otoyol Yakıt'].mean()),
        
        # Ağırlık Bilgileri
        'agirlik_max': int(subset['Boş Ağırlık'].max()),
        'agirlik_ort': int(subset['Boş Ağırlık'].mean())
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)