# app.py

from flask import Flask, render_template, request
from modules.data_loader import DataLoader
from modules.stats_engine import StatsEngine
from modules.outlier_detector import OutlierDetector

app = Flask(__name__)

# --- GLOBAL DEĞİŞKENLER ---
loader = DataLoader()
data_loaded = False 
detector = None

# --- SİSTEMİ BAŞLATAN FONKSİYON ---
def init_system():
    global data_loaded, loader, detector
    
    if not data_loaded:
        print("--- SİSTEM BAŞLATILIYOR: Veri Hazırlanıyor ---")
        
        # 1. Veriyi Çek
        success = loader.load_data()
        
        if success:
            # 2. Veriyi Temizle ve Türkçeleştir
            loader.preprocess_data()
            
            # 3. Modeli Eğit (Yeni Türkçe sütunlarla)
            detector = OutlierDetector(loader.df_clean)
            detector.train_and_detect()
            
            data_loaded = True
            print("--- SİSTEM HAZIR: Tüm analizler tamamlandı ---")
        else:
            print("--- HATA: Veri çekilemedi ---")

# Uygulamayı başlat
init_system()

# --- 1. DASHBOARD (ANA SAYFA) ---
@app.route('/')
def dashboard():
    df_outliers = detector.df_outliers
    df_clean = detector.df_cleaned
    r_squared = detector.get_clean_metrics()
    
    # Fırsat Araçları (Z-Score < -1.5)
    opportunities = df_outliers[df_outliers['Durum'] == "FIRSAT"].sort_values(by='Z_Skoru').head(5)
    
    return render_template('dashboard.html', 
                           r_sq=round(r_squared, 3),
                           opportunities=opportunities.to_dict(orient='records'),
                           chart_data=df_clean[['Beygir Gücü', 'Fiyat']].values.tolist(),
                           outlier_data=opportunities[['Beygir Gücü', 'Fiyat']].values.tolist()
                           )

# --- 2. DETAYLI ANALİZ (AKADEMİK) ---
@app.route('/analysis')
def analysis():
    df = loader.df_clean
    stats = StatsEngine(df)
    
    # İstatistikler (Fiyat ve Beygir Gücü gibi sayısal veriler üzerinden)
    corr_results = stats.calculate_correlations()
    cat_corr = stats.get_categorical_correlations(target_col='Fiyat')
    
    # Yeni Detaylı Karşılaştırma Raporu
    consistency = stats.compare_methods()
    
    return render_template('analysis.html',
                           cat_corr=cat_corr,
                           consistency=consistency,
                           pearson_corr=corr_results['pearson'].to_html(classes='table table-sm table-bordered'),
                           spearman_corr=corr_results['spearman'].to_html(classes='table table-sm table-bordered'),
                           kendall_corr=corr_results['kendall'].to_html(classes='table table-sm table-bordered')
                           )

# --- 3. HESAPLAMA VE TAHMİN (DETAYLI SEÇİM) ---
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    
    # Dropdown (Açılır Menü) İçin Seçenekleri Hazırla
    # unique() ile benzersiz değerleri alıp sorted() ile alfabetik sıralıyoruz.
    options = {
        'markalar': sorted(loader.df_clean['Marka'].unique()),
        'yakitlar': sorted(loader.df_clean['Yakıt Tipi'].unique()),
        'kasalar': sorted(loader.df_clean['Kasa Tipi'].unique()),
        'cekisler': sorted(loader.df_clean['Çekiş'].unique()),
        'beslemeler': sorted(loader.df_clean['Hava Besleme'].unique())
    }
    
    if request.method == 'POST':
        try:
            # Formdan gelen verileri al (İsimler HTML'deki name="" ile aynı olmalı)
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
            
            # Tahmin yap
            predicted_price = detector.predict_single_car(user_input)
            
            prediction_result = {
                'tahmin': predicted_price,
                'marka': user_input['Marka'].upper()
            }
            
            # Fırsat Analizi (Varsa)
            if request.form.get('bulunan_fiyat'):
                found_price = float(request.form['bulunan_fiyat'])
                diff = found_price - predicted_price
                
                if diff < -2000:
                    prediction_result['analiz'] = "🔥 FIRSAT! (Piyasa değerinin altında)"
                    prediction_result['renk'] = "success"
                elif diff > 2000:
                    prediction_result['analiz'] = "⚠️ PAHALI! (Piyasa değerinin üzerinde)"
                    prediction_result['renk'] = "danger"
                else:
                    prediction_result['analiz'] = "✅ NORMAL (Piyasa değerinde)"
                    prediction_result['renk'] = "primary"
                    
        except Exception as e:
            prediction_result = {'hata': f"Hata: {e}"}

    # options sözlüğünü de sayfaya gönderiyoruz (dropdownları doldurmak için)
    return render_template('predict.html', options=options, result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)