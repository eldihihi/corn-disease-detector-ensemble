# app.py

import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

# Impor fungsi dan variabel global dari modul utilitas
from utils.model_loader import load_models_in_background, ENSEMBLE_MODELS, MODELS_LOADED, MODEL_LOAD_ERROR
from utils.image_processor import preprocess_image # Memastikan ini diimpor untuk preprocessing

# Menginisialisasi aplikasi Flask
app = Flask(__name__)

# Mengatur secret key untuk sesi dan flash messages (SANGAT PENTING!)
app.secret_key = os.urandom(24)

# Konfigurasi Upload Folder
# Menyimpan gambar yang diunggah di dalam direktori 'static/uploads'
# agar bisa diakses oleh browser melalui URL
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Batas ukuran file 16MB

# Tipe file yang diizinkan untuk diunggah
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Kelas-kelas penyakit (GANTI INI SESUAI DENGAN MODEL ANDA)
# Pastikan urutan ini sesuai dengan output prediksi model Anda
CLASS_NAMES = [
    'Blight',
    'Common_Rust',
    'Gray_Leaf_Spot',
    'Healthy'
    # Tambahkan semua kelas penyakit yang model Anda prediksi
]

# Fungsi untuk memeriksa apakah ekstensi file diizinkan
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========== FLASK ROUTES ==========

# Route untuk halaman utama
@app.route('/')
def index():
    # Menampilkan pesan berdasarkan status pemuatan model
    if MODEL_LOAD_ERROR:
        flash("Kesalahan fatal saat startup model. Periksa log Railway.", 'error')
    elif not MODELS_LOADED:
        flash("Model sedang dimuat (di latar belakang), mohon tunggu sebentar...", 'info')
    
    # Render template index.html
    return render_template('index.html')

# Route untuk menangani unggahan file dan prediksi
@app.route('/upload', methods=['POST'])
def upload_file():
    # Memastikan metode request adalah POST
    if request.method == 'POST':
        # Memeriksa apakah ada file di request
        if 'file' not in request.files:
            flash('Tidak ada bagian file', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Memeriksa apakah nama file kosong
        if file.filename == '':
            flash('Tidak ada file terpilih', 'error')
            return redirect(request.url)
        
        # Memeriksa apakah file ada dan ekstensinya diizinkan
        if file and allowed_file(file.filename):
            # Memeriksa status pemuatan model sebelum melakukan prediksi
            if MODEL_LOAD_ERROR:
                flash(f"Tidak dapat melakukan prediksi: {MODEL_LOAD_ERROR}", 'error')
                return redirect(url_for('index'))
            if not MODELS_LOADED or len(ENSEMBLE_MODELS) < 3:
                flash("Model belum sepenuhnya dimuat. Mohon tunggu beberapa saat dan coba lagi.", 'info')
                return redirect(url_for('index'))

            # Menyimpan file yang diunggah dengan nama yang aman
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Inisialisasi daftar prediksi dari setiap model
                predictions = []
                
                # Melakukan prediksi dengan setiap model dalam ensemble
                for model_name, model in ENSEMBLE_MODELS.items():
                    # Preprocess gambar sesuai dengan model yang sedang digunakan
                    # PENTING: Pastikan ukuran input di image_processor.py sesuai
                    # (224x224 untuk ResNet/VGG, 299x299 untuk Inception)
                    processed_image = preprocess_image(filepath, model_name)
                    
                    # Lakukan prediksi
                    pred = model.predict(processed_image)
                    predictions.append(pred)

                # Ensemble Averaging: Rata-ratakan prediksi dari semua model
                ensemble_prediction = np.mean(predictions, axis=0)
                
                # Mendapatkan kelas yang diprediksi dan tingkat keyakinan
                predicted_class_index = np.argmax(ensemble_prediction)
                result = CLASS_NAMES[predicted_class_index]
                confidence = np.max(ensemble_prediction) * 100 # Konversi ke persen

                # Menampilkan hasil prediksi ke pengguna
                flash(f"Deteksi: {result} (Keyakinan: {confidence:.2f}%)", 'success')
                return render_template('index.html',
                                       result=result,
                                       confidence=f"{confidence:.2f}%",
                                       uploaded_image_url=url_for('static', filename='uploads/' + filename))
            
            except Exception as e:
                # Tangani kesalahan yang terjadi selama proses prediksi
                flash(f"Terjadi kesalahan saat memproses gambar atau melakukan prediksi: {e}", 'error')
                print(f"ERROR: Prediksi gagal karena: {e}") # Cetak ke log Railway
                
                # Hapus gambar yang mungkin rusak atau menyebabkan error (opsional)
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(url_for('index'))
        else:
            # Jika tipe file tidak diizinkan
            flash('Tipe file tidak diizinkan. Hanya gambar (png, jpg, jpeg, gif) yang diizinkan.', 'error')
            return redirect(request.url)
    
    # Redirect ke halaman utama jika bukan POST request
    return redirect(url_for('index'))

# ========== STARTUP APLIKASI ==========
# Pastikan direktori 'models' ada sebelum memulai proses pemuatan model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Memulai proses pemuatan model di latar belakang saat aplikasi Flask startup.
# Ini penting untuk mencegah Railway timeout karena pengunduhan model yang besar.
load_models_in_background()

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    # Gunakan variabel lingkungan 'PORT' yang disediakan oleh Railway,
    # atau default ke port 8080 jika tidak ada.
    port = int(os.environ.get('PORT', 8080))
    print(f"Aplikasi Flask akan berjalan di host 0.0.0.0, port {port}")
    
    # debug=False untuk lingkungan produksi (SANGAT PENTING untuk keamanan & performa)
    app.run(host='0.0.0.0', port=port, debug=False)