# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
import threading
import time

# Impor fungsi dari file utilitas
from utils.model_loader import load_all_models, MODEL_DIR, MODEL_FILENAMES
from utils.image_processor import preprocess_image

app = Flask(__name__)

# Konfigurasi Upload Folder
UPLOAD_FOLDER = 'static/uploads' # Simpan di static agar bisa diakses browser
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Batas ukuran file 16MB

# Tipe file yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Kelas-kelas penyakit Anda (GANTI INI SESUAI DENGAN MODEL ANDA)
CLASS_NAMES = [
    'Corn_Common_Rust',
    'Corn_Gray_Leaf_Spot',
    'Corn_Healthy',
    'Corn_Northern_Leaf_Blight'
    # Tambahkan semua kelas penyakit yang model Anda prediksi
]

# Variabel global untuk menyimpan model yang dimuat
LOADED_MODELS = {}
MODEL_LOAD_ERROR = None # Untuk menyimpan pesan error jika gagal memuat model

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models_in_background():
    """Memuat model di latar belakang saat aplikasi startup."""
    global LOADED_MODELS, MODEL_LOAD_ERROR
    try:
        print("Memulai proses memuat model di latar belakang...")
        LOADED_MODELS = load_all_models()
        if not LOADED_MODELS:
            MODEL_LOAD_ERROR = "Gagal memuat semua model. Periksa log."
            print(MODEL_LOAD_ERROR)
        else:
            print("Semua model berhasil dimuat dan siap digunakan.")
    except Exception as e:
        MODEL_LOAD_ERROR = f"Kesalahan fatal saat startup model: {e}"
        print(MODEL_LOAD_ERROR)

# Jalankan fungsi pemuatan model di thread terpisah saat aplikasi dimulai
# Ini penting agar aplikasi bisa start cepat dan memuat model di latar belakang
# tanpa membuat Railway timeout.
with app.app_context():
    model_thread = threading.Thread(target=load_models_in_background)
    model_thread.start()

@app.route('/')
def index():
    if MODEL_LOAD_ERROR:
        flash(MODEL_LOAD_ERROR, 'error')
    elif not LOADED_MODELS:
        flash("Model sedang dimuat, mohon tunggu sebentar...", 'info')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada bagian file', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file terpilih', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if MODEL_LOAD_ERROR:
                flash(MODEL_LOAD_ERROR, 'error')
                return redirect(url_for('index'))
            if not LOADED_MODELS or len(LOADED_MODELS) < 3: # Pastikan semua 3 model termuat
                flash("Model belum sepenuhnya dimuat. Mohon tunggu beberapa saat dan coba lagi.", 'info')
                return redirect(url_for('index'))

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Prediksi menggunakan ensemble
                predictions = []
                for model_name, model in LOADED_MODELS.items():
                    processed_image = preprocess_image(filepath, model_name)
                    pred = model.predict(processed_image)
                    predictions.append(pred)

                # Ensemble Averaging
                ensemble_prediction = np.mean(predictions, axis=0)
                predicted_class_index = np.argmax(ensemble_prediction)
                predicted_class = CLASS_NAMES[predicted_class_index]
                confidence = np.max(ensemble_prediction) * 100 # Dalam persen

                # Hapus gambar setelah prediksi (opsional, tergantung kebutuhan)
                # os.remove(filepath)

                flash(f"Deteksi: {predicted_class} (Keyakinan: {confidence:.2f}%)", 'success')
                return render_template('index.html',
                                       result=predicted_class,
                                       confidence=f"{confidence:.2f}%",
                                       uploaded_image_url=url_for('static', filename='uploads/' + filename))
            except Exception as e:
                flash(f"Terjadi kesalahan saat memproses gambar: {e}", 'error')
                # Hapus gambar yang mungkin rusak
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(url_for('index'))
        else:
            flash('Tipe file tidak diizinkan. Hanya gambar (png, jpg, jpeg, gif) yang diizinkan.', 'error')
            return redirect(request.url)
    return redirect(url_for('index'))

# Jalankan aplikasi
if __name__ == '__main__':
    # Pastikan direktori models ada sebelum memuat model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # PORT environment variable untuk deployment di Railway
    port = int(os.environ.get('PORT', 5000))
    print(f"Aplikasi Flask akan berjalan di host 0.0.0.0, port {port}")
    # debug=True hanya untuk pengembangan lokal, ubah ke False di produksi
    app.run(host='0.0.0.0', port=port, debug=False)
