# app.py

import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

# Impor fungsi dan variabel global dari modul utilitas
# Pastikan MODEL_DIR juga diimpor
from utils.model_loader import load_models_in_background, ENSEMBLE_MODELS, MODELS_LOADED, MODEL_LOAD_ERROR, MODEL_DIR
from utils.image_processor import preprocess_image

# Menginisialisasi aplikasi Flask
app = Flask(__name__)

# ... (sisa kode app.py Anda seperti sebelumnya) ...

# ========== STARTUP APLIKASI ==========
# Pastikan direktori 'models' ada sebelum memulai proses pemuatan model
# Baris ini sekarang akan berfungsi karena MODEL_DIR sudah didefinisikan
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Memulai proses pemuatan model di latar belakang saat aplikasi Flask startup.
load_models_in_background()

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Aplikasi Flask akan berjalan di host 0.0.0.0, port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)