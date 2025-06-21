# utils/model_loader.py (REVISI FINAL UNTUK MASALAH GLOBAL DAN GDOWN)
import os
import gdown # Pastikan ini diimpor
from tensorflow.keras.models import load_model
import threading

MODEL_DIR = "models"
MODEL_URLS = {
    # Gunakan URL dengan format "uc?id=" seperti yang Anda berikan
    "resnet": "https://drive.google.com/uc?id=1jVmr1kHY8cDSYgJEnIQ-OhqcUV8cj-qM",
    "vgg": "https://drive.google.com/uc?id=1kKUN75slUQtEsqv8tULqAC-HIVy_OBU8",
    "inception": "https://drive.google.com/uc?id=12YT-eiq09i3B8gY60KBjkOnJFHEKBIob"
}
MODEL_FILENAMES = {
    "resnet": "resnet_best_model.h5",
    "vgg": "vgg_best_model.h5",
    "inception": "inception_best_model.h5"
}

# Variabel global didefinisikan di sini.
# Pastikan variabel-variabel ini hanya diinisialisasi di tingkat global.
ENSEMBLE_MODELS = {}
MODELS_LOADED = False
MODEL_LOAD_ERROR = False


def download_model(model_name):
    """Mengunduh model dari URL yang diberikan menggunakan gdown."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"DEBUG: Memastikan direktori '{MODEL_DIR}' ada. Path: {os.path.abspath(MODEL_DIR)}")

    model_filename = MODEL_FILENAMES[model_name]
    model_filepath = os.path.join(MODEL_DIR, model_filename)
    model_url = MODEL_URLS[model_name]
    
    if os.path.exists(model_filepath):
        print(f"DEBUG: Model {model_name} sudah ada di {model_filepath}. Melewatkan unduhan.")
        file_size = os.path.getsize(model_filepath)
        print(f"DEBUG: Ukuran file {model_filename}: {file_size} bytes.")
        return model_filepath

    print(f"DEBUG: Mulai mengunduh model {model_name} dari: {model_url} ke: {model_filepath}")
    try:
        gdown.download(url=model_url, output=model_filepath, quiet=False)

        print(f"DEBUG: Model {model_name} berhasil diunduh ke: {model_filepath}")
        if os.path.exists(model_filepath):
            final_size = os.path.getsize(model_filepath)
            print(f"DEBUG: Verifikasi: File {model_filename} ada dan ukurannya {final_size} bytes.")
            return model_filepath
        else:
            print(f"ERROR: Model {model_name} seharusnya diunduh tetapi file tidak ditemukan di {model_filepath}.")
            return None

    except Exception as e:
        print(f"ERROR: Gagal mengunduh model {model_name} dari {model_url}: {e}")
        return None

def load_all_models():
    """Mengunduh dan memuat ketiga model."""
    # Deklarasikan variabel global yang akan dimodifikasi di awal fungsi
    global ENSEMBLE_MODELS
    global MODELS_LOADED
    global MODEL_LOAD_ERROR # Ini adalah baris 81 di contoh error Anda!

    MODEL_LOAD_ERROR = False # Reset status error setiap kali pemuatan dimulai
    loaded_models = {}
    model_names = ["resnet", "vgg", "inception"]

    for name in model_names:
        filepath = download_model(name)
        if filepath:
            try:
                print(f"DEBUG: Memuat model {name} dari: {filepath}")
                model = load_model(filepath)
                loaded_models[name] = model
                print(f"DEBUG: Model {name} berhasil dimuat.")
            except Exception as e:
                print(f"ERROR: Gagal memuat model {name} dari {filepath}: {e}")
                MODEL_LOAD_ERROR = True # Tetapkan nilai setelah deklarasi global
                break # Hentikan proses jika ada satu model yang gagal dimuat
        else:
            print(f"ERROR: Tidak dapat menemukan file untuk memuat model {name}.")
            MODEL_LOAD_ERROR = True # Tetapkan nilai setelah deklarasi global
            break # Hentikan proses jika ada satu model yang gagal diunduh

    ENSEMBLE_MODELS = loaded_models
    MODELS_LOADED = not MODEL_LOAD_ERROR

    if MODELS_LOADED:
        print("INFO: Semua model berhasil dimuat.")
    else:
        print("INFO: Gagal memuat beberapa model.")
    return loaded_models

# Fungsi untuk memuat model di latar belakang
def load_models_in_background():
    """
    Meluncurkan thread untuk memuat model di latar belakang agar aplikasi dapat segera responsif.
    """
    print("DEBUG: Memulai proses memuat model di latar belakang...")
    thread = threading.Thread(target=load_all_models)
    thread.daemon = True  # Memungkinkan aplikasi keluar meskipun thread ini masih berjalan
    thread.start()

if __name__ == '__main__':
    # Ini hanya akan berjalan jika Anda menjalankan model_loader.py secara langsung
    # Bukan ketika diimpor oleh app.py
    print("Menjalankan model_loader.py secara langsung untuk pengujian...")
    models = load_all_models()
    if models and not MODEL_LOAD_ERROR:
        print("\nSemua model berhasil dimuat:")
        for name, model in models.items():
            print(f"- {name}: Model dimuat.")
    else:
        print("Ada masalah dalam memuat model.")