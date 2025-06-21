import os
# import requests # Hapus baris ini
import gdown # Tambahkan baris ini
from tensorflow.keras.models import load_model

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

    # gdown biasanya memerlukan ID, jadi kita bisa ekstrak atau gunakan URL langsung jika gdown mendukung
    # Untuk gdown, URL dengan "uc?id=" sudah cukup.
    
    if os.path.exists(model_filepath):
        print(f"DEBUG: Model {model_name} sudah ada di {model_filepath}. Melewatkan unduhan.")
        file_size = os.path.getsize(model_filepath)
        print(f"DEBUG: Ukuran file {model_filename}: {file_size} bytes.")
        return model_filepath

    print(f"DEBUG: Mulai mengunduh model {model_name} dari: {model_url} ke: {model_filepath}")
    try:
        # gdown.download handles redirects and authentication better for Google Drive
        gdown.download(url=model_url, output=model_filepath, quiet=False)

        print(f"DEBUG: Model {model_name} berhasil diunduh ke: {model_filepath}")
        if os.path.exists(model_filepath):
            final_size = os.path.getsize(model_filepath)
            print(f"DEBUG: Verifikasi: File {model_filename} ada dan ukurannya {final_size} bytes.")
            # gdown does not easily give total_size, so we skip size comparison for now
            return model_filepath
        else:
            print(f"ERROR: Model {model_name} seharusnya diunduh tetapi file tidak ditemukan di {model_filepath}.")
            return None

    except Exception as e:
        print(f"ERROR: Gagal mengunduh model {model_name} dari {model_url}: {e}")
        return None

def load_all_models():
    """Mengunduh dan memuat ketiga model."""
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
                global MODEL_LOAD_ERROR
                MODEL_LOAD_ERROR = True
        else:
            print(f"ERROR: Tidak dapat menemukan file untuk memuat model {name}.")
            global MODEL_LOAD_ERROR
            MODEL_LOAD_ERROR = True

    global ENSEMBLE_MODELS
    ENSEMBLE_MODELS = loaded_models

    global MODELS_LOADED
    MODELS_LOADED = not MODEL_LOAD_ERROR

    return loaded_models

if __name__ == '__main__':
    models = load_all_models()
    if models:
        print("\nSemua model berhasil dimuat:")
        for name, model in models.items():
            print(f"- {name}: Model dimuat.")
    else:
        print("Ada masalah dalam memuat model.")