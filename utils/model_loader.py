import os
import requests
from tensorflow.keras.models import load_model

MODEL_DIR = "models" # Folder lokal untuk menyimpan model yang diunduh

# URL Google Drive Anda (Ganti dengan ID Google Drive Anda!)
# Formatnya adalah: https://drive.google.com/uc?export=download&id=YOUR_FILE_ID
MODEL_URLS = {
    "resnet": "https://drive.google.com/uc?export=download&id=1Jvmr1KHY8cDSYgJEnIQ-OhqcUV8cj-qM",     # ID ResNet50 Anda
    "vgg": "https://drive.google.com/uc?export=download&id=1KKUN75s1UQTESqv8tULqAC-HlVy_OBU8",       # ID VGG16 Anda
    "inception": "https://drive.google.com/uc?export=download&id=12YT--eiq09i3I8BgY60KBJkOnJFHEKBiob" # ID Inception Anda
}

# Mapping nama file ke URL
MODEL_FILENAMES = {
    "resnet": "resnet_best_model.h5",
    "vgg": "vgg_best_model.h5",
    "inception": "inception_best_model.h5"
}

def download_model(model_name):
    """Mengunduh model dari URL yang diberikan."""
    # UBAH BARIS INI
    os.makedirs(MODEL_DIR, exist_ok=True) # Tambahkan exist_ok=True

    model_filename = MODEL_FILENAMES[model_name]
    model_filepath = os.path.join(MODEL_DIR, model_filename)
    model_url = MODEL_URLS[model_name]

    if os.path.exists(model_filepath):
        print(f"Model {model_name} sudah ada: {model_filepath}. Melewatkan unduhan.")
        return model_filepath

    print(f"Mengunduh model {model_name} dari: {model_url}")
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status() # Akan memunculkan HTTPError untuk status kode 4xx/5xx

        with open(model_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model {model_name} berhasil diunduh ke: {model_filepath}")
        return model_filepath
    except requests.exceptions.RequestException as e:
        print(f"Gagal mengunduh model {model_name}: {e}")
        return None

def load_all_models():
    """Mengunduh dan memuat ketiga model."""
    loaded_models = {}
    model_names = ["resnet", "vgg", "inception"]

    for name in model_names:
        filepath = download_model(name)
        if filepath:
            try:
                print(f"Memuat model {name} dari: {filepath}")
                model = load_model(filepath)
                loaded_models[name] = model
                print(f"Model {name} berhasil dimuat.")
            except Exception as e:
                print(f"Gagal memuat model {name} dari {filepath}: {e}")
        else:
            print(f"Tidak dapat menemukan file untuk memuat model {name}.")
    return loaded_models

if __name__ == '__main__':
    # Contoh penggunaan:
    models = load_all_models()
    if models:
        print("\nSemua model berhasil dimuat:")
        for name, model in models.items():
            print(f"- {name}: Model dimuat.")
    else:
        print("Ada masalah dalam memuat model.")