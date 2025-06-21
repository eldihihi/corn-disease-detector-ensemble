import os
import requests
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
MODEL_URLS = {
    "resnet": "https://drive.google.com/uc?export=download&id=1Jvmr1KHY8cDSYgJEnIQ-OhqcUV8cj-qM",
    "vgg": "https://drive.google.com/uc?export=download&id=1KKUN75s1UQTESqv8tULqAC-HlVy_OBU8",
    "inception": "https://drive.google.com/uc?export=download&id=12YT--eiq09i3I8BgY60KBJkOnJFHEKBiob"
}
MODEL_FILENAMES = {
    "resnet": "resnet_best_model.h5",
    "vgg": "vgg_best_model.h5",
    "inception": "inception_best_model.h5"
}

# Variabel global didefinisikan di sini.
# Mereka bisa diakses dan dimodifikasi langsung dari fungsi tanpa kata kunci 'global'
# karena mereka sudah ada di namespace modul (global).
ENSEMBLE_MODELS = {}
MODELS_LOADED = False
MODEL_LOAD_ERROR = False


def download_model(model_name):
    """Mengunduh model dari URL yang diberikan."""
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

    print(f"DEBUG: Mulai mengunduh model {model_name} dari: {model_url}")
    try:
        response = requests.get(model_url, stream=True, allow_redirects=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(model_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
        print(f"DEBUG: Model {model_name} berhasil diunduh ke: {model_filepath}")
        if os.path.exists(model_filepath):
            final_size = os.path.getsize(model_filepath)
            print(f"DEBUG: Verifikasi: File {model_filename} ada dan ukurannya {final_size} bytes.")
            if total_size > 0 and final_size != total_size:
                print(f"WARNING: Ukuran file {model_name} tidak cocok! Diharapkan {total_size}, didapat {final_size}.")
            return model_filepath
        else:
            print(f"ERROR: Model {model_name} seharusnya diunduh tetapi file tidak ditemukan di {model_filepath}.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Gagal mengunduh model {model_name} dari {model_url}: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan tak terduga saat mengunduh {model_name}: {e}")
        return None

def load_all_models():
    """Mengunduh dan memuat ketiga model."""
    # HAPUS BARIS 'global' untuk MODEL_LOAD_ERROR dan MODELS_LOADED
    # Karena variabel ini sudah didefinisikan di tingkat modul (global scope),
    # kita bisa langsung memodifikasinya tanpa perlu mendeklarasikannya lagi dengan 'global'.

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
                # Memodifikasi variabel global tanpa 'global' karena sudah ada
                MODEL_LOAD_ERROR = True
        else:
            print(f"ERROR: Tidak dapat menemukan file untuk memuat model {name}.")
            MODEL_LOAD_ERROR = True

    # Setelah semua model selesai diproses (berhasil atau gagal),
    # atur MODELS_LOADED berdasarkan apakah ada error.
    # ENSEMBLE_MODELS harus diperbarui di sini setelah semua model dimuat
    global ENSEMBLE_MODELS # ENSEMBLE_MODELS perlu 'global' karena diinisialisasi ulang
    ENSEMBLE_MODELS = loaded_models

    global MODELS_LOADED # MODELS_LOADED perlu 'global' karena diinisialisasi ulang
    MODELS_LOADED = not MODEL_LOAD_ERROR # Jika tidak ada error, maka model berhasil dimuat

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