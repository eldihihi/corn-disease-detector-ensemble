# utils/model_loader.py
# ... (bagian atas kode sama) ...

def load_all_models():
    global ENSEMBLE_MODELS
    global MODELS_LOADED
    global MODEL_LOAD_ERROR

    MODEL_LOAD_ERROR = False
    loaded_models = {}
    # Hanya coba muat satu model untuk pengujian OOM
    model_names = ["resnet"] # UBAH DARI ["resnet", "vgg", "inception"] menjadi ["resnet"] saja

    for name in model_names:
        # ... (sisa logika download dan load model sama) ...