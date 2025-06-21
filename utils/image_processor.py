# utils/image_processor.py
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

# Definisi ukuran input untuk setiap model
IMAGE_SIZE = {
    "resnet": (224, 224),
    "vgg": (224, 224),
    "inception": (299, 299)
}

def preprocess_image(image_path, model_name):
    """
    Memuat dan melakukan preprocessing gambar agar sesuai dengan input model.
    """
    target_size = IMAGE_SIZE.get(model_name)
    if not target_size:
        raise ValueError(f"Ukuran gambar tidak dikenal untuk model: {model_name}")

    img = Image.open(image_path).convert('RGB') # Pastikan gambar dalam format RGB
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch

    if model_name == "resnet":
        return resnet_preprocess(img_array)
    elif model_name == "vgg":
        return vgg_preprocess(img_array)
    elif model_name == "inception":
        return inception_preprocess(img_array)
    else:
        return img_array # Default atau jika tidak ada preprocess spesifik