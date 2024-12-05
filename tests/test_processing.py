import numpy as np
from utils.processing import preprocess_image, load_images

def test_preprocess_image():
    img_path = "test_faces/Abdoulaye_Wade_0001.jpg"
    preprocessed = preprocess_image(img_path)
    assert preprocessed.shape == (160, 160, 3), "Imagem não foi redimensionada corretamente."

def test_load_images():
    base_path = "test_faces"
    images = load_images(base_path)
    assert len(images) > 0, "Nenhuma imagem foi carregada."
    assert isinstance(images, np.ndarray), "As imagens carregadas não estão em um array numpy."
