import os
import cv2
import numpy as np

def preprocess_image(img_path: str) -> np.array:
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Imagem não pode ser carregada: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))
        return img
    
    except Exception as e:
        print(f"Erro ao pré-processar a imagem {img_path}: {e}")
        return None


def load_images(base_path: str) -> np.array:
    images = []
    try:
        for root, _, files in os.walk(base_path):
            for file in files:
                img_path = os.path.join(root, file)
                img = preprocess_image(img_path)
                if img is not None:
                    images.append(img)
        if not images:
            raise ValueError("Nenhuma imagem válida encontrada.")
        return np.array(images)
    except Exception as e:
        print(f"Erro ao carregar imagens do diretório {base_path}: {e}")
        return np.array([])