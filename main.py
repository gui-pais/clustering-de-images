import os
import cv2
import dlib
import numpy as np
from sklearn.cluster import DBSCAN
import tensorflow as tf
from tensorflow.keras.applications.vggface import VGGFace

def preprocess_image(img_path: str) -> np.array:
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Imagem não pode ser carregada: {img_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        normalized_img = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))
        return normalized_img
    
    except Exception as e:
        print(f"Erro ao pré-processar a imagem {img_path}: {e}")
        return None

def load_images(base_path: str) -> list:
    images = []
    try:
        for root, _, files in os.walk(base_path):
            for file in files:
                img_path = os.path.join(root, file)
                img = preprocess_image(img_path)
                if img is not None:
                    images.append(img)
    except Exception as e:
        print(f"Erro ao carregar imagens do diretório {base_path}: {e}")
    
    return images

def detect_faces(imgs: list):
    try:
        vggface_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
        features = vggface_model.predict(imgs)
        return features
    except Exception as e:
        print(f"Erro ao detectar rostos: {e}")
        return []

def cluster_images(features):
    try:
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(features)
        return clustering.labels_
    except Exception as e:
        print(f"Erro ao realizar clustering: {e}")
        return []
    
def make_groups_dir(base_dir, images, labels):
    for i, label in enumerate(labels):
            cluster_dir = os.path.join(base_dir, f"cluster_{label}")
            os.makedirs(cluster_dir, exist_ok=True)

    for i, (img, lab) in enumerate(zip(images, labels)):
        cluster_dir = os.path.join(base_dir, f"cluster_{lab}")
        img_path = os.path.join(cluster_dir, f"image_{i}.jpg")
        try:
            cv2.imwrite(img_path, img)
        except Exception as e:
            print(f"Erro ao salvar a imagem {img_path}: {e}")

def group_faces(base_dir, model_path):

    try:
        images = load_images(base_dir)
        if not images:
            raise ValueError("Nenhuma imagem foi carregada para o agrupamento.")
        
        
        features = detect_faces(images)
        
        labels = cluster_images(features)
        
        make_groups_dir(base_dir, images, labels)
    
    except Exception as e:
        print(f"Erro no agrupamento de rostos: {e}")

if __name__ == "__main__":
    dataset_path = 'C:/Users/eduar/Desktop/clustering de images/faces/'
    model_path = 'path_to_your_model.h5'
    
    for index, image in enumerate(load_images(dataset_path)):
        print(f"começa aqui\nimagem{index}{image}\ntermina aqui")
