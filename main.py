import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from keras_facenet import FaceNet

def preprocess_image(img_path: str) -> np.array:
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Imagem não pode ser carregada: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))
        normalized_img = img / 255.0
        return normalized_img
    
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
            print("Nenhuma imagem válida encontrada.")
        return np.array(images)
    except Exception as e:
        print(f"Erro ao carregar imagens do diretório {base_path}: {e}")
        return np.array([])


def detect_faces(imgs: np.array) -> np.array:
    try:
        if len(imgs) == 0:
            raise ValueError("Nenhuma imagem foi fornecida para detecção de faces.")
        
        facenet_model = FaceNet()

        features = facenet_model.embeddings(imgs)
        
        embeddings = np.array(features)
        return embeddings
    except Exception as e:
        print(f"Erro ao detectar rostos: {e}")
        return np.array([])


def cluster_images(features: np.array) -> np.array:
    try:
        if len(features) == 0:
            raise ValueError("Nenhum recurso foi fornecido para clustering.")
        
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(features)
        return clustering.labels_
    except Exception as e:
        print(f"Erro ao realizar clustering: {e}")
        return np.array([])


def make_groups_dir(cluster_dir, images, labels):
    try:
        unique_labels = set(labels)
        print(f"Clusters identificados: {unique_labels}")
        
        for label in unique_labels:
            cluster_path = os.path.join(cluster_dir, f"cluster_{label}")
            os.makedirs(cluster_path, exist_ok=True)
        
        for i, (img, label) in enumerate(zip(images, labels)):
            cluster_path = os.path.join(cluster_dir, f"cluster_{label}")
            img_path = os.path.join(cluster_path, f"image_{i}.jpg")
            cv2.imwrite(img_path, (img * 255).astype(np.uint8))
        
        print("Imagens agrupadas e salvas com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar as imagens nos diretórios: {e}")


def group_faces(base_dir, cluster_dir):
    try:
        print("Carregando imagens...")
        images = load_images(base_dir)
        if images.size == 0:
            raise ValueError("Nenhuma imagem foi carregada para o agrupamento.")
        
        print("Detectando características faciais...")
        features = detect_faces(images)
        if features.size == 0:
            raise ValueError("Nenhuma característica foi extraída.")
        
        print("Realizando agrupamento...")
        labels = cluster_images(features)
        if labels.size == 0:
            raise ValueError("Nenhum agrupamento foi realizado.")
        
        print("Criando diretórios de cluster...")
        make_groups_dir(cluster_dir, images, labels)
    except Exception as e:
        print(f"Erro no agrupamento de rostos: {e}")


if __name__ == "__main__":
    base_dir = 'C:/Users/eduar/Desktop/clustering de images/faces/'
    cluster_dir = 'C:/Users/eduar/Desktop/clustering de images/cluster'
    group_faces(base_dir, cluster_dir)
