import os
import shutil
import cv2
import numpy as np
from sklearn.cluster import KMeans
from keras_facenet import FaceNet

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


def cluster_images(features: np.array, n_clusters: int) -> np.array:
    try:
        if len(features) == 0:
            raise ValueError("Nenhum recurso foi fornecido para clustering.")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        return kmeans.labels_
    except Exception as e:
        print(f"Erro ao realizar clustering: {e}")
        return np.array([])


def del_group_dir():
    cluster_dir = os.path.abspath("cluster")
    if os.path.exists(cluster_dir):
        try:
            shutil.rmtree(cluster_dir)
            print(f"Diretório {cluster_dir} e seu conteúdo foram excluídos com sucesso.")
        except Exception as e:
            print(f"Erro ao excluir o diretório {cluster_dir}: {e}")
    else:
        print(f"O diretório {cluster_dir} não existe.")


def make_groups_dir(cluster_dir, images, labels):
    try:
        del_group_dir()
        os.makedirs("cluster", exist_ok=True)
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_path = os.path.join(cluster_dir, f"cluster_{label}")
            os.makedirs(cluster_path, exist_ok=True)
        
        for i, (img, label) in enumerate(zip(images, labels)):
            cluster_path = os.path.join(cluster_dir, f"cluster_{label}")
            img_path = os.path.join(cluster_path, f"image_{i}.jpg")
            cv2.imwrite(img_path, (img).astype(np.uint8))
        
        print("Imagens agrupadas e salvas com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar as imagens nos diretórios: {e}")


def group_faces(base_dir, cluster_dir, n_clusters):
    try:
        images = load_images(base_dir)
        if images.size == 0:
            raise ValueError("Nenhuma imagem foi carregada para o agrupamento.")
        
        features = detect_faces(images)
        if features.size == 0:
            raise ValueError("Nenhuma característica foi extraída.")
        
        labels = cluster_images(features, n_clusters)
        if labels.size == 0:
            raise ValueError("Nenhum agrupamento foi realizado.")
        
        make_groups_dir(cluster_dir, images, labels)
    except Exception as e:
        print(f"Erro no agrupamento de rostos: {e}")


if __name__ == "__main__":
    base_dir = 'C:/Users/eduar/Desktop/clustering de images/faces/'
    cluster_dir = os.path.abspath("cluster")
    n_clusters = 10  
    group_faces(base_dir, cluster_dir, n_clusters)
