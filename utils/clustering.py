import numpy as np
from sklearn.cluster import AgglomerativeClustering
from keras_facenet import FaceNet
from utils.processing import load_images
from utils.dirs import make_groups_dir

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
        
        # clt = AgglomerativeClustering(
        # n_clusters=None,
        # distance_threshold=0.6,
        # metric="cosine",
        # linkage="complete",
        # )
        
        clt = AgglomerativeClustering(
            n_clusters=28,
            linkage="complete",
            metric="cosine"
        )
        
        clt.fit(features)
        return clt.labels_
    
    except Exception as e:
        print(f"Erro ao realizar clustering: {e}")
        return np.array([])

def grouping_faces(base_dir):
    try:
        images = load_images(base_dir)
        if images.size == 0:
            raise ValueError("Nenhuma imagem foi carregada para o agrupamento.")
        
        features = detect_faces(images)
        if features.size == 0:
            raise ValueError("Nenhuma característica foi extraída.")
        
        labels = cluster_images(features)
        if labels.size == 0:
            raise ValueError("Nenhum agrupamento foi realizado.")
        
        make_groups_dir(images, labels)
    except Exception as e:
        print(f"Erro no agrupamento de rostos: {e}")