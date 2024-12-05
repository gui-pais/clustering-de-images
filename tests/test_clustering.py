import numpy as np
from utils.clustering import detect_faces, cluster_images

def test_detect_faces():
    mock_images = np.random.rand(5, 160, 160, 3)
    embeddings = detect_faces(mock_images)
    assert embeddings.shape[0] == 5, "Embeddings não correspondem ao número de imagens."

def test_cluster_images():
    mock_features = np.random.rand(10, 128)
    labels = cluster_images(mock_features)
    assert len(labels) == 10, "Número de labels não corresponde ao número de embeddings."

