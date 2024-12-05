import os
import shutil
import cv2
import numpy as np


def del_group_dir(dir_name):
    cluster_dir = os.path.abspath(dir_name)
    if os.path.exists(cluster_dir):
        try:
            shutil.rmtree(cluster_dir) 
            print(f"Diretório {cluster_dir} e seu conteúdo foram excluídos com sucesso.")
        except Exception as e:
            print(f"Erro ao excluir o diretório {cluster_dir}: {e}")
    else:
        print(f"O diretório {cluster_dir} não existe.")


def make_groups_dir(images, labels):
    try:
        del_group_dir("cluster")
        os.makedirs("cluster", exist_ok=True)
        cluster_dir = os.path.abspath("cluster")
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_path = os.path.join(cluster_dir, f"cluster_{label}")
            os.makedirs(cluster_path, exist_ok=True)
        
        for i, (img, label) in enumerate(zip(images, labels)):
            cluster_path = os.path.join(cluster_dir, f"cluster_{label}")
            img_path = os.path.join(cluster_path, f"image_{i+1}.jpg")
            cv2.imwrite(img_path, (img).astype(np.uint8))
        
        print("Imagens agrupadas e salvas com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar as imagens nos diretórios: {e}")
