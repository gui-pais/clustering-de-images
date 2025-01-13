import os
import shutil
import cv2
import numpy as np
import shutil


def sort_files(base_dir="faces_recognizeds", output_dir="sort_faces_recognizeds"):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(base_dir):
        label = file.split('.')[0].split('_')[0]
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        source_path = os.path.join(base_dir, file)
        dest_path = os.path.join(label_dir, file)
        
        if os.path.isfile(source_path):
            shutil.copy(source_path, dest_path)

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
        os.makedirs("cluster", exist_ok=True)
        cluster_dir = os.path.abspath("cluster")
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_path = os.path.join(cluster_dir, f"cluster_{label}")
            os.makedirs(cluster_path, exist_ok=True)
        
        for i, (img, label) in enumerate(zip(images, labels)):
            cluster_path = os.path.join(cluster_dir, f"cluster_{label}")
            img_path = os.path.join(cluster_path, f"image_{i+1}.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, (img).astype(np.uint8), )
        
        print("Imagens agrupadas e salvas com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar as imagens nos diretórios: {e}")
