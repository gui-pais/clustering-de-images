import os
from cv2 import imread, IMREAD_COLOR


def load_images(base_path:str) -> list:
    images = []
    for root, _, files in os.walk(base_path):
        for file in files:
            img_path = os.path.join(root, file)
            try:
                img = imread(img_path, IMREAD_COLOR)
                if img is not None:
                    images.append(img)
            except Exception as e:
                print(f"Erro ao carregar a imagem {img_path}: {e}")
    return images

if __name__ == "__main__":
    dataset_path ='C:/Users/eduar/Desktop/clustering de images/faces/'
    images = load_images(dataset_path)
