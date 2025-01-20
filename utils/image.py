import cv2
import os

def get_valid_image(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Imagem não encontrada: {image_path}")
            return None
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Formato de imagem inválido: {image_path}")
            return None
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erro ao carregar a imagem: {image_path}")
            return None
        return img
    except Exception as e:
        raise ValueError(f"Erro ao obter a imagem: {e}")
    
def save_cropped_faces(image, embeddings_face,output_dir="output"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (name, face) in enumerate(embeddings_face.items(), start=1):
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cropped_face = image[y:y+h, x:x+w]
            output_path = os.path.join(output_dir, f"{name}_{i}.jpg")
            cv2.imwrite(output_path, cropped_face)
    except Exception as e:
        raise ValueError(f"Erro ao salvar faces recortadas: {e}")
    
def save_face(image, face, output_dir="recognized"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cropped_face = image[y:y+h, x:x+w]
        n = sum([len(files) for _, _, files in os.walk(output_dir)])
        output_path = os.path.join(output_dir, f"{os.path.basename(output_dir)}{n+1}.jpg")
        cv2.imwrite(output_path, cropped_face)
    except Exception as e:
        raise ValueError(f"Erro ao salvar face: {e}")