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
    
def split_image(image_path, output_dir="split_images"):
    try:
        image = get_valid_image(image_path)
        if image is None:
            return
        
        height, width, _ = image.shape
        if width != 3840 or height != 2160:
            print(f"A imagem não tem o tamanho esperado de 3840x2160 pixels. Tamanho atual: {width}x{height}")

        img1 = image[0:1080, 0:1920]
        img2 = image[0:1080, 1920:3840]
        img3 = image[1080:2160, 0:1920]
        img4 = image[1080:2160, 1920:3840]

        os.makedirs(output_dir, exist_ok=True)

        for i, img in enumerate([img1, img2, img3, img4], 1):
            cv2.imwrite(os.path.join(output_dir, f'img{i}.jpg'), img)
    except Exception as e:
        raise ValueError(f"Erro ao dividir a imagem: {e}")
    
def save_cropped_faces(image, persons_recognized, file_name,output_dir="faces_recognizeds", ):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (_, face) in enumerate(persons_recognized.items(), start=1):
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cropped_face = image[y:y+h, x:x+w]
            first = file_name.split(" ")[0]
            try:
                second = file_name.split(" ")[1]
                if second.endswith(".jpg"):
                    second = second.split(".")[0]
                first = f"{first} {second}"
            except:
                pass
            if first.endswith(".jpg"):
                first = first.split(".")[0]
            output_path = os.path.join(output_dir, f"{first}_{i}.jpg")
            cv2.imwrite(output_path, cropped_face)
    except Exception as e:
        raise ValueError(f"Erro ao salvar faces recortadas: {e}")