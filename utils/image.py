import cv2
import os

def load_image(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Invalid image format: {image_path}")
            return None
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

def save_cropped_faces(cropped_face, hash_hex, output_directory="recognized"):
    try:
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, f"{hash_hex}.jpg")
        cv2.imwrite(output_path, cropped_face)
            
    except Exception as e:
        raise ValueError(f"Error saving cropped faces: {e}")
    
def draw_bounding_boxes(detected_faces, memory, output_directory="output", ):

    try:
        os.makedirs(output_directory, exist_ok=True)

        for name, values in detected_faces.items():
            print("valores recebidos ",values)
            img_path, face = values
            
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            if img_path not in memory.cache_image:
                memory.cache_image[img_path] = load_image(img_path)

            image = memory.cache_image[img_path]
            print("Memoria de imagens: ", memory.cache_image)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            memory.cache_image.update({img_path: image})
              
    except Exception as e:
        raise ValueError(f"Error saving cropped faces: {e}")