import cv2
import os

class ImageMemory:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageMemory, cls).__new__(cls)
            cls._instance.image = None
            cls._instance.last_img = None
        return cls._instance

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

def save_cropped_faces(detected_faces, image=None,output_directory="output"):
    try:
        os.makedirs(output_directory, exist_ok=True)

        for _, face in detected_faces.items():
            if image is not None:
                x, y, width, height = face.left(), face.top(), face.width(), face.height()
                cropped_face = image[y:y+height, x:x+width]
            else:
                face = face.replace("\\", "/")
                cropped_face = load_image(face)
            total_files = sum(len(files) for _, _, files in os.walk(output_directory))
            output_path = os.path.join(output_directory, f"{total_files + 1}.jpg")
            cv2.imwrite(output_path, cropped_face)
            
    except Exception as e:
        raise ValueError(f"Error saving cropped faces: {e}")
    
def draw_bounding_boxes(detected_faces, output_directory="output"):
    memory = ImageMemory()
    try:
        os.makedirs(output_directory, exist_ok=True)

        for name, values in detected_faces.items():
            img_path, face = values
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            
            if memory.image is None or memory.last_img_path != img_path:
                memory.image = load_image(img_path)
                memory.last_img_path = img_path

            cv2.rectangle(memory.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(memory.image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
        if memory.image is not None:
            output_path = os.path.join(output_directory, os.path.basename(memory.last_img_path))
            cv2.imwrite(output_path, memory.image)
              
    except Exception as e:
        raise ValueError(f"Error saving cropped faces: {e}")