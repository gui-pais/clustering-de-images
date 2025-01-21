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

def save_cropped_faces(detected_faces, image=None,output_directory="output"):
    try:
        os.makedirs(output_directory, exist_ok=True)

        for name, face in detected_faces.items():
            print(face)
            if image is not None:
                x, y, width, height = face.left(), face.top(), face.width(), face.height()
                cropped_face = image[y:y+height, x:x+width]
            else:
                face = face.replace("\\", "/")
                cropped_face = load_image(face)
            total_files = sum(len(files) for _, _, files in os.walk(output_directory))
            output_path = os.path.join(output_directory, f"{name}_{total_files + 1}.jpg")
            cv2.imwrite(output_path, cropped_face)
            
    except Exception as e:
        raise ValueError(f"Error saving cropped faces: {e}")