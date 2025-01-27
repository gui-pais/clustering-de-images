import os
import dlib
import numpy as np
import cv2
import hashlib 
from re import search
from subprocess import call
from ultralytics import YOLO as YOLOv10
from .cache import save_data, load_data
from .image import load_image, save_cropped_faces, draw_bounding_boxes

class ImageMemory:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageMemory, cls).__new__(cls)
            cls._instance.cache_image = {}
        return cls._instance

class SharedMemory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedMemory, cls).__new__(cls)
            cls._instance.iteration_cache = {}
        return cls._instance
    
class YoloToSuperResoluiton:
    def __init__(self, face_detection_model_path):
        self.face_detector = YOLOv10(face_detection_model_path)
        self.memory = SharedMemory()
        
    def _recognize_faces(self, image_path: str, output_directory):
        try:
            image = load_image(image_path)

            detected_faces = self.face_detector(image)[0]
            print(f"Quantidade de faces reconhecidas", len(detected_faces))
            for detected_face in detected_faces.boxes:
                
                cords = detected_face.xywh[0].cpu().numpy().tolist()
                x_center, y_center, width, height = cords
                x1 = int(x_center - width / 2) 
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                cropped_face = image[y1:y2, x1:x2]
                detected_face = dlib.rectangle(x1,y1,x2,y2)
                
                
                data = cropped_face.tobytes()
                hash_obj = hashlib.sha256(data)
                hash_hex = hash_obj.hexdigest()
                
                save_cropped_faces(cropped_face, hash_hex, output_directory)
                
                bounding_boxes = (image_path, detected_face)
                self.memory._instance.iteration_cache[hash_hex] = bounding_boxes        
                
        except Exception as e:
            raise ValueError(f"Error during face recognition: {e}")
        
    def process_images(self, input_directory: str, output_directory: str, batch_command: str):
        for root, _, files in os.walk(input_directory):
            for file in files:
                image_path = os.path.join(root, file)
                image = load_image(image_path)
                if image is None:
                    continue
                self._recognize_faces(image_path, output_directory)
        call(batch_command, shell=True)

class FaceRecognizer:
    def __init__(self, predictor_model_path: str, face_recognition_model_path: str, similarity_threshold: float = 0.6):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)
        self.face_recognizer = dlib.face_recognition_model_v1(face_recognition_model_path)
        self.similarity_threshold = similarity_threshold
        self.memory = SharedMemory()
        self.image_memory = ImageMemory()

    def _calculate_distances(self, known_faces: dict, image, detected_face) -> tuple:
        facial_landmarks = self.shape_predictor(image, detected_face)
        face_descriptor = np.asarray(self.face_recognizer.compute_face_descriptor(image, facial_landmarks), dtype=np.float64)[np.newaxis, :]

        distances = [np.linalg.norm(face_descriptor - known_descriptor) for known_descriptor in known_faces.values()]
        min_index = np.argmin(distances)
        match_name = list(known_faces.keys())[min_index]

        return match_name, distances, min_index
    
    def _recognize_faces(self, image_path: str, known_faces: dict) -> dict:
        recognized_faces = {}
        try:
            image = load_image(image_path)
            index_match = 0
            detected_faces = self.face_detector(image, 2)
            for detected_face in detected_faces:
                x, y, w, h = detected_face.left(), detected_face.top(), detected_face.width(), detected_face.height()
                cropped_face = image[y:y+h, x:x+w]
                data = cropped_face.tobytes()
                hash_obj = hashlib.sha256(data)
                hash_hex = hash_obj.hexdigest()
                save_cropped_faces(cropped_face, hash_hex, "dlib_dec")
                pattern = r"([a-fA-F0-9]{64})"
                index_match = search(pattern, image_path)
                match_name, distances, min_index = self._calculate_distances(known_faces, image, detected_face)
                print(image_path)
                print(f"Foi encontrado o aluno {match_name} com {distances[min_index]} no path {index_match.group(1)}")
                if distances[min_index] < self.similarity_threshold:
                    recognized_faces[match_name] = self.memory._instance.iteration_cache[index_match.group(1)]

            return recognized_faces
        except Exception as e:
            raise ValueError(f"Error during face recognition: {e}")
    
    def process_images(self, input_directory: str, output_directory: str):
        known_faces = load_data("rec_faces_dlib")
        for root, _, files in os.walk(input_directory):
            for file in files:
                image_path = os.path.join(root, file)
                try:
                    image = load_image(image_path)
                    if image is None:
                        continue
                    detected_faces = self._recognize_faces(image_path, known_faces)
                    if detected_faces:
                        draw_bounding_boxes(detected_faces, self.image_memory, output_directory)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        for path, img in self.image_memory.cache_image.items():
            output = os.path.join("output", os.path.basename(path))
            cv2.imwrite(output, img)
            
    def extract_known_faces(self, face_images_directory="faces"):
        known_faces = {}
        try:
            if os.path.exists(face_images_directory):
                for file in os.listdir(face_images_directory):
                    face_name, _ = os.path.splitext(file)
                    image_path = os.path.join(face_images_directory, file)
                    image = load_image(image_path)
                    if image is None:
                        continue
                    detected_faces = self.face_detector(image, 1)

                    for detected_face in detected_faces:
                        facial_landmarks = self.shape_predictor(image, detected_face)
                        face_descriptor = np.array(self.face_recognizer.compute_face_descriptor(image, facial_landmarks), dtype=np.float64)[np.newaxis, :]
                        known_faces[face_name.lower().capitalize()] = face_descriptor

            save_data(known_faces, "rec_faces_dlib")
        except Exception as e:
            raise ValueError(f"Error extracting known faces: {e}")
            