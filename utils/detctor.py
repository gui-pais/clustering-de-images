import os
import cv2
import dlib
import hashlib
import numpy as np
import subprocess
from .cache import save_data, load_data
from .image import load_image, save_cropped_faces, draw_bounding_boxes

class SharedMemory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedMemory, cls).__new__(cls)
            cls._instance.iteration_cache = []
            cls._instance.cache_index = 0
        return cls._instance

class FaceDetector:
    def __init__(self, predictor_model_path: str, face_recognition_model_path: str, similarity_threshold: float = 0.6):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)
        self.face_recognizer = dlib.face_recognition_model_v1(face_recognition_model_path)
        self.similarity_threshold = similarity_threshold
        self.memory = SharedMemory()

    def _resize_image(self, image: np.ndarray, target_width: int) -> np.ndarray:
        target_height = int(image.shape[0] * (target_width / image.shape[1]))
        return cv2.resize(image, (target_width, target_height))
    
    def _calculate_distances(self, known_faces: dict, image, detected_face) -> tuple:
            facial_landmarks = self.shape_predictor(image, detected_face)
            face_descriptor = np.array(self.face_recognizer.compute_face_descriptor(image, facial_landmarks), dtype=np.float64)[np.newaxis, :]

            distances = [np.linalg.norm(face_descriptor - known_descriptor) for known_descriptor in known_faces.values()]
            min_index = np.argmin(distances)
            match_name = list(known_faces.keys())[min_index]
            
            return match_name, distances, min_index
        
    def get_hash(self, image: np.ndarray) -> str:
        arr_bytes = image.tobytes()
        hash_object = hashlib.sha256(arr_bytes)
        return hash_object.hexdigest()

    def _recognize_faces(self, image_path: str, known_faces: dict, iteration: int) -> dict:
        recognized_faces = {}
        known_faces_copy = known_faces.copy()
        try:
            image = load_image(image_path)
            if image is None:
                raise ValueError(f"Unable to load image: {image_path}")
            
            if iteration == 1:
                detected_faces = self.face_detector(image, 1)
                for detected_face in detected_faces:
                    match_name, distances, min_index = self._calculate_distances(known_faces_copy, image, detected_face)
                    if distances[min_index] < self.similarity_threshold:
                        del known_faces_copy[match_name]
                        recognized_faces[match_name.lower().capitalize()] = detected_face
                        bounding_boxes = (image_path, detected_face)
                        print(bounding_boxes)
                        self.memory._instance.iteration_cache.append(bounding_boxes)
            
            elif iteration == 2:
                print("cahce_index ",self.memory._instance.cache_index)
                if len(self.memory._instance.iteration_cache) > self.memory._instance.cache_index:
                    detected_face = dlib.rectangle(0, 0, image.shape[1], image.shape[0])
                    print(image_path)
                    match_name, distances, min_index = self._calculate_distances(known_faces_copy, image, detected_face)
                    print("match_name ",match_name)
                    print("teste", self.memory._instance.iteration_cache[self.memory._instance.cache_index])
                    if distances[min_index] < self.similarity_threshold:
                        recognized_faces[match_name.lower().capitalize()] = self.memory._instance.iteration_cache[self.memory._instance.cache_index]                       
                    self.memory._instance.cache_index += 1
            
            return recognized_faces
        except Exception as e:
            raise ValueError(f"Error during face recognition: {e}")

    def process_images(self, input_directory: str, output_directory: str, iteration: int):
        known_faces = load_data("rec_faces_dlib")
        for root, _, files in os.walk(input_directory):
            for file in files:
                image_path = os.path.join(root, file)
                try:
                    image = load_image(image_path)
                    if image is None:
                        continue
                    detected_faces = self._recognize_faces(image_path, known_faces, iteration)
                    if iteration == 2:
                        draw_bounding_boxes(detected_faces, output_directory=output_directory)
                    elif iteration == 1:
                        save_cropped_faces(detected_faces, image=image, output_directory=output_directory)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

class SuperResolutionFaceDetector(FaceDetector):
    def __init__(self, predictor_model_path: str, face_recognition_model_path: str, similarity_threshold: float = 0.6):
        super().__init__(predictor_model_path, face_recognition_model_path, similarity_threshold)

    def process_images(self, input_directory: str, output_directory: str, iteration: int, batch_command: str):
        super().process_images(input_directory, output_directory, iteration)
        subprocess.call(batch_command, shell=True)


class DlibFaceDetector(FaceDetector):
    def __init__(self, predictor_model_path: str, face_recognition_model_path: str, similarity_threshold: float = 0.6):
        super().__init__(predictor_model_path, face_recognition_model_path, similarity_threshold)

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

                    resized_image = self._resize_image(image, target_width=150)
                    detected_faces = self.face_detector(resized_image, 1)

                    for detected_face in detected_faces:
                        facial_landmarks = self.shape_predictor(resized_image, detected_face)
                        face_descriptor = np.array(self.face_recognizer.compute_face_descriptor(resized_image, facial_landmarks), dtype=np.float64)[np.newaxis, :]
                        known_faces[face_name.lower().capitalize()] = face_descriptor

            save_data(known_faces, "rec_faces_dlib")
        except Exception as e:
            raise ValueError(f"Error extracting known faces: {e}")
