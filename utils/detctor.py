import os
import dlib
import numpy as np
from re import search
from subprocess import call
from ultralytics import YOLO as YOLOv10
from time import time
from .cache import save_data, load_data
from .image import load_image, save_cropped_faces, draw_bounding_boxes

class SharedMemory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedMemory, cls).__new__(cls)
            cls._instance.iteration_cache = []
        return cls._instance

class FaceDetector:
    def __init__(self, predictor_model_path: str, face_recognition_model_path: str, similarity_threshold: float = 0.6):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)
        self.face_recognizer = dlib.face_recognition_model_v1(face_recognition_model_path)
        self.similarity_threshold = similarity_threshold
        self.memory = SharedMemory()

    def _calculate_distances(self, known_faces: dict, image, detected_face) -> tuple:
        facial_landmarks = self.shape_predictor(image, detected_face)
        face_descriptor = np.asarray(self.face_recognizer.compute_face_descriptor(image, facial_landmarks), dtype=np.float64)[np.newaxis, :]

        distances = [np.linalg.norm(face_descriptor - known_descriptor) for known_descriptor in known_faces.values()]
        min_index = np.argmin(distances)
        match_name = list(known_faces.keys())[min_index]

        return match_name, distances, min_index
    
    def process_images(self, input_directory: str, output_directory: str, func_save: function):
        known_faces = load_data("rec_faces_dlib")
        for root, _, files in os.walk(input_directory):
            start1 = time()
            for file in files:
                start2 = time()
                image_path = os.path.join(root, file)
                try:
                    image = load_image(image_path)
                    if image is None:
                        continue
                    detected_faces = self._recognize_faces(image_path, known_faces)
                    func_save(detected_faces, image=image, output_directory=output_directory)
                    print(f"Tempo de execução para a imagem {image_path}: {time() - start2}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
            print(f"Tempo de execução: {time() - start1}")

class DlibFaceDetector(FaceDetector):
    def __init__(self, predictor_model_path: str, face_recognition_model_path: str, similarity_threshold: float = 0.6):
        super().__init__(predictor_model_path, face_recognition_model_path, similarity_threshold)
        
    def _recognize_faces(self, image_path: str, known_faces: dict) -> dict:
        recognized_faces = {}
        known_faces_copy = known_faces.copy()
        try:
            image = load_image(image_path)

            detected_faces = self.face_detector(image, 1)
            
            print(f"Para a imagem {image_path} foi detectado {len(detected_faces)}")
            for detected_face in detected_faces:
                match_name, distances, min_index = self._calculate_distances(known_faces_copy, image, detected_face)
                print(f"Na primeira iteração foi encontrado o aluno {match_name} com {distances[min_index]}")
                if distances[min_index] < self.similarity_threshold:
                    del known_faces_copy[match_name]
                    recognized_faces[match_name] = detected_face
                    bounding_boxes = (image_path, detected_face)
                    self.memory._instance.iteration_cache.append(bounding_boxes)

            return recognized_faces
        except Exception as e:
            raise ValueError(f"Error during face recognition: {e}")

    def process_images(self, input_directory: str, output_directory: str, batch_command: str):
        super().process_images(input_directory, output_directory, save_cropped_faces)
        start = time()
        call(batch_command, shell=True)
        print(f"Tempo para a rede generativa gerar as imagens de super resolução: {time() - start}\nQuantidade de imagens para esse tempo {sum(len(files) for _, _, files in os.walk("recognized"))}")


class SuperResolutionFaceDetector(FaceDetector):
    def __init__(self, predictor_model_path: str, face_recognition_model_path: str, similarity_threshold: float = 0.6):
        super().__init__(predictor_model_path, face_recognition_model_path, similarity_threshold)
        
    def _recognize_faces(self, image_path: str, known_faces: dict) -> dict:
        recognized_faces = {}
        known_faces_copy = known_faces.copy()
        try:
            image = load_image(image_path)
            if len(self.memory._instance.iteration_cache) > index_match:
                detected_face = dlib.rectangle(0, 0, image.shape[1], image.shape[0])
                pattern = r"(\d+)"
                index_match = search(pattern, image_path)
                index_match = int(index_match.group(1)) - 1
                print(image_path)
                match_name, distances, min_index = self._calculate_distances(known_faces_copy, image, detected_face)
                print(f"Na segunda iteração foi encontrado o aluno {match_name} com {distances[min_index]} na posição {index_match}")
                if distances[min_index] < self.similarity_threshold:
                    recognized_faces[match_name.lower().capitalize()] = self.memory._instance.iteration_cache[index_match]

            return recognized_faces
        except Exception as e:
            raise ValueError(f"Error during face recognition: {e}")
        
    def process_images(self, input_directory: str, output_directory: str):
        super().process_images(input_directory, output_directory, draw_bounding_boxes)

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
        
class YoloFaceDetector(FaceDetector):
    def __init__(self, face_detection_model_path, predictor_model_path, face_recognition_model_path, similarity_threshold = 0.6):
        super().__init__(predictor_model_path, face_recognition_model_path, similarity_threshold)
        self.face_detector = YOLOv10(face_detection_model_path)
        
    def _recognize_faces(self, image_path: str, known_faces: dict) -> dict:
        recognized_faces = {}
        known_faces_copy = known_faces.copy()
        try:
            image = load_image(image_path)

            detected_faces = self.face_detector(image)[0]
            
            print(f"Para a imagem {image_path} foi detectado {len(detected_faces)}")
            for detected_face in detected_faces.boxes:
                cords = detected_face.xywh[0].cpu().numpy().tolist()
                x, y, w, h = cords
                detected_face = dlib.rectangle(x, y, w, h)
                match_name, distances, min_index = self._calculate_distances(known_faces_copy, image, detected_face)
                print(f"Na primeira iteração foi encontrado o aluno {match_name} com {distances[min_index]}")
                if distances[min_index] < self.similarity_threshold:
                    del known_faces_copy[match_name]
                    recognized_faces[match_name] = detected_face
                    bounding_boxes = (image_path, detected_face)
                    self.memory._instance.iteration_cache.append(bounding_boxes)

            return recognized_faces
        except Exception as e:
            raise ValueError(f"Error during face recognition: {e}")
        
    def process_images(self, input_directory: str, output_directory: str, batch_command: str):
        super().process_images(input_directory, output_directory, save_cropped_faces)
        start = time()
        call(batch_command, shell=True)
        print(f"Tempo para a rede generativa gerar as imagens de super resolução: {time() - start}\nQuantidade de imagens para esse tempo {sum(len(files) for _, _, files in os.walk("recognized"))}")