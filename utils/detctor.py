import os
import cv2
import dlib
import numpy as np
from abc import ABC, abstractmethod
from time import time
from retinaface import RetinaFace
from deepface import DeepFace
from .cache import save, load
from .image import get_valid_image, save_cropped_faces
from keras_facenet import FaceNet

class IDetector(ABC):
    @abstractmethod
    def extract_faces(self, dir: str):
        pass
    
    @abstractmethod
    def run(self, images_path: str):
        pass

class DlibDetector(IDetector):
    def __init__(self, predictor_model: str, model_describer: str, threshold:float = 0.6):
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(predictor_model)
        self._describer = dlib.face_recognition_model_v1(model_describer)
        self._threshold = threshold

    def _predict(self, image: np.ndarray, recognized_faces: dict) -> dict:
        persons_recognized = {}
        try:
            faces = self._detector(image, 1)
            recognized_faces_copy = recognized_faces.copy()

            for face in faces:
                points = self._predictor(image, face)
                desc = self._describer.compute_face_descriptor(image, points)
                desc = np.array(desc, dtype=np.float64)[np.newaxis, :]
                min_index = np.argmin([
                    np.linalg.norm(desc - descriptor)
                    for descriptor in recognized_faces_copy.values()
                ])
                name = list(recognized_faces_copy.keys())[min_index]
                if np.linalg.norm(desc - recognized_faces_copy[name]) < self._threshold:
                    del recognized_faces_copy[name]
                    persons_recognized[name.lower().capitalize()] = face
            return persons_recognized

        except Exception as e:
            raise ValueError(f"Erro ao realizar predição: {e}")
        
    def extract_faces(self, group_dir="faces"):
        recognized_faces = {}
        try:
            if os.path.exists(group_dir):
                for file in os.listdir(group_dir):
                    name = file.split(".")[0]
                    image_path = os.path.join(group_dir, file)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    faces = self._detector(image, 1)
                    if faces is None:
                        continue
                    for face in faces:
                        shape = self._predictor(image, face)
                        descriptor = self._describer.compute_face_descriptor(image, shape)
                        recognized_faces[name.lower().capitalize()] = np.array(descriptor, dtype=np.float64)[np.newaxis, :]
            save(recognized_faces, "recognized_faces_dlib")
        except Exception as e:
            raise ValueError(f"Erro ao criar faces reconhecidas: {e}")

    def run(self, images_path: str):
        for root, _, files in os.walk(images_path):
            start_time = time()
            for file in files:
                image_path = os.path.join(root, file)
                try:
                    recognized_faces = load("recognized_faces_dlib")
                    image = get_valid_image(image_path)
                    persons_recognized = self._predict(image, recognized_faces)
                    save_cropped_faces(image, persons_recognized)
                except Exception as e:
                    print(f"Erro ao processar {file}: {e}")
            print(f"Tempo total: {time() - start_time:.2f} segundos")
            break

class RetinaFaceDetector(IDetector):
    def __init__(self, model_name:str, detector_backend:str, threshold:float = 0.6):
        self._detector = RetinaFace
        self._model = DeepFace
        self._threshold = threshold
        self._model_name = model_name
        self._detector_backend = detector_backend

    def extract_faces(self, image):
        detections = self._detector.detect_faces(image)
        faces = {}
            for _, detection in detections.items():
                x, y, w, h = detection['facial_area']
                cropped_face = image[y:h, x:w]
                faces[] = cropped_face
                
        save_cropped_faces(image, faces)
        
    def _predict(self, image: np.ndarray) -> dict:
        persons_recognized = {}
        try:
            for img in os.listdir("all_faces"):
                dec = os.path.join("all_faces", img)
                for file in os.listdir("faces"):
                    face = os.path.join("faces", file)
                    result = self._model.verify(face, dec, model_name=self._model_name, detector_backend=self._detector_backend)
                    if result["verified"]:
                        persons_recognized[file.lower().captalize().split(".")[0]] = cv2.imread(dec)
                        break
                        
            return persons_recognized

        except Exception as e:
            raise ValueError(f"Erro ao realizar predição com RetinaFace: {e}")
        

    def run(self, images_path: str):
        for root, _, files in os.walk(images_path):
            start_time = time()
            for file in files:
                image_path = os.path.join(root, file)
                try:
                    image = get_valid_image(image_path)
                    persons_recognized = self._predict(image)
                    save_cropped_faces(image, persons_recognized)
                except Exception as e:
                    print(f"Erro ao processar {file}: {e}")
            print(f"Tempo total: {time() - start_time:.2f} segundos")
            break
