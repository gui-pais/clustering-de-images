import os
import cv2
import dlib
import numpy as np
from abc import ABC, abstractmethod
from time import time
from retinaface import RetinaFace
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
    def __init__(self, threshold:float = 0.6):
        self._detector = RetinaFace
        self._model = FaceNet()
        self._threshold = threshold
        
    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))
        img = img.astype('float32') / 255.0
        return img


    def _predict(self, image: np.ndarray, recognized_faces: dict) -> dict:
        persons_recognized = {}
        recognized_faces_copy = recognized_faces.copy()
        try:
            image = self._preprocess(image)
            detections = self._detector.detect_faces(image)

            for _, detection in detections.items():
                x, y, w, h = detection['facial_area']
                cropped_face = image[y:h, x:w]

                if cropped_face.size == 0:
                    continue

                embeddings = self._model.embeddings([cropped_face])


                min_distance = float('inf')
                recognized_name = None

                for name, ref_embedding in recognized_faces_copy.items():
                    distance = np.linalg.norm(embeddings[0] - ref_embedding)
                    if distance < min_distance and distance < self._threshold:
                        min_distance = distance
                        recognized_name = name                       

                if recognized_name:
                    del recognized_faces_copy[recognized_name]
                    persons_recognized[recognized_name.lower().capitalize()] = (x, y, w, h)

            return persons_recognized

        except Exception as e:
            raise ValueError(f"Erro ao realizar predição com RetinaFace: {e}")
        
    def extract_faces(self, group_dir="faces"):
        recognized_faces = {}
        try:
            if os.path.exists(group_dir):
                for file in os.listdir(group_dir):
                    name = file.split(".")[0]
                    image_path = os.path.join(group_dir, file)
                    image = get_valid_image(image_path)
                    if image is None:
                        continue
                    image = self._preprocess(image)
                    detections = self._detector.detect_faces(image)
                    if not detections:
                        print(f"Nenhuma face detectada em: {image_path}")
                        continue

                    for detection in detections:
                        x, y, w, h = detection['facial_area']
                        cropped_face = image[y:h, x:w]

                        if cropped_face.size == 0:
                            continue

                        embeddings = self._model.embeddings([cropped_face])

                        recognized_faces[name.lower().capitalize()] = embeddings[0]

            save(recognized_faces, "recognized_faces_retina")
        except Exception as e:
            raise ValueError(f"Erro ao criar faces reconhecidas: {e}")

    def run(self, images_path: str):
        for root, _, files in os.walk(images_path):
            start_time = time()
            for file in files:
                image_path = os.path.join(root, file)
                try:
                    recognized_faces = load("recognized_faces_retina")
                    image = get_valid_image(image_path)
                    persons_recognized = self._predict(image, recognized_faces)
                    save_cropped_faces(image, persons_recognized)
                except Exception as e:
                    print(f"Erro ao processar {file}: {e}")
            print(f"Tempo total: {time() - start_time:.2f} segundos")
            break