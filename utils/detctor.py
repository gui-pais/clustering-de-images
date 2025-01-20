import os
import cv2
import dlib
import numpy as np
from .cache import save, load
from .image import get_valid_image, save_cropped_faces
import subprocess

class Detector():
    def __init__(self, predictor_model: str, model_describer: str, threshold: float = 0.6):
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

    def run(self, images_path: str, output_path: str):
        for root, _, files in os.walk(images_path):
            for file in files:
                image_path = os.path.join(root, file)
                try:
                    rec_faces = load("rec_faces_dlib")
                    image = get_valid_image(image_path)
                    embeddings_face = self._predict(image, rec_faces)
                    save_cropped_faces(image, embeddings_face, output_dir=output_path)
                except Exception as e:
                    print(f"Erro ao processar {file}: {e}")


class SuperResolutionDetector(Detector):
    def __init__(self, predictor_model: str, model_describer: str, threshold: float = 0.6):
        super().__init__(predictor_model=predictor_model, model_describer=model_describer, threshold=threshold)

    def run(self, images_path: str, output_path: str, bat):
        super().run(images_path, output_path)
        subprocess.call(bat, shell=True)

class DlibDetector(Detector):
    def __init__(self, predictor_model: str, model_describer: str, threshold:float = 0.6):
        super().__init__(predictor_model=predictor_model, model_describer=model_describer, threshold=threshold)
        
    def run(self, images_path, output_path):
        return super().run(images_path, output_path)
                    
    def extract_faces(self, group_dir="faces"):
        rec_faces = {}
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
                        rec_faces[name.lower().capitalize()] = np.array(descriptor, dtype=np.float64)[np.newaxis, :]
            save(rec_faces, "rec_faces_dlib")
        except Exception as e:
            raise ValueError(f"Erro ao criar faces reconhecidas: {e}")
