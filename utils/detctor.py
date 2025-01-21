import os
import cv2
import dlib
import numpy as np
import subprocess
from .cache import save_data, load_data
from .image import load_image, save_cropped_faces

class FaceDetector:
    def __init__(self, predictor_model_path: str, face_recognition_model_path: str, similarity_threshold: float = 0.6):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)
        self.face_recognizer = dlib.face_recognition_model_v1(face_recognition_model_path)
        self.similarity_threshold = similarity_threshold

    def _resize_image(self, image: np.ndarray, target_width: int) -> np.ndarray:
        target_height = int(image.shape[0] * (target_width / image.shape[1]))
        return cv2.resize(image, (target_width, target_height))

    def _recognize_faces(self, image: np.ndarray, known_faces: dict) -> dict:
        recognized_faces = {}
        try:
            detected_faces = self.face_detector(image, 1)
            known_faces_copy = known_faces.copy()

            for detected_face in detected_faces:
                facial_landmarks = self.shape_predictor(image, detected_face)
                face_descriptor = np.array(self.face_recognizer.compute_face_descriptor(image, facial_landmarks), dtype=np.float64)[np.newaxis, :]

                distances = [np.linalg.norm(face_descriptor - known_descriptor) for known_descriptor in known_faces_copy.values()]
                min_index = np.argmin(distances)
                match_name = list(known_faces_copy.keys())[min_index]

                if distances[min_index] < self.similarity_threshold:
                    del known_faces_copy[match_name]
                    recognized_faces[match_name.lower().capitalize()] = detected_face

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

                    detected_faces = self._recognize_faces(image, known_faces)
                    save_cropped_faces(image, detected_faces, output_dir=output_directory)

                except Exception as e:
                    print(f"Error processing {file}: {e}")

class SuperResolutionFaceDetector(FaceDetector):
    def __init__(self, predictor_model_path: str, face_recognition_model_path: str, similarity_threshold: float = 0.6):
        super().__init__(predictor_model_path, face_recognition_model_path, similarity_threshold)

    def process_images(self, input_directory: str, output_directory: str, batch_command: str):
        super().process_images(input_directory, output_directory)
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
