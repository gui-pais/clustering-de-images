import dlib
import cv2
import os
import numpy as np
from time import time
import pickle
from collections import namedtuple

def get_valid_image(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Imagem não encontrada: {image_path}")
            return None
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Formato de imagem inválido: {image_path}")
            return None
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erro ao carregar a imagem: {image_path}")
            return None
        return img
    except Exception as e:
        raise ValueError(f"Erro ao obter a imagem: {e}")

def get_faces(image, unique=False):
    try:
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 1)
        if not faces:
            print("Nenhum rosto detectado.")
        return faces[0] if unique and faces else faces
    except Exception as e:
        raise ValueError(f"Erro ao detectar rostos: {e}")

def get_points(image, face, predictor_model="media/shape_predictor_68_face_landmarks.dat"):
    try:
        predictor = dlib.shape_predictor(predictor_model)
        return predictor(image, face)
    except Exception as e:
        raise ValueError(f"Erro ao obter pontos faciais: {e}")

def get_descriptors(image, shape, model_describer="media/dlib_face_recognition_resnet_model_v1.dat"):
    try:
        describer = dlib.face_recognition_model_v1(model_describer)
        descriptors = describer.compute_face_descriptor(image, shape)
        return np.array(descriptors, dtype=np.float64)[np.newaxis, :]
    except Exception as e:
        raise ValueError(f"Erro ao obter descritores faciais: {e}")

def predict(image, recognized_faces, method=3):
    recognized_faces_copy = recognized_faces.copy()
    persons_recognized = {}
    try:
        faces = get_faces(image)      
        for face in faces:
            match method:
                case 1:
                    points = get_points(image, face)
                    desc = get_descriptors(image, points)
                    for name, descriptor in recognized_faces.items():
                        if np.linalg.norm(desc - descriptor) < 0.6:
                            persons_recognized[name] = face
                case 2:
                    Person = namedtuple("Person", ["name", "face", "distance"])
                    distances = []
                    points = get_points(image, face)
                    desc = get_descriptors(image, points)
                    for name, descriptor in recognized_faces.items():
                        distances.append(Person(name, face, np.linalg.norm(descriptor - desc)))
                    min_distance = min(distances, key=lambda x: x.distance)
                    persons_recognized[min_distance.name] = min_distance.face
                case 3:
                    points = get_points(image, face)
                    desc = get_descriptors(image, points)
                    min_index = np.argmin([np.linalg.norm(desc - descriptor) for descriptor in recognized_faces_copy.values()])
                    name = list(recognized_faces_copy.keys())[min_index]
                    if np.linalg.norm(desc - recognized_faces_copy[name]) < 0.58888888888888888888:
                        del recognized_faces_copy[name]
                        persons_recognized[name] = face
                            
        return persons_recognized
    except Exception as e:
        raise ValueError(f"Erro ao realizar predição: {e}")
    
def get_recognized_faces(group_dir="points"):
    recognized_faces = {}
    try:
        if os.path.exists(group_dir):
            for file in os.listdir(group_dir):
                name = file.split(".")[0]
                image_path = os.path.join(group_dir, file)
                image = get_valid_image(image_path)
                if image is None:
                    continue
                faces = get_faces(image, unique=True)
                if faces is None:
                    continue
                shape = get_points(image, faces)
                descriptor = get_descriptors(image, shape)
                recognized_faces[name] = descriptor
        return recognized_faces
    except Exception as e:
        raise ValueError(f"Erro ao criar faces reconhecidas: {e}")

def save_recognized_faces(save_name):
    try:
        start_time = time()
        recognized_faces = get_recognized_faces()
        print(f"Tempo para obter as faces reconhecidas: {time() - start_time:.2f} segundos")
        with open(f'{save_name}.pkl', 'wb') as f:
            pickle.dump(recognized_faces, f)
    except Exception as e:
        raise ValueError(f"Erro ao salvar faces reconhecidas: {e}")

def load_recognized_faces():
    try:
        with open('dict_file.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise ValueError(f"Erro ao carregar faces reconhecidas: {e}")

def split_image(image_path, output_dir="split_images"):
    try:
        image = get_valid_image(image_path)
        if image is None:
            return
        
        height, width, _ = image.shape
        if width != 3840 or height != 2160:
            print(f"A imagem não tem o tamanho esperado de 3840x2160 pixels. Tamanho atual: {width}x{height}")

        img1 = image[0:1080, 0:1920]
        img2 = image[0:1080, 1920:3840]
        img3 = image[1080:2160, 0:1920]
        img4 = image[1080:2160, 1920:3840]

        os.makedirs(output_dir, exist_ok=True)

        for i, img in enumerate([img1, img2, img3, img4], 1):
            cv2.imwrite(os.path.join(output_dir, f'img{i}.jpg'), img)
    except Exception as e:
        raise ValueError(f"Erro ao dividir a imagem: {e}")

def save_cropped_faces(image, persons_recognized, output_dir="faces_recognizeds"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (name, face) in enumerate(persons_recognized.items(), start=1):
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cropped_face = image[y:y+h, x:x+w]
            output_path = os.path.join(output_dir, f"{name}_{i}.jpg")
            cv2.imwrite(output_path, cropped_face)
    except Exception as e:
        raise ValueError(f"Erro ao salvar faces recortadas: {e}")

def draw_faces(image):
    try:
        faces = get_faces(image)
        for face in faces:
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    except Exception as e:
        raise ValueError(f"Erro ao desenhar faces: {e}")
    
def draw_points(image):
    try:
        faces = get_faces(image)
        for face in faces:
            shape = get_points(image, face)
            for i in range(68):
                x, y = shape.part(i).x, shape.part(i).y
                cv2.circle(face, (x, y), 2, (0, 0, 255), -1)
    except Exception as e:
        raise ValueError(f"Erro ao desenhar pontos: {e}")

def process_image(image_path):
    try:
        start_time = time()  
        recognized_faces = load_recognized_faces()
        image = get_valid_image(image_path)
        persons_recognized = predict(image, recognized_faces, method=3)
        
        print(f"Tempo para reconhecer as faces: {time() - start_time:.2f} segundos")
        
        save_cropped_faces(image, persons_recognized)
        
    except Exception as e:
        raise ValueError(f"Erro ao processar a imagem: {e}")

def recognized(images_path):
    for root, _, files in os.walk(images_path):
        count = len(files)
        start_time = time()
        for file in files:
            image_path = os.path.join(root, file)
            process_image(image_path)
        end_time = time() - start_time
        print(f"faces reconhecidas por minuto {count/(end_time/60)}")
        print(f"Tempo para reconhecer as faces gerais: {time() - start_time:.2f} segundos")
        break
    