import os
from time import time
from .detector_factory import DetectorFactory

def pipeline(batch_command="GFPGAN.bat"):
    start_time = time()
    predictor_model_path = "media/shape_predictor_68_face_landmarks.dat"
    face_recognition_model_path = "media/dlib_face_recognition_resnet_model_v1.dat"

    super_resolution_detector = DetectorFactory.create_detector(
        "super_resolution",
        predictor_model_path=predictor_model_path,
        face_recognition_model_path=face_recognition_model_path,
        similarity_threshold=0.59,
    )

    dlib_detector = DetectorFactory.create_detector(
        "dlib",
        predictor_model_path=predictor_model_path,
        face_recognition_model_path=face_recognition_model_path,
        similarity_threshold=0.54,
    )

    if not os.path.exists("rec_faces_dlib.pkl"):
        dlib_detector.extract_known_faces()

    super_resolution_detector.process_images("uploads", "recognized", batch_command)
    dlib_detector.process_images("super_resolution/restored_faces", "output")
    print(f"Total execution time: {time() - start_time}")
    