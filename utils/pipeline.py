import logging
from pathlib import Path
from time import time
from .detector_factory import DetectorFactory

logging.basicConfig(level=logging.INFO)

def pipeline(batch_command="GFPGAN.bat"):
    start_time = time()
    
    # face_detection_model_path = "media/faceTest3.pt"
    predictor_model_path = "media/shape_predictor_68_face_landmarks.dat"
    face_recognition_model_path = "media/dlib_face_recognition_resnet_model_v1.dat"
    known_faces_file = Path("rec_faces_dlib.pkl")
    uploads_dir = Path("uploads")
    recognized_dir = Path("recognized")
    output_dir = Path("output")

    dlib_detector = DetectorFactory.create_detector(
        "dlib",
        predictor_model_path=predictor_model_path,
        face_recognition_model_path=face_recognition_model_path,
        similarity_threshold=0.61,
    )
    
    # yolo_detector = DetectorFactory.create_detector(
    #     "Yolo",
    #     face_detection_model_path,
    #     predictor_model_path=predictor_model_path,
    #     face_recognition_model_path=face_recognition_model_path,
    #     similarity_threshold=0.61,
    # )
    
    super_resolution_detector = DetectorFactory.create_detector(
        "super_resolution",
        predictor_model_path=predictor_model_path,
        face_recognition_model_path=face_recognition_model_path,
        similarity_threshold=0.58,
    )

    if not known_faces_file.exists():
        logging.info("Extraindo faces conhecidas...")
        super_resolution_detector.extract_known_faces()

    logging.info("Transformando imagens com super-resolução...")
    dlib_detector.process_images(
        input_directory=uploads_dir,
        output_directory=recognized_dir,
        batch_command=batch_command,
    )
    
    logging.info("Refinando detecção...")
    super_resolution_detector.process_images(
        input_directory=recognized_dir / "restored_imgs",
        output_directory=output_dir,
    )

    logging.info(f"Tempo total de execução: {time() - start_time:.2f} segundos")