import os
from time import time
from .detector_factory import DetectorFactory

def pipeline(bat="GFPGAN.bat"):
    start = time()
    predictor_model = "media/shape_predictor_68_face_landmarks.dat"
    model_describer = "media/dlib_face_recognition_resnet_model_v1.dat"
    threshold = 0.588888888888888
    
    super_resolution = DetectorFactory.create_detector(
        "super_resolution", 
        predictor_model=predictor_model, 
        model_describer=model_describer, 
        threshold=threshold
        )
    
    detector = DetectorFactory.create_detector(
        'dlib', 
        predictor_model=predictor_model, 
        model_describer=model_describer, 
        threshold=threshold
        )
    
    if not os.path.exists("rec_faces_dlib.pkl"):
        detector.extract_faces()
        
    super_resolution.run("uploads", "recognized", bat)
    
    detector.run(f"super_resolution/restored_faces", "output")
    print(f"Tempo total para executar o algoritmo: {time() - start}")
    