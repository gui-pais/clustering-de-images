from .detctor import *

class DetectorFactory:
    @staticmethod
    def create_detector(detector_type: str, **kwargs) -> FaceDetector:
        predictor_model_path = kwargs.get("predictor_model_path")
        face_recognition_model_path = kwargs.get("face_recognition_model_path")
        similarity_threshold = kwargs.get("similarity_threshold", 0.6)
        
        if not predictor_model_path or not face_recognition_model_path:
            raise ValueError("Both predictor and face recognition models are required.")
        
        match detector_type:
            case "dlib":
                return DlibFaceDetector(predictor_model_path, face_recognition_model_path, similarity_threshold)

            case "super_resolution":
                return SuperResolutionFaceDetector(predictor_model_path, face_recognition_model_path, similarity_threshold)
            case _:
                raise ValueError(f"Unknown detector type: {detector_type}")