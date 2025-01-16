from .detctor import *

class DetectorFactory:
    @staticmethod
    def create_detector(detector_type: str, **kwargs) -> IDetector:
        match detector_type:
            
            case "dlib":
                predictor_model = kwargs.get("predictor_model")
                model_describer = kwargs.get("model_describer")
                threshold = kwargs.get("threshold")
                if not predictor_model or not model_describer:
                    raise ValueError("Modelos de predição e descrição são necessários para o DlibDetector.")
                return DlibDetector(predictor_model, model_describer, threshold)
            
            case "retina":
                model_name = kwargs.get("model_name")
                detector_backend = kwargs.get("detector_backend")
                if not model_name or not detector_backend:
                    raise ValueError("Modelo e backend são necessários para o RetinaFaceDetector.")
                return RetinaFaceDetector(model_name, detector_backend)
            case "embeddings":
                predictor_model = kwargs.get("predictor_model")
                model_describer = kwargs.get("model_describer")
                if not predictor_model or not model_describer:
                    raise ValueError("Modelos de predição e descrição são necessários para o DlibDetector.")
                return DelibEmbeddings(predictor_model, model_describer)
            case _:
                raise ValueError(f"Tipo de detector desconhecido: {detector_type}")