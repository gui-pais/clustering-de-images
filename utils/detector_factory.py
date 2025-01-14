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
                threshold = kwargs.get("threshold")
                return RetinaFaceDetector(threshold)
            case _:
                raise ValueError(f"Tipo de detector desconhecido: {detector_type}")