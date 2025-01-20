from .detctor import *

class DetectorFactory:
    @staticmethod
    def create_detector(detector_type: str, **kwargs) -> Detector:
        
        predictor_model = kwargs.get("predictor_model")
        model_describer = kwargs.get("model_describer")
        threshold = kwargs.get("threshold")
        if not predictor_model or not model_describer:
            raise ValueError("Modelos de predição e descrição são necessários para o DlibDetector.")
        
        match detector_type:
            case "dlib":
                return DlibDetector(predictor_model, model_describer, threshold)

            case "super_resolution":
                return SuperResolutionDetector(predictor_model, model_describer, threshold)
            case _:
                raise ValueError(f"Tipo de detector desconhecido: {detector_type}")