from src.server.models.Deepface_implement import (
    VGGFace,
    Facenet,
    ArcFace
)
from mtcnn import MTCNN
from src.utils import download_pretrained_weights
import tensorflow as tf
# from tensorflow import keras
from typing import Tuple, Union, List, Callable, Literal
import os
import numpy as np
import cv2
import uuid

class ModelBase(object):
    parent_weight_path = os.path.dirname(os.path.realpath(__file__))
    configs = {
            "VGG_Face": {"resize": (224, 224), 
                        "model": VGGFace.baseModel,
                        "weight_path": parent_weight_path+"\\models\\pretrained_weights\\vgg_face_weights.h5" 
                        },
            "Facenet": {"resize": (160, 160),
                        "dim": 128 ,
                        "model": Facenet.InceptionResNetV1,
                        "weight_path": parent_weight_path+"\\models\\pretrained_weights\\facenet_weights.h5"
                        },
            "Facenet512": {"resize": (160, 160),
                        "dim": 512,
                        "model": Facenet.InceptionResNetV1,
                        "weight_path": parent_weight_path+"\\models\\pretrained_weights\\facenet512_weights.h5"
                        },
            "ArcFace": {"resize": (112, 112),
                        "model": ArcFace.Get_ArcFace,
                        "weight_path": parent_weight_path+"\\models\\pretrained_weights\\arcface_weights.h5"
                        }
    }
    
    def __init__(self, 
                 detector_model_name: Literal['MTCNN'],
                 reg_model_name: str,
                 face_standard_shape: tuple = (160,160)
                 ) -> None:

        self.face_target_size = ModelBase.configs.get(reg_model_name).get("resize")
        self.face_standard_shape = face_standard_shape

        # get detector
        self.detector_forward = ModelBase.build_detector(detector_name= detector_model_name)
        
        # build and load weight of regconition model
        self.face_reg_instance = ModelBase.build_recognition(reg_model_name)

    @classmethod
    def build_recognition(cls,model_name: str):
        """Construct and load pretrained weights to face recognition model"""
        model_cfg = ModelBase.configs.get(model_name)
        model_make_func = model_cfg.get("model")
        model = None
        # get architecture
        if model_cfg.get("dim") is not None:
            model = model_make_func(model_cfg.get("dim"))
        else:
            model = model_make_func()
        # load weigths
        status, weigth_path = download_pretrained_weights(recog_model_name= model_name)
        assert status == True
        model.load_weights(weigth_path)
        # model.load_weights(model_cfg.get("weight_path"))
        print("Model is ready for serving")
        return model

    @classmethod
    def build_detector(cls, 
                       detector_name: Literal['MTCNN']
                       )-> Tuple[Union[MTCNN], Callable]:
        if detector_name == "MTCNN":
            return MTCNN().detect_faces
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,160,160,3), dtype=tf.float32)])
    def recog_graph_inference(self, data: tf.Tensor)-> List[tf.Tensor]:
        # face recongition
        data =  self.face_reg_instance(data, training = False)
        # normalize embeddings
        normalized_data , old_norm = tf.linalg.normalize(data, axis = 1)
        return normalized_data

class EmbeddingModel(ModelBase):
    def __init__(self, 
                 detector_model_name: str, 
                 reg_model_name: str, 
                 face_standard_shape: Tuple = (160, 160)
                 ) -> None:
        super().__init__(detector_model_name = detector_model_name,
                        reg_model_name = reg_model_name, 
                        face_standard_shape = face_standard_shape)

    def forward(self,input_image: np.ndarray)-> Union[np.ndarray, bool]:
        """"Main pipeline use for inference operation
        Returns:
            embeddings: float32 dtype numpy array
        """
        # stage 0: resize input image
        input_image = cv2.resize(input_image, (960,1280))
        # stage 1: get faces
        results = self.detector_forward(input_image)
        
        if len(results) != 0:
            # stage 2:
            boxes = [result['box'] for result in results
                     if result['confidence'] > 0.5]
            
            detected_faces = [input_image[box[1]:+box[1]+box[3],box[0]:box[0]+box[2]] 
                              for box in boxes]
            detected_faces = [cv2.resize(face, self.face_standard_shape) 
                              for face in detected_faces]
            
            detected_faces = np.stack(detected_faces, axis = 0)
            embeddings = self.recog_graph_inference(detected_faces)
            return embeddings.numpy()
        else:
            # this case is for no face detected
            return False
            
