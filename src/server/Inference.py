from src.server.models.Deepface_implement import (
    VGGFace,
    Facenet,
    ArcFace
)
from mtcnn import MTCNN
from src.utils import download_pretrained_weights
import tensorflow as tf
# from tensorflow import keras
from typing import Tuple, Union, List, Callable, Literal, Dict, Any
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
                 use_lite_model:bool,
                 detector_model_name: Literal['MTCNN'],
                 reg_model_name: str,
                 camera_standard_shape: tuple = (368, 640),
                #  face_standard_shape: tuple = (160,160)
                 ) -> None:

        self.face_target_size = ModelBase.configs.get(reg_model_name).get("resize")
        self.camera_standard_shape = camera_standard_shape
        self.face_standard_shape = self.configs[reg_model_name]['resize']

        # get detector
        self.detector_forward = ModelBase.build_detector(detector_name= detector_model_name)
        
        # build and load weight of regconition model
        build_results = ModelBase.build_recognition(reg_model_name, use_lite_model)
        if use_lite_model:
            self.face_reg_instance, self.model_out_specs, self.model_in_specs = build_results
            self.reg_inference = self.recog_graph_inference_lite
        else:
            self.face_reg_instance = build_results
            self.reg_inference = self.recog_graph_inference

    @classmethod
    def build_recognition(cls,model_name: str, use_lite_model:bool):
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
        if use_lite_model:
            interpreter = tf.lite.Interpreter(
                model_path="src\server\models\pretrained_weights\lite_version\\facenet.tflite",
                model_content=None,
                experimental_delegates=None,
                num_threads=1,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO,
                experimental_preserve_all_tensors=False,
                experimental_disable_delegate_clustering=False,
                experimental_default_delegate_latest_features=False
            )
            interpreter.allocate_tensors()  # Needed before execution!
            model_out_specs = interpreter.get_output_details()[0]  # Model has single output.
            model_in_specs = interpreter.get_input_details()[0]  # Model has single input.
            return interpreter, model_out_specs, model_in_specs

        else:
            status, weigth_path = download_pretrained_weights(recog_model_name= model_name)
            model.load_weights(weigth_path)
            print("Model is ready for serving")
            return model

    @classmethod
    def build_detector(cls, 
                       detector_name: Literal['MTCNN']
                       )-> Tuple[Union[MTCNN], Callable]:
        if detector_name == "MTCNN":
            return MTCNN(stages = 'face_and_landmarks_detection').detect_faces
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 160, 160,3), 
                                                dtype=tf.float32)])
    def recog_graph_inference(self, data: tf.Tensor)-> List[tf.Tensor]:
        # face recongition
        data =  self.face_reg_instance(data, training = False)
        # normalize embeddings
        normalized_data , old_norm = tf.linalg.normalize(data, axis = 1)
        return normalized_data
    
    def recog_graph_inference_lite(self, data: tf.Tensor)-> List[tf.Tensor]:
        self.face_reg_instance.set_tensor(self.model_in_specs['index'], data)
        self.face_reg_instance.invoke()
        # face recongition
        data_out  = self.face_reg_instance.get_tensor(self.model_out_specs['index'])
        # normalize embeddings
        normalized_data , old_norm = tf.linalg.normalize(data_out, axis = 1)
        return normalized_data

class EmbeddingModel(ModelBase):
    def __init__(self,
                 use_lite_model:bool,
                 detector_model_name: str, 
                 reg_model_name: str, 
                 running_mode: str,
                #  face_standard_shape: Tuple = (160, 160)
                 ) -> None:
        super().__init__(use_lite_model = use_lite_model,
                         detector_model_name = detector_model_name,
                         reg_model_name = reg_model_name, 
                        #  face_standard_shape = face_standard_shape
                         )
        self.running_mode = running_mode

    def resize_cam_img(self, input_image: Union[np.ndarray, List[np.ndarray]])->np.ndarray:
        if isinstance(input_image, list):
            resized_img = [cv2.resize(src = img, 
                                      dsize = self.camera_standard_shape
                                      )
                            for img in input_image
            ]

            return ([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) \
                        if self.running_mode == 'checkin' else img
                        for img in resized_img
                    ],
                    True)
        else:
            return (cv2.resize(input_image, self.camera_standard_shape),
                    False)

    def _gather_faces_from_a_result(self, 
                                    result: List[Dict[str,Any]],
                                    image: np.ndarray,
                                    delta: int = 50
                                    )->List[np.ndarray]:
        """Use for an image"""
        boxes = [each_result['box'] for each_result in result
                        if each_result['confidence'] > 0.7]
                
        detected_faces = [image[box[1]-delta:box[1]+box[3]+delta,
                                box[0]-delta:box[0]+box[2]+delta] 
                        if box[1]-delta > 0 and box[0]-delta>0 \
                        else image[0:box[1]+box[3]+delta,
                                0:box[0]+box[2]+delta] 
                        for box in boxes]
        detected_faces = [cv2.resize(face, self.face_standard_shape)
                        for face in detected_faces]
        return detected_faces

    def forward(self, input_image: Union[np.ndarray, List[np.ndarray]]
                )-> Union[Tuple[np.ndarray, bool], bool]:
        """"Main pipeline use for inference operation
        Returns:
            embeddings: float32 dtype numpy array,
            is_batch: if input image is a batch of image or not
        """
        # stage 0: resize input image
        input_image, is_batch = self.resize_cam_img(input_image)
        # stage 1: get faces
        results = self.detector_forward(input_image, 
                                        limit_boundaries_landmarks = True,
                                        threshold_pnet = 0.75,
                                        threshold_rnet = 0.75)
        
        # if is batch, 'results' is [[img1_dict1, img1_dict2, ...,img1_dictM],
        #                            [img2_dict1, img2_dict2, ...,img2_dictM],
        #                            [img3_dict1, img3_dict2, ...,img3_dictM],
        # ...
        #                            [imgN_dict1, imgN_dict2, ...,imgN_dictM]]
        # with N is batch value, M is number of faces in an image
        if is_batch:
            total_faces = []
            assert len(results) == len(input_image), f"not equal length"
            for each_img_results, each_input_img in zip(results, input_image):
                if len(each_img_results) == 0:
                    continue
                else:
                    faces_in_a_img = self._gather_faces_from_a_result(each_img_results, 
                                                                      each_input_img)
                    total_faces.extend(faces_in_a_img)
            
            if len(total_faces) != 0:
                detected_faces = np.stack(total_faces, axis = 0)
                detected_faces_tf = tf.convert_to_tensor(detected_faces, dtype= tf.float32)
                embeddings = self.reg_inference(detected_faces_tf)
                return embeddings.numpy().astype(np.float32), is_batch, total_faces
            else:
                return False

        else:
            # single image
            if len(results) != 0:
                # stage 2:
                detected_faces = self._gather_faces_from_a_result(results, input_image)
                detected_faces = np.stack(detected_faces, axis = 0)
                detected_faces_tf = tf.convert_to_tensor(detected_faces, dtype= tf.float32)
                embeddings = self.reg_inference(detected_faces_tf)
                return embeddings.numpy().astype(np.float32), is_batch, results
            else:
                # this case is for no face detected
                return False
