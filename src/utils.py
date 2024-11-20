from dotenv import load_dotenv
import os
import gdown
from typing import Literal, List

from src.server.models.Facenet_pytorch.mtcnn import MTCNN
from src.server.models.Facenet_pytorch.inception_resnet_v1 import InceptionResnetV1

import cv2 as cv
import numpy as np
import glob



def get_program_config()-> dict:
    load_dotenv()
    return {
        'mongodb_name': os.getenv('mongodb_name'),
        'mongodb_pass': os.getenv('mongodb_pass'),
        'detetor_name': os.getenv('detetor_name'),
        'reg_model_name': os.getenv('reg_model_name'),
        'vector_emebd_dim':os.getenv('vector_emebd_dim'),
        'ngrok_auth_token': os.getenv('NGROK_AUTH_TOKEN'),
        'port': os.getenv('APPLICATION_PORT'),
        'https_server': os.getenv('HTTPS_SERVER'),
        'deploy_domain': os.getenv('DEPLOY_DOMAIN')
    }


def download_pretrained_weights(recog_model_name: Literal["Facenet", "VGG_Face","Facenet512", "ArcFace"]):
    """Download a single weight for recognition model"""
    model_dict = {"Facenet": "facenet_weights",
                  "VGG_Face": "vgg_face_weights",
                  "Facenet512": "facenet512_weights",
                  "ArcFace": "arcface_weights"
                  }
    # path of current file
    curr_path = os.path.dirname(os.path.realpath(__file__)) 
    target_path = curr_path + "\\server\\models\\pretrained_weights\\" + model_dict[recog_model_name] + ".h5"
    # check if the file already exist
    if os.path.isfile(target_path):
        print("The {} model has already donwloaded".format(target_path))
        return False, target_path
    else:
        print("The {} is downloaded in {}".format(model_dict[recog_model_name], target_path))
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/"+model_dict[recog_model_name]+".h5"
        gdown.download(url, target_path)
        return True, target_path


def converth5tolite(model_name:str, target_path:str):
    import tensorflow as tf
    from src.server.Inference import EmbeddingModel as EM
    embedding_model = EM(detector_model_name= "MTCNN", 
                                    reg_model_name= "Facenet"
                                    )
    recognition_model = embedding_model.face_reg_instance
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(recognition_model)
    tflite_model = converter.convert()

    # Save the model.
    with open(target_path+'/'+model_name+'.tflite', 'wb') as f:
        f.write(tflite_model)


def get_faces(camera_standard_shape = (368, 640)):
    detector = MTCNN(image_size=160, 
                    margin=0, 
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7], 
                    factor=0.709, 
                    post_process=True,
                    select_largest=True, 
                    selection_method=None, 
                    keep_all=True, 
                    device=None
                    )


    for user_folder in glob.glob('face_dataset\\images\\*'):
        user_name = user_folder.split('\\')[-1]

        user_img_list = []
        user_img_path_list = []
        for user_img_path in glob.glob(f'face_dataset\\images\\{user_name}\\*'):
            img = cv.imread(user_img_path) # RGB -> BGR
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, camera_standard_shape)
            user_img_list.append(img)

            save_path = user_img_path.replace('images','faces_only')
            user_img_path_list.append(save_path)

        if not os.path.isdir(f'face_dataset\\faces_only\\{user_name}'):
            os.mkdir(f'face_dataset\\faces_only\\{user_name}')


        # forward img to model need RGB image
        faces: List[np.ndarray] = None # List of image shape (3, img_Height, img_Weight)
        faces, probs = detector(img = user_img_list, 
                                save_path = user_img_path_list,
                                return_prob= True)
        
        assert probs.shape[0] == len(user_img_list), f"Found {probs.shape[0]} and  {len(user_img_list)}"


def resize_celeb_face(target_size = (160,160)):

    for celeb_user_folder in glob.glob('face_dataset\\VN-celeb\\*'):
        user_name = celeb_user_folder.split('\\')[-1]

        if not os.path.isdir(f'face_dataset\\faces_only\\{user_name}'):
            os.mkdir(f'face_dataset\\faces_only\\{user_name}')

        for user_img_path in glob.glob(f'face_dataset\\VN-celeb\\{user_name}\\*'):
            img = cv.imread(user_img_path) # RGB -> BGR
            if img.shape[-1] == 1:
                continue
            img = cv.resize(img, target_size)

            save_user_img_path = user_img_path.replace('VN-celeb','faces_only').replace('png','jpg')

            cv.imwrite(save_user_img_path, img) # BGR -> RGB