from dotenv import load_dotenv
import os
import gdown
from typing import Literal

def get_program_config()-> dict:
    load_dotenv()
    return {
        'dbURL': os.getenv('dbURL'),
        'storageBucketURL': os.getenv('storageBucketURL'),
        'project_id':os.getenv('project_id'),
        'detetor_name': os.getenv('detetor_name'),
        'reg_model_name': os.getenv('reg_model_name'),
        'account_config': {
                            "type": os.getenv("service_account_type"),
                            "project_id": os.getenv("service_account_type_project_id"),
                            "private_key_id": os.getenv("service_account_private_key_id"),
                            "private_key": os.getenv("service_account_private_key"), 
                            "client_email": os.getenv("client_email"), 
                            "client_id": os.getenv("client_id"), 
                            "auth_uri": os.getenv("auth_uri"), 
                            "token_uri": os.getenv("token_uri"), 
                            "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
                            "client_x509_cert_url": os.getenv("client_x509_cert_url"), 
                            "universe_domain": os.getenv("universe_domain") 
        }

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
