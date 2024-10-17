"""This file is only for local development"""
from src.server.Inference import EmbeddingModel
from src.firebase import Firebase_Handler
from src.utils import get_program_config
import cv2 as cv
import glob
from tqdm import tqdm

master_config = get_program_config()
model = EmbeddingModel(detector_model_name= master_config['detetor_name'], 
                                    reg_model_name= master_config['reg_model_name']
                                    )
engine = Firebase_Handler(master_cfg= master_config)


paths = glob.glob("face_dataset\images\*\*")
for path in tqdm(paths, total= len(paths)):
    user_name = path.split('\\')[-2]
    
    image = cv.imread(path)
    embeddings = model.forward(input_image= image)
    if isinstance(embeddings, bool):
        continue
    else:
        for i in range(embeddings.shape[0]):
            engine.insert(user_name= user_name, 
                        image= image, 
                        embedding= embeddings[i]
                        )