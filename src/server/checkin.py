from typing import Any
from src.firebase import Firebase_Handler
from src.server.Inference import EmbeddingModel
import numpy as np
from src.utils import get_program_config
import time

class Checking_Engine(object):
    def __init__(self) -> None:
        master_config = get_program_config()
        self.model = EmbeddingModel(detector_model_name= master_config['detetor_name'], 
                                    reg_model_name= master_config['reg_model_name']
                                    )
        
        self.db_handler = Firebase_Handler(master_cfg= master_config)
        self.dataset = self.db_handler.get_dataset()
        print('start at :',time.time())
    
    def __call__(self, input_image: np.ndarray, user_name:str) -> str:
        new_embeddings = self.model.forward(input_image=input_image)

        results = {}
        for user_name, registered_embedding in self.dataset:
            score = np.dot(new_embeddings, registered_embedding)
            if results.get(user_name) is None:
                results[user_name] = []
            results[user_name].append(score)
        
        scores = {}
        for user, list_score in results.items():
            mean_score = sum(list_score)/len(list_score)
            scores[user] = mean_score
        
        sorted_score = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

        detect_user_name = list(sorted_score.keys())[-1]
        self.db_handler.insert(user_name= user_name, image= input_image, embedding= new_embeddings)
        return detect_user_name
