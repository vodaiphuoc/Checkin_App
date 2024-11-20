from typing import List, Union, Tuple, Literal
from src.firebase import Firebase_Handler
from src.mongodb import Mongo_Handler
from src.server.Inference import EmbeddingModel
import numpy as np
import time

class Checking_Engine(object):
    def __init__(self, 
                 master_config: dict,
                 running_mode: Literal['init_push','checkin'],
                 db_handler: Union[Firebase_Handler, Mongo_Handler]
                 ) -> None:
        print("Init Checking Engine")
        
        self.model = EmbeddingModel(use_lite_model= False,
                                    detector_model_name= master_config['detetor_name'], 
                                    reg_model_name= master_config['reg_model_name'],
                                    running_mode = running_mode
                                    )
        
        self.db_handler = db_handler
    
    def __call__(self, 
                 input_images: List[np.ndarray], 
                 return_embeddings_only: bool = False
                 ) -> Union[ bool, str, Tuple[str,np.ndarray]]:
        forward_results = self.model.forward(input_image = input_images)
        if isinstance(forward_results, bool):
            return False
        else:
            new_embeddings, is_batch = forward_results
            if return_embeddings_only:
                return new_embeddings
            else:
                if not is_batch:
                    new_embeddings = np.expand_dims(new_embeddings, axis= 0)
                return self.db_handler.searchUserWithEmbeddings(
                    batch_query_embeddings= new_embeddings
                )
