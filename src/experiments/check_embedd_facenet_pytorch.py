from src.server.models.Facenet_pytorch.mtcnn import MTCNN, fixed_image_standardization
from src.server.models.Facenet_pytorch.inception_resnet_v1 import InceptionResnetV1
from src.mongodb import Mongo_Handler
from src.utils import get_program_config
import cv2 as cv
from typing import List, Literal
import numpy as np
import glob
import torch
import uuid
from tqdm import tqdm
import re
import os

class Test_Embeddings(object):
	def __init__(self,
				data_folder_path: str, 
				pretrained_weight_dir: str,
				model_string: Literal['casia-webface','fine_tuning'],
				# p_state_dict_path: str,
                # r_state_dict_path: str,
                # o_state_dict_path: str,
				):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# self.detector = MTCNN(image_size=160, 
        #             margin=0, 
        #             min_face_size=20,
        #             thresholds=[0.6, 0.7, 0.7], 
        #             factor=0.709, 
        #             post_process=True,
        #             select_largest=True, 
        #             selection_method=None, 
        #             keep_all=True,
        #             device=self.device,
        #             p_state_dict_path = p_state_dict_path,
        #             r_state_dict_path = r_state_dict_path,
        #             o_state_dict_path = o_state_dict_path,
        #             )
		self.recognition_model = InceptionResnetV1(pretrained = model_string, 
			    									classify=False, 
			    									num_classes=None, 
			    									dropout_prob=0.6,
			    									device=self.device,
			    									pretrained_weight_dir = pretrained_weight_dir
	    									)
		self.data_folder_path = data_folder_path

	def _run_single_user(self, 
						user_name:str, 
						return_embedding_only: bool = False
						):
		user_imgs = []
		for path in glob.glob(f"{self.data_folder_path}/{user_name}/*"):
			image = cv.imread(path)
			assert image is not None, f"{path}"
			image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
			user_imgs.append(image)
		
		stack_faces = torch.tensor(np.stack(user_imgs)).permute(0,3,1,2)
		assert stack_faces.shape[0] == len(user_imgs) and stack_faces.shape[1] == 3
		stack_faces =fixed_image_standardization(stack_faces).to(self.device)
		embeddings = self.recognition_model(stack_faces)
		# assert embeddings.shape[0] == len(filtered_faces)
		assert embeddings.shape[1] == 512

		if return_embedding_only:
			return embeddings
		else:
			user_init_data = [{
				'user_name': user_name,
				'password': '123',
				'embedding': embeddings[id].tolist(),
			} 
			for id in range(embeddings.shape[0])
			]
			return user_init_data

	def _get_total_init_user_data(self):
		user_folders = glob.glob(f"{self.data_folder_path}/*")
		master_init_data = []

		for user_folder in tqdm(user_folders, total = len(user_folders)):
			user_name = os.path.split(user_folder)[-1].split('.')[0]
			user_init_data = self._run_single_user(user_name = user_name)
			master_init_data.extend(user_init_data)
		
		return master_init_data

	def pipelines(self, run_init_push: bool, evaluation: bool):
		master_config = get_program_config()
		
		if run_init_push:
			master_init_data = self._get_total_init_user_data()
			print('number init data: ',len(master_init_data))
			db_engine = Mongo_Handler(master_config= master_config,
									ini_push= True,
									init_data= master_init_data)
			

		if evaluation:
			db_engine = Mongo_Handler(master_config= master_config,
									ini_push= False)
			
			result = {}
			for main_user_dir in glob.glob(f"{self.data_folder_path}/*_*"):
				user_name = os.path.split(main_user_dir)[-1].split('.')[0]
				embeddings = self._run_single_user(user_name = user_name, 
													return_embedding_only = True)

				num_embeddings = embeddings.shape[0]
				step = int(num_embeddings)//3
				predict_name_list = []
				for embedd_idx in range(0, num_embeddings, step):
					query_embeddings = embeddings[embedd_idx: embedd_idx+step,:]
					pred_name = db_engine.searchUserWithEmbeddings(batch_query_embeddings = query_embeddings)
					predict_name_list.append(pred_name)

				result[user_name] = predict_name_list

			print(result)