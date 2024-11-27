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
import json

class Test_Embeddings(object):
	def __init__(self,
				data_folder_path: str,
				pretrained_weight_dir: str,
				model_string: Literal['casia-webface','fine_tuning'],
				users_from_json: bool = False,
				):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.recognition_model = InceptionResnetV1(pretrained = model_string, 
													classify=False, 
													num_classes=None, 
													dropout_prob=0.6,
													device=self.device,
													pretrained_weight_dir = pretrained_weight_dir
	    									).to(self.device)
		self.recognition_model.eval()
		self.data_folder_path = data_folder_path
		self.users_from_json = users_from_json

	def _run_single_user(self, 
				user_name:str, 
				return_embedding_as_matrix: bool = False
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
		with torch.no_grad():
			embeddings = self.recognition_model(stack_faces)
		# assert embeddings.shape[0] == len(filtered_faces)
		assert embeddings.shape[1] == 512

		if return_embedding_as_matrix:
			return {'user_name': user_name,
					'embeddings': embeddings
					}
		else:
			user_init_data = [{
				'user_name': user_name,
				'password': '123',
				'embedding': embeddings[id],
			} 
			for id in range(embeddings.shape[0])
			]
			return user_init_data

	def _get_total_init_user_data(self, return_embedding_as_matrix:bool):
		
		if self.users_from_json:
			with open('face_dataset\\dataset.json','r') as f:
				user_folders_list = json.load(f)

				self.user_folders = [self.data_folder_path +'/'+ list(ele.values())[0] 
										for ele in user_folders_list
										# if '_' in list(ele.values())[0]
										]
		else:
			self.user_folders = glob.glob(f"{self.data_folder_path}/*")

		master_init_data = []

		for user_folder in tqdm(self.user_folders, total = len(self.user_folders)):
			user_name = os.path.split(user_folder)[-1].split('.')[0]
			user_init_data = self._run_single_user(user_name = user_name, 
								return_embedding_as_matrix = return_embedding_as_matrix)

			if return_embedding_as_matrix:
				master_init_data.append(user_init_data)
			else:
				master_init_data.extend(user_init_data)
		
		return master_init_data

	@staticmethod
	def get_cosim(input1:torch.Tensor, input2:torch.Tensor)->float:
		input2 = torch.cat([input2, input2, input2, input2, input2],dim = 0)
		norm_1 = torch.unsqueeze(torch.norm(input1, dim =1), dim = 1) 
		norm_2 = torch.unsqueeze(torch.norm(input2, dim =1), dim = 0)
		length_mul_matrix = 1/torch.mul(norm_1, norm_2)

		dot_product = torch.matmul(input1, torch.transpose(input2,0,1))

		dot_product_score = torch.mul(dot_product, length_mul_matrix)

		return torch.max(dot_product_score).item()


	def pipelines(self, 
				run_init_push: bool, 
				evaluation: bool,
				return_embedding_as_matrix: bool
				):
		master_config = get_program_config()
		master_init_data = self._get_total_init_user_data(return_embedding_as_matrix = return_embedding_as_matrix)
		print('number init data: ',len(master_init_data))

		if run_init_push and not return_embedding_as_matrix:
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
													return_embedding_only = False)

				num_embeddings = embeddings.shape[0]
				step = int(num_embeddings)//3
				predict_name_list = []
				for embedd_idx in range(0, num_embeddings, step):
					query_embeddings = embeddings[embedd_idx: embedd_idx+step,:]
					pred_name = db_engine.searchUserWithEmbeddings(batch_query_embeddings = query_embeddings)
					predict_name_list.append(pred_name)

				result[user_name] = predict_name_list
			print(result)

			# result = {}
			# for main_user_dir in glob.glob(f"{self.data_folder_path}/*_*"):
			# 	user_name = os.path.split(main_user_dir)[-1].split('.')[0]
			# 	user_embeddings_dict = self._run_single_user(user_name = user_name, 
			# 												return_embedding_as_matrix = True)
				
			# 	score_dict = {
			# 		user_dict['user_name']: Test_Embeddings.get_cosim(user_dict['embeddings'], 
			# 														user_embeddings_dict['embeddings'])
			# 		for user_dict in master_init_data
			# 	}
			# 	print(user_name, score_dict, '\n')
			# 	sorted_score_dict = {k:v for k,v in \
			# 						sorted(score_dict.items(), key=lambda item: item[1])
			# 						}
			# 	result[user_name] = list(sorted_score_dict.keys())[-1]

			# print(result)