from src.server.models.Facenet_pytorch.mtcnn import MTCNN
from src.server.models.Facenet_pytorch.inception_resnet_v1 import InceptionResnetV1
from src.mongodb import Mongo_Handler
from src.utils import get_program_config
import cv2 as cv
from typing import List
import numpy as np
import glob
import torch
import uuid
from tqdm import tqdm


class Test_Embeddings(object):
	def __init__(self,
				data_folder_path: str, 
				pretrained_weight_dir: str,
				p_state_dict_path: str,
                r_state_dict_path: str,
                o_state_dict_path: str,
				):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
		self.detector = MTCNN(image_size=160, 
                    margin=0, 
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7], 
                    factor=0.709, 
                    post_process=True,
                    select_largest=True, 
                    selection_method=None, 
                    keep_all=True,
                    device=device,
                    p_state_dict_path = p_state_dict_path,
                    r_state_dict_path = r_state_dict_path,
                    o_state_dict_path = o_state_dict_path,
                    )
		self.recognition_model = InceptionResnetV1(pretrained = 'casia-webface', 
			    									classify=False, 
			    									num_classes=None, 
			    									dropout_prob=0.6,
			    									device=device,
			    									pretrained_weight_dir = pretrained_weight_dir
	    									)
		self.data_folder_path = data_folder_path

	def _run_single_user(self, user_name:str):
		user_imgs = []
		for path in glob.glob(f"{self.data_folder_path}\\{user_name}\\*"):
			image = cv.imread(path)
			assert image is not None, f"{path}"
			image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
			user_imgs.append(image)

		# infernce in batch
		faces, probs = self.detector(img = user_imgs,
									save_path = None,
									return_prob= True)

		assert len(faces) == len(user_imgs)
		assert len(faces) == len(probs)
		
		filtered_faces = []
		for each_face, each_prob in zip(faces, probs):
			if each_prob[0] is None:
				continue
			else:
				if each_prob[0] > 0.8:
					filtered_faces.append(each_face[0])
		assert len(filtered_faces) != 0
		
		stack_faces = torch.stack(filtered_faces)
		embeddings = self.recognition_model(stack_faces)
		assert embeddings.shape[0] == len(filtered_faces)
		assert embeddings.shape[1] == 512

		user_init_data = [{
			'user_name': user_name,
			'password': '123',
			'embedding': embeddings[id].tolist(),
		} 
		for id in range(embeddings.shape[0])
		]
		return user_init_data

	def _get_total_init_user_data(self):
		user_folders = glob.glob(f"{self.data_folder_path}\\*")
		master_init_data = []

		for user_folder in tqdm(user_folders, total = len(user_folders)):
			user_name = user_folder.split('\\')[-1]

			try:
				user_init_data = self._run_single_user(user_name = user_name)
				master_init_data.extend(user_init_data)
			except:
				print(user_name)
				continue

		return master_init_data

	def pipeline(self):
		master_init_data = self._get_total_init_user_data()
		master_config = get_program_config()
		db_engine = Mongo_Handler(master_config= master_config,
								ini_push= True,
								init_data= master_init_data)
		