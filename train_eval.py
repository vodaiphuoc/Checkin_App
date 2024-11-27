# from src.server.models.Facenet_pytorch.fine_tuning import FineTuner
from src.experiments.check_embedd_facenet_pytorch import Test_Embeddings
import torch
from collections import OrderedDict
import json
import glob

if __name__ == '__main__':
	Test_Embeddings(data_folder_path = 'face_dataset/faces_only', 
					users_from_json = True,
					json_path = 'face_dataset/dataset.json',
					pretrained_weight_dir = 'src\\server\\models\\pretrained_weights\\Facenet_pytorch',
					model_string = 'fine_tuning'
				).pipelines(run_init_push = True, 
				evaluation = True, 
				return_embedding_as_matrix = False)

	# path = 'src\\server\\models\\pretrained_weights\\Facenet_pytorch\\20180408-102900-casia-webface.pt'
	# fine_tuning_path = 'src\\server\\models\\pretrained_weights\\Facenet_pytorch\\fine_tuning.pt'
	# state_dict = torch.load(fine_tuning_path, 
	# 		weights_only = True, map_location= torch.device('cpu'))

	# new_state_dict = OrderedDict()
	# for k, v in state_dict.items():
	# 	new_k = k.replace('_orig_mod.module.','')
	# 	new_state_dict[new_k] = v
