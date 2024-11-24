from src.server.models.Facenet_pytorch.fine_tuning import FineTuner
from src.experiments.check_embedd_facenet_pytorch import Test_Embeddings
import torch


if __name__ == '__main__':
# 	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



	Test_Embeddings(data_folder_path = 'face_dataset/faces_only', 
					pretrained_weight_dir = 'src\\server\\models\\pretrained_weights\\Facenet_pytorch',
					model_string = 'fine_tuning'
				).pipelines(run_init_push = True, 
				evaluation = True, 
				return_embedding_as_matrix = True)


