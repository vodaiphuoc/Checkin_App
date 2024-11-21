import torch
import glob
from typing import List, Dict, Tuple
import itertools
import random
from tqdm import tqdm
import cv2 as cv
import json
from src.server.models.Facenet_pytorch.mtcnn import fixed_image_standardization
from src.server.models.Facenet_pytorch.inception_resnet_v1 import InceptionResnetV1


class TripLetDataset(torch.utils.data.Dataset):
	"""For train dataset, all data, 
	For validation set, collected users are combined with 100 celeb users
	"""
	def __init__(self, 
				is_train: bool,
				data_folder_path: str,
				ratio_other_user: float
				)->None:
		return None

	@staticmethod
	def _offload_user_folders(usr_folder_list: List[str])->None:
		with open('usr_folder_list.json','w') as f:
			json.dump([{k:v} for k, v in enumerate(usr_folder_list)],f, indent = 5)

	@staticmethod
	def _make_index_list(is_train: bool,
						offload:bool,
						shard_length:int,
						data_folder_path: str,
						ratio_other_user: float,
						number_celeb_in_train: int = 500,
						number_celeb_in_val: int = 150
						)->None:
		# user folders
		if is_train:
			glob_iter = glob.glob(f"{data_folder_path}/*_*") + \
						random.sample(glob.glob(f"{data_folder_path}/[0-9]*"), k = number_celeb_in_train)
		else:
			glob_iter = glob.glob(f"{data_folder_path}/*_*") + \
						glob.glob(f"{data_folder_path}/[0-9]*")[:number_celeb_in_val]

		if offload:
			TripLetDataset._offload_user_folders(glob_iter)

		number_other_user = int((len(glob_iter)-1)*ratio_other_user)
		# user maps to other users
		userIdx2other_usersIdx = {
			user_dir_idx:random.sample([ _each_dix for _each_dix in range(len(glob_iter)) 
											if glob_iter[user_dir_idx] != glob_iter[_each_dix]
										],
										k = number_other_user)
			for user_dir_idx in range(len(glob_iter))
		}
		# user maps to its image paths
		user2img_path = {
			user_dir_idx: [
				_path for _path in glob.glob('*.*',root_dir = f"{glob_iter[user_dir_idx]}")
			]
			for user_dir_idx in range(len(glob_iter))
		}

		print('prepare index ...')
		count_shard = 0
		master_index = []
		for user_dir_idx in tqdm(range(len(glob_iter)), total = len(glob_iter)):
			# get total images of current user
			anchor_imgs_path = user2img_path[user_dir_idx]
			positives = [{user_dir_idx:file_name_pair} 
							for file_name_pair in itertools.permutations(anchor_imgs_path,2)
						]
			
			neg_img_list = []
			for other_user_idx in userIdx2other_usersIdx[user_dir_idx]:
				neg_img_list.extend([
										{other_user_idx: img_file_name} 
										for img_file_name in user2img_path[other_user_idx]
									])

			# merge dict from itertool.product
			product_list = [ k_ap|k_n for (k_ap, k_n) in
							itertools.product(positives,neg_img_list)
							]
			
			master_index.extend(product_list)

			if offload and len(master_index) == shard_length:
				save_index_file = f'master_index_train_{count_shard}.json' if is_train \
							else f'master_index_val_{count_shard}.json'
				with open(save_index_file,'w') as f:
					json.dump(master_index,f)
				master_index = []

		print(f'Done index list: {len(master_index)}, train_set?: {is_train}')
		if not offload:
			return master_index

	def __len__(self):
		return len(self.master_index)

	def __getitem__(self, index:int)->Tuple[torch.Tensor]:
		a_path, p_path, n_path = self.master_index[index]

		return_triplet = []

		for _path in [a_path, p_path, n_path]:
			image = cv.imread(_path)
			image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
			image_tensor = torch.tensor(image).permute(2,0,1).to(float)
			image_tensor = fixed_image_standardization(image_tensor)
			return_triplet.append(image_tensor)

		return tuple(return_triplet)

class FineTuner(object):
	freeze_list = ['mixed_7a','repeat_3', 'block8', 'avgpool_1a', 'last_linear', 'last_bn']

	def __init__(self, 
				num_epochs:int,
				gradient_accumulate_steps: int,
				lr: float,
				data_folder_path:str,
				ratio_other_user:float,
				pretrained_weight_dir: str,
				batch_size: int,
				num_workers:int,
				device: torch.device
				):
		
		self.train_loader = FineTuner._make_loaders(is_train = True,
													data_folder_path = data_folder_path, 
													ratio_other_user = ratio_other_user,
													batch_size = batch_size,
													num_workers = num_workers)
		self.val_loader = FineTuner._make_loaders(is_train = False,
													data_folder_path = data_folder_path, 
													ratio_other_user = ratio_other_user,
													batch_size = batch_size,
													num_workers = num_workers)

		self.model = InceptionResnetV1(pretrained = 'casia-webface', 
									classify=False,
									num_classes=None, 
									dropout_prob=0.6,
									device = device,
									pretrained_weight_dir = pretrained_weight_dir)

		self.optimizer = torch.optim.Adam(self.model.parameters(),lr = lr)

		for name, module in self.model.named_modules():
			if name not in self.freeze_list:
				for param in module.parameters():
					param.requires_grad = False
			else:
				for param in module.parameters():
					param.requires_grad = True
		
		self.num_epochs = num_epochs
		self.gradient_accumulate_steps = gradient_accumulate_steps
		self.device = device
		self.batch_size = batch_size
		self.loss_fn = torch.nn.TripletMarginLoss(margin=1.0, 
												p=2.0, 
												eps=1e-06, 
												swap=False, 
												size_average=None, 
												reduce=None, 
												reduction='mean')
	@staticmethod
	def _make_loaders(is_train: bool,
					data_folder_path: str, 
					ratio_other_user: str,
					batch_size:int,
					num_workers:int
					):
		dataset = TripLetDataset(is_train = True,
								data_folder_path = data_folder_path, 
								ratio_other_user = ratio_other_user
		)
		
		return torch.utils.data.DataLoader(dataset, 
										batch_size= batch_size, 
										shuffle=True if is_train else False, 
										num_workers=num_workers,
										pin_memory=True, 
										drop_last=True if is_train else False,
										prefetch_factor=2,
										persistent_workers=True
		)

	def train(self, save_path:str):
		train_logs = {}
		val_logs = {}
		for epoch in range(self.num_epochs):

			mean_train_loss = 0
			for batch_idx, (a_batch, p_batch, n_batch) in enumerate(self.train_loader):
				a_batch = a_batch.to(self.device)  
				p_batch = p_batch.to(self.device)
				n_batch = n_batch.to(self.device)

				embeddings = self.model(torch.cat([a_batch, p_batch, n_path], dim = 0))

				a_embeddings = embeddings[0: self.batch_size,:]
				p_embeddings = embeddings[self.batch_size: 2*self.batch_size,:]
				n_embeddings = embeddings[2*self.batch_size:,:]

				loss = self.loss_fn(a_embeddings, p_embeddings, n_embeddings)

				loss = loss/self.gradient_accumulate_steps

				mean_train_loss += loss
				if ((batch_idx + 1) % self.gradient_accumulate_steps == 0) or \
					(batch_idx + 1 == len(self.train_loader)):

					self.optimizer.step()
					self.optimizer.zero_grad()

			mean_train_loss = mean_train_loss/len(self.train_loader)
			train_logs[epoch] = mean_train_loss.clone().detach().cpu().numpy()[0]

			if self.num_epochs//epoch == 2 or epoch == self.num_epochs -1:
				mean_val_loss = 0
				with torch.no_grad():
					for batch_idx, (a_batch, p_batch, n_batch) in enumerate(self.val_loader):
						a_batch = a_batch.to(self.device)  
						p_batch = p_batch.to(self.device)
						n_batch = n_batch.to(self.device)

						embeddings = self.model(torch.cat([a_batch, p_batch, n_path], dim = 0))

						a_embeddings = embeddings[0: self.batch_size,:]
						p_embeddings = embeddings[self.batch_size: 2*self.batch_size,:]
						n_embeddings = embeddings[2*self.batch_size:,:]

						mean_val_loss += self.loss_fn(a_embeddings, p_embeddings, n_embeddings)

				mean_val_loss = mean_val_loss/len(self.val_loader)
				val_logs[epoch] = mean_val_loss.clone().detach().cpu().numpy()[0]

		print(train_logs)
		print(val_logs)


		# save checkpoints
		torch.save(self.model.state_dict(), save_path)
