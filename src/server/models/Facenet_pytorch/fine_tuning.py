import torch
import glob
from typing import List, Dict, Tuple, Any
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
				is_train = True,
				return_examples = 512,
				data_folder_path = 'face_dataset\\faces_only', 
				ratio_other_user = 0.2,
				number_celeb_in_train = 500,
				number_celeb_in_val = 150
				)->None:

		(
			self.glob_iter,
			self.userIdx2other_usersIdx, 
			self.user2img_path
		) = TripLetDataset._make_index_list(is_train = is_train,
											data_folder_path = data_folder_path, 
											ratio_other_user = ratio_other_user,
											number_celeb_in_train = number_celeb_in_train,
											number_celeb_in_val = number_celeb_in_val
											)
		self.data_folder_path = data_folder_path
		self.return_examples = return_examples

		self.index_iter = [ user_dir_idx
							for user_dir_idx in range(len(self.glob_iter))
							]
		return None

	@staticmethod
	def _make_index_list(is_train: bool,
						data_folder_path: str,
						ratio_other_user: float,
						number_celeb_in_train: int = 500,
						number_celeb_in_val: int = 150
						)->None:
		# user folders
		if is_train:
			glob_iter = glob.glob("*_*",root_dir = f"{data_folder_path}") + \
						random.sample(glob.glob("[0-9]*", root_dir = f"{data_folder_path}"), 
									k = number_celeb_in_train)
		else:
			glob_iter = glob.glob("*_*",root_dir = f"{data_folder_path}") + \
						glob.glob("[0-9]*", root_dir = f"{data_folder_path}")[:number_celeb_in_val]

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
				_path for _path in glob.glob('*.*',
									root_dir = f"{data_folder_path+'/'+glob_iter[user_dir_idx]}")
			]
			for user_dir_idx in range(len(glob_iter))
		}

		return glob_iter, userIdx2other_usersIdx, user2img_path
		

	def __len__(self):
		return len(self.index_iter)

	def _get_triplet_index(self, user_dir_idx: int)-> List[Dict[str,Any]]:
		"""
		Given a list of index (user folder dir)
		return mapping from user anchor/positive files with 
		negative files
		"""
		# get total images of current user
		anchor_imgs_path = self.user2img_path[user_dir_idx]
		positives = [
						{user_dir_idx:file_name_pair}
						for file_name_pair \
						in itertools.combinations(anchor_imgs_path,2)
					]
		
		neg_img_list = []
		for other_user_idx in self.userIdx2other_usersIdx[user_dir_idx]:
			if len(neg_img_list) > int(self.return_examples//len(positives)):
				break
			else:
				neg_img_list.extend([
										{other_user_idx: img_file_name} 
										for img_file_name in 
										random.sample(self.user2img_path[other_user_idx], 
														k = len(self.user2img_path[other_user_idx])//7)
									])
		
		# merge dict from itertool.product
		product_list = [ k_ap|k_n for (k_ap, k_n) in
						itertools.product(positives,neg_img_list)
						]
		assert len(product_list) != 0
		
		# print('Length product_list: ',len(product_list))
		return product_list[: self.return_examples]

	def _paths2tensor(self, path_list: List[str])->torch.Tensor:
		return_tensor = []
		for _path in path_list:
			image = cv.imread(_path)
			image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
			image_tensor = torch.tensor(image).permute(2,0,1)
			return_tensor.append(image_tensor)

		return fixed_image_standardization(torch.stack(return_tensor)).to(torch.float32)

	def __getitem__(self, index:int)->Tuple[torch.Tensor]:
		"""
		Return:
			- Tuple of tensor shape (self.return_examples,3,160,160)
		"""
		idx = self.index_iter[index]
		triplet_idx = self._get_triplet_index(idx)

		assert len(triplet_idx) == self.return_examples, f"Found {len(triplet_idx)}"

		a_path, p_path, n_path = [], [], []
		for map_dict in triplet_idx:
			full_paths = []
			for k, v in map_dict.items():
				if isinstance(v,tuple):
					a_path.append(self.data_folder_path+'/'+ self.glob_iter[k]+'/'+v[0])
					p_path.append(self.data_folder_path+'/'+ self.glob_iter[k]+'/'+v[1])
				else:
					n_path.append(self.data_folder_path+'/'+ self.glob_iter[k]+'/'+v)

		# assert len(a_path) == len(p_path) and \
		# 		len(a_path) == len(n_path) and \
		# 		len(a_path) == self.return_examples, \
		# 	f"Found {len(a_path)}, {len(p_path)}, {len(n_path)}"

		anchors = self._paths2tensor(a_path)
		positives = self._paths2tensor(p_path)
		negatives = self._paths2tensor(n_path)

		# assert anchors.shape[0] == self.return_examples, f"Found {anchors.shape[0]}"
		# assert positives.shape[0] == self.return_examples, f"Found {positives.shape[0]}"
		# assert negatives.shape[0] == self.return_examples, f"Found {negatives.shape[0]}"
		return anchors, positives, negatives

class FineTuner(object):
	freeze_list = ['mixed_7a','repeat_3', 'block8', 'avgpool_1a', 'last_linear', 'last_bn']

	def __init__(self, 
				num_epochs:int,
				gradient_accumulate_steps: int,
				lr: float,
				device: torch.device,
				pretrained_weight_dir: str = 'src\\server\\models\\pretrained_weights\\Facenet_pytorch',
				return_examples:int = 512,
				data_folder_path:str = 'face_dataset/faces_only',
				ratio_other_user:float = 0.2,
				number_celeb_in_train:int = 500,
				number_celeb_in_val:int = 150,
				batch_size:int = 64,
				num_workers:int = 2
				):
		
		self.train_loader = FineTuner._make_loaders(True,
													return_examples,
													data_folder_path, 
													ratio_other_user,
													number_celeb_in_train,
													number_celeb_in_val,
													batch_size, 
													num_workers)
		self.val_loader = FineTuner._make_loaders(False,
													return_examples,
													data_folder_path, 
													ratio_other_user,
													number_celeb_in_train,
													number_celeb_in_val,
													batch_size,
													num_workers
													)

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
		self.return_examples = return_examples

		self.master_batch_size = self.batch_size*self.return_examples
		self.loss_fn = torch.nn.TripletMarginLoss(margin=1.0, 
												p=2.0, 
												eps=1e-06, 
												swap=False, 
												size_average=None, 
												reduce=None, 
												reduction='mean')
	@staticmethod
	def _make_loaders(is_train:bool,
					return_examples:int,
					data_folder_path:str, 
					ratio_other_user:float,
					number_celeb_in_train:int,
					number_celeb_in_val:int,
					batch_size:int,
					num_workers:int
					):
		dataset = TripLetDataset(return_examples = return_examples,
								is_train = is_train,
								data_folder_path = data_folder_path, 
								ratio_other_user = ratio_other_user,
								number_celeb_in_train = number_celeb_in_train,
								number_celeb_in_val = number_celeb_in_val
		)
		
		return torch.utils.data.DataLoader(dataset, 
										batch_size= batch_size, 
										shuffle=False if is_train else False, 
										num_workers=num_workers,
										pin_memory=True, 
										drop_last=True if is_train else False,
										prefetch_factor=2,
										persistent_workers=True
		)

	def _pre_process_batch_data(self, batch: torch.Tensor)->torch.Tensor:
		return batch.reshape((self.master_batch_size, 3, 160, 160)).to(self.device)

	def train(self, save_path:str):
		"""Main training function"""
		train_logs = {}
		val_logs = {}
		for epoch in range(self.num_epochs):
			mean_train_loss = 0
			for batch_idx, (a_batch, p_batch, n_batch) in tqdm(enumerate(self.train_loader),
																total = self.num_epochs):
				
				# assert a_batch.shape[0] == self.batch_size, f"Found {a_batch.shape[0]}"
				# assert a_batch.shape[1] == self.return_examples, f"Found {a_batch.shape}, {batch_idx}"
				# assert a_batch.shape[2] == 3, f"Found {a_batch.shape[2]}"

				a_batch = self._pre_process_batch_data(a_batch)
				p_batch = self._pre_process_batch_data(p_batch)
				n_batch = self._pre_process_batch_data(n_batch)

				embeddings = self.model(torch.cat([a_batch, p_batch, n_batch], dim = 0))

				a_embeddings = embeddings[0: self.master_batch_size,:]
				p_embeddings = embeddings[self.master_batch_size: 2*self.master_batch_size,:]
				n_embeddings = embeddings[2*self.master_batch_size:,:]

				loss = self.loss_fn(a_embeddings, p_embeddings, n_embeddings)

				loss = loss/self.gradient_accumulate_steps

				mean_train_loss += loss
				if ((batch_idx + 1) % self.gradient_accumulate_steps == 0) or \
					(batch_idx + 1 == len(self.train_loader)):

					self.optimizer.step()
					self.optimizer.zero_grad()

			mean_train_loss = mean_train_loss/len(self.train_loader)
			train_logs[epoch] = mean_train_loss.clone().detach().cpu().numpy()[0]

			if self.num_epochs//epoch == 1 or epoch == self.num_epochs -1:
				mean_val_loss = 0
				with torch.no_grad():
					for batch_idx, (val_a_batch, val_p_batch, val_n_batch) in enumerate(self.val_loader):
						val_a_batch = self._pre_process_batch_data(val_a_batch)
						val_p_batch = self._pre_process_batch_data(val_p_batch)
						val_n_batch = self._pre_process_batch_data(val_n_batch)

						embeddings = self.model(torch.cat([val_a_batch, val_p_batch, val_n_path], 
												dim = 0))

						a_embeddings = embeddings[0: self.batch_size*self.return_examples,:]
						p_embeddings = embeddings[self.batch_size: 2*self.batch_size,:]
						n_embeddings = embeddings[2*self.batch_size:,:]

						mean_val_loss += self.loss_fn(a_embeddings, p_embeddings, n_embeddings)

				mean_val_loss = mean_val_loss/len(self.val_loader)
				val_logs[epoch] = mean_val_loss.clone().detach().cpu().numpy()[0]

		print(train_logs)
		print(val_logs)


		# save checkpoints
		torch.save(self.model.state_dict(), save_path)