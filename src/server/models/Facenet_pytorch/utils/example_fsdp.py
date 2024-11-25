
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from src.server.models.Facenet_pytorch.inception_resnet_v1 import InceptionResnetV1
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.multiprocessing as mp
import os

# class MyModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin = torch.nn.Conv2d(3, 32, kernel_size = (3,3), stride = 1)

#     def forward(self, x):
#         return torch.mean(torch.nn.functional.relu(self.lin(x)))

def setup(rank:int, world_size:int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_mapping(rank, world_size, batch):
        setup(rank, world_size)
        mod = InceptionResnetV1(pretrained = 'casia-webface', 
								classify=False,
								num_classes=None, 
								dropout_prob=0.6,
								device = rank,
								pretrained_weight_dir = '/kaggle/input/massive-faces'
        					).to(rank)
        opt_mod = torch.compile(mod)
        opt_mod = FSDP(opt_mod, use_orig_params = True)
        t = torch.randn(2,3,160,160).to(rank)
        print(opt_mod(t))
        cleanup()
