from train import setup_training_loop_kwargs
from dnnlib import EasyDict

args = EasyDict()

gpus = 1
snap = 50
metrics = ['fid50k_full']
seed = 0


args.gpus = 1
args.image_snapshot_ticks = snap
args.network_snapshot_ticks = snap
args.metrics = metrics
args.dataset = EasyDict()
args.data_loader_kwargs = EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)



