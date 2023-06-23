import wandb
import argparse
import src.image_systems as image_systems
from src.utils.utils import process_config, seed_everything

SYSTEM = {
    'CentralSystem': image_systems.CentralSystem,
    'CentralDPSystem': image_systems.CentralDPSystem,
    'DiNNOSystem': image_systems.DiNNOSystem,
    'DPDiNNOSystem': image_systems.DPDiNNOSystem,
    'DSGTSystem': image_systems.DSGTSystem,
    'DPDSGTSystem': image_systems.DPDSGTSystem,
    'DistributedConsensusSystem': image_systems.DistributedConsensusSystem,
    'DistributedDPConsensusSystem': image_systems.DistributedDPConsensusSystem,
    'DistributedSystemNoComms': image_systems.DistributedSystemNoComms,
}


def run(args, gpu_device):
    config = process_config(args.config)
    if not gpu_device:
        gpu_device = 'cpu'
    config.gpu_device = gpu_device
    seed_everything(config.seed)
    wandb.init(project="project_name", entity='entity_name', name=config.exp_name, config=config, sync_tensorboard=True, settings=wandb.Settings(start_method='thread'))
    SystemClass = SYSTEM[config.system]
    system = SystemClass(config)
    system.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    parser.add_argument('--gpu-device', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    args = parser.parse_args()
    gpu_device = str(args.gpu_device) if args.gpu_device else None

    run(args, gpu_device)
