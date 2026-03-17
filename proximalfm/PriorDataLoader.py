import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from .Prior import ProximalPrior, ProximalPriorConfig

class ProximalDataset(Dataset):
    """
    A PyTorch Dataset that generates Proximal Causal datasets on-the-fly.
    """
    def __init__(
        self, 
        prior_config: ProximalPriorConfig, 
        num_episodes: int, 
        samples_per_episode: int, 
        seed_offset: int = 0
    ):
        self.prior_config = prior_config
        self.prior_config.device = 'cpu' 
        self.prior = ProximalPrior(config=self.prior_config)
        
        self.num_episodes = num_episodes
        self.samples_per_episode = samples_per_episode
        self.seed_offset = seed_offset

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        episode_seed = self.seed_offset + idx
        
        episode = self.prior.generate_episode(
            batch_size=self.samples_per_episode, 
            episode_seed=episode_seed
        )
        
        O = torch.cat([episode.X, episode.Z, episode.W], dim=-1)
        
        A = episode.A.unsqueeze(-1)
        Y = episode.Y.unsqueeze(-1)
        Y0 = episode.Y0.unsqueeze(-1)
        Y1 = episode.Y1.unsqueeze(-1)
        
        ITE = Y1 - Y0
        
        return {
            'X': O,
            'a': A,
            'y': Y,
            'y0': Y0,
            'y1': Y1,
            'ite': ITE
        }


def create_causal_data_loaders(
    prior_config: ProximalPriorConfig,
    batch_size: int = 16,            # Number of datasets to feed the transformer at once
    samples_per_episode: int = 1000, # Number of rows (patients) per dataset
    train_episodes: int = 10000,     # How many datasets constitute one training epoch
    val_episodes: int = 1000,        # How many datasets to evaluate during validation
    num_workers: int = 4,
    seed: int = 42
):
    """
    Creates DataLoaders for on-the-fly Proximal Causal Learning.
    """
    
    # Train Dataset
    train_dataset = ProximalDataset(
        prior_config=prior_config,
        num_episodes=train_episodes,
        samples_per_episode=samples_per_episode,
        seed_offset=seed
    )
    
    # Validation Dataset
    # Shift the seed offset to guarantee the validation datasets are entirely unseen
    val_dataset = ProximalDataset(
        prior_config=prior_config,
        num_episodes=val_episodes,
        samples_per_episode=samples_per_episode,
        seed_offset=seed + train_episodes 
    )
    
    # We use pin_memory=True to speed up the CPU -> GPU transfer in the trainer loop
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, None