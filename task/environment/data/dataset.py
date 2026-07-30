import torch
from torch.utils.data import Dataset, Sampler

class SyntheticTextDataset(Dataset):
    def __init__(self, num_samples=256, seq_len=16, vocab_size=64, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1), generator=g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]

class ResumableSampler(Sampler):
    def __init__(self, data_source, seed=42, epoch=0, consumed_samples=0):
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.seed = seed
        self.epoch = epoch
        self.consumed_samples = consumed_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.num_samples, generator=g).tolist()
        indices = indices[self.consumed_samples:]
        for idx in indices:
            self.consumed_samples += 1
            yield idx
        self.consumed_samples = 0
        self.epoch += 1

    def __len__(self):
        return self.num_samples - self.consumed_samples

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "consumed_samples": self.consumed_samples,
            "seed": self.seed
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.consumed_samples = state_dict.get("consumed_samples", 0)
        self.seed = state_dict.get("seed", self.seed)
