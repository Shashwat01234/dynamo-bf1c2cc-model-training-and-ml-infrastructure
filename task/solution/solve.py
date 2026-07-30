import os
import sys

FIXED_CHECKPOINT_PY = '''import torch
import random
import numpy as np

def save_checkpoint(filepath, model, optimizer, scheduler, scaler, sampler, epoch, global_step, micro_step):
    """
    BIT-EXACT COMPLETE MARKOV CHECKPOINT:
    Saves model, optimizer, scheduler, scaler, sampler, epoch, global_step, micro_step,
    accumulated gradients, and all RNG streams.
    """
    grad_state_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_state_dict[name] = param.grad.detach().clone()

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "sampler_state_dict": sampler.state_dict() if sampler is not None else None,
        "grad_state_dict": grad_state_dict,
        "epoch": epoch,
        "global_step": global_step,
        "micro_step": micro_step,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
    }
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer, scheduler=None, scaler=None, sampler=None):
    """
    BIT-EXACT COMPLETE MARKOV RESUMPTION:
    Restores model, optimizer, scheduler, scaler, sampler, epoch, global_step, micro_step,
    accumulated gradients, and all RNG streams.
    """
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
    if sampler is not None and checkpoint.get("sampler_state_dict") is not None:
        sampler.load_state_dict(checkpoint["sampler_state_dict"])

    if "grad_state_dict" in checkpoint:
        for name, param in model.named_parameters():
            if name in checkpoint["grad_state_dict"]:
                param.grad = checkpoint["grad_state_dict"][name].clone()
        
    if "rng_states" in checkpoint:
        torch.set_rng_state(checkpoint["rng_states"]["torch"])
        np.random.set_state(checkpoint["rng_states"]["numpy"])
        random.setstate(checkpoint["rng_states"]["python"])
        
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    micro_step = checkpoint.get("micro_step", 0)
    
    return epoch, global_step, micro_step
'''

FIXED_TRAIN_PY = '''import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SmallTransformer
from dataset import SyntheticTextDataset, ResumableSampler
from checkpoint import save_checkpoint, load_checkpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def run_training(
    seed=42,
    total_micro_steps=60,
    grad_accum_steps=4,
    batch_size=8,
    interrupt_at=None,
    checkpoint_path="checkpoint.pt",
    resume=False,
):
    set_seed(seed)
    device = torch.device("cpu")

    dataset = SyntheticTextDataset(num_samples=256, seq_len=16, vocab_size=64, seed=seed)
    sampler = ResumableSampler(dataset, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    model = SmallTransformer(vocab_size=64, d_model=32, nhead=2, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cpu', enabled=True)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    global_step = 0
    micro_step = 0
    current_micro_step = 0

    if resume and os.path.exists(checkpoint_path):
        start_epoch, global_step, micro_step = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, sampler
        )
        current_micro_step = global_step * grad_accum_steps + micro_step
    else:
        optimizer.zero_grad()

    losses = []
    done = False
    epoch = start_epoch

    while not done and current_micro_step < total_micro_steps:
        for inputs, targets in loader:
            if current_micro_step >= total_micro_steps:
                done = True
                break

            inputs, targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast('cpu', enabled=True):
                logits = model(inputs)
                loss = criterion(logits.view(-1, 64), targets.view(-1))
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()
            micro_step += 1
            current_micro_step += 1
            losses.append((current_micro_step, loss.item()))

            if micro_step % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                micro_step = 0

            if interrupt_at is not None and current_micro_step == interrupt_at:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    sampler,
                    epoch,
                    global_step,
                    micro_step,
                )
                done = True
                break

        epoch += 1

    return losses, model.state_dict()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=60)
    parser.add_argument("--interrupt-at", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_training(
        seed=args.seed,
        total_micro_steps=args.total_steps,
        interrupt_at=args.interrupt_at,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
    )
'''

def main():
    app_dir = "/app"
    if not os.path.exists(app_dir):
        # Fallback to local task directory if running outside container
        app_dir = os.path.join(os.path.dirname(__file__), "..", "environment", "data")
        
    ckpt_path = os.path.join(app_dir, "checkpoint.py")
    train_path = os.path.join(app_dir, "train.py")
    
    with open(ckpt_path, "w") as f:
        f.write(FIXED_CHECKPOINT_PY)
        
    with open(train_path, "w") as f:
        f.write(FIXED_TRAIN_PY)
        
    print(f"Successfully updated {ckpt_path} and {train_path}")

if __name__ == "__main__":
    main()
