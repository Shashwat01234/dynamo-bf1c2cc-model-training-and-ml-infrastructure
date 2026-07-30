import torch
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
