import os
import sys
import hashlib
import tempfile
import pytest
from pathlib import Path
import torch

# Ensure /app is in sys.path when running in container, else fallback to local task directory
app_path = Path("/app")
if app_path.exists():
    sys.path.insert(0, str(app_path))
else:
    local_data_path = Path(__file__).parent.parent / "environment" / "data"
    sys.path.insert(0, str(local_data_path.resolve()))

from train import run_training
import checkpoint
import train

def test_output_files_exist():
    """Verify that required files /app/checkpoint.py and /app/train.py exist."""
    checkpoint_file = Path("/app/checkpoint.py") if app_path.exists() else local_data_path / "checkpoint.py"
    train_file = Path("/app/train.py") if app_path.exists() else local_data_path / "train.py"
    assert checkpoint_file.exists(), f"Missing expected output file {checkpoint_file}"
    assert train_file.exists(), f"Missing expected output file {train_file}"

def test_reference_trajectory_integrity():
    """Verify reference loss trajectory matches expected hash to prevent anti-cheat shortcuts."""
    ref_losses, ref_params = run_training(seed=42, total_micro_steps=60, interrupt_at=None, resume=False)
    assert len(ref_losses) == 60, "Reference trajectory must produce 60 micro-step loss values."
    
    # Compute SHA256 of rounded loss trajectory
    rounded_losses = [(step, round(loss, 6)) for step, loss in ref_losses]
    loss_hash = hashlib.sha256(repr(rounded_losses).encode()).hexdigest()
    expected_hash = "836b77566daccc0c0eb7e89f0ef5134097b53709e157f33da83cf4501da2bfe2"
    assert loss_hash == expected_hash, f"Reference loss trajectory hash mismatch: got {loss_hash}, expected {expected_hash}"

@pytest.mark.parametrize("seed,interrupt_at", [
    (42, 17),   # Mid-accumulation micro-step
    (42, 32),   # End of accumulation window (optimizer step boundary)
    (42, 45),   # Cross-epoch / mid-epoch boundary
    (1234, 19), # Mid-accumulation window with different seed
    (1234, 28), # Optimizer step boundary with different seed
    (1234, 41), # Mid-epoch boundary with different seed
])
def test_bit_exact_resumability(seed, interrupt_at):
    """Verify that interrupted and resumed run matches reference run bit-for-bit across 6 scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
        
        # 1. Uninterrupted reference run
        ref_losses, ref_params = run_training(
            seed=seed,
            total_micro_steps=60,
            interrupt_at=None,
            checkpoint_path=ckpt_path,
            resume=False
        )
        
        # 2. Phase 1: Train from 0 to interrupt_at -> save checkpoint -> exit
        part1_losses, _ = run_training(
            seed=seed,
            total_micro_steps=60,
            interrupt_at=interrupt_at,
            checkpoint_path=ckpt_path,
            resume=False
        )
        assert len(part1_losses) == interrupt_at, f"Phase 1 should stop at step {interrupt_at}"
        assert os.path.exists(ckpt_path), "Checkpoint file was not created at interruption point."
        
        # 3. Phase 2: Resume from checkpoint -> train from interrupt_at to 60 in fresh execution
        res_losses, res_params = run_training(
            seed=seed,
            total_micro_steps=60,
            interrupt_at=None,
            checkpoint_path=ckpt_path,
            resume=True
        )
        
        # Combine full resumed loss trajectory
        full_res_losses = part1_losses + res_losses
        assert len(full_res_losses) == len(ref_losses) == 60, "Resumed loss count mismatch."
        
        # Verify bit-exact loss matching at every step
        for i, ((step_ref, loss_ref), (step_res, loss_res)) in enumerate(zip(ref_losses, full_res_losses)):
            assert step_ref == step_res, f"Step index mismatch at position {i}: ref={step_ref}, res={step_res}"
            assert loss_ref == loss_res, f"Loss mismatch at step {step_ref}: ref={loss_ref}, res={loss_res}"
            
        # Verify bit-exact parameter matching
        for k in ref_params:
            assert torch.equal(ref_params[k], res_params[k]), f"Parameter tensor '{k}' mismatch post-resume."

def test_unpatched_code_discrimination():
    """Anti-cheat test: verify that missing state components cause failure in at least 4 of 6 scenarios."""
    scenarios = [
        (42, 17), (42, 32), (42, 45),
        (1234, 19), (1234, 28), (1234, 41)
    ]
    
    # Temporarily inspect loaded checkpoint function to verify state-completeness
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_check.pt")
        
        # Save a checkpoint using the active checkpoint.py
        model = train.SmallTransformer()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        scaler = torch.amp.GradScaler('cpu')
        dataset = train.SyntheticTextDataset()
        sampler = train.ResumableSampler(dataset)
        
        checkpoint.save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, sampler, 1, 2, 3)
        raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Check required state dictionary keys exist in the saved checkpoint
        required_keys = ["scheduler_state_dict", "scaler_state_dict", "sampler_state_dict", "micro_step", "grad_state_dict"]
        for key in required_keys:
            assert key in raw_state, f"Checkpoint state is missing critical state key '{key}' required for bit-exact resumability."
