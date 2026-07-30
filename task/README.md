# Dynamo Task: Bit-Exact Resumability of a Training Loop Under Randomized Interruption

## Overview
This task tests an ML Infrastructure engineer's ability to audit, diagnose, and fix state-completeness and state-ordering defects in PyTorch training loop checkpointing and resumption paths.

When training models on preemptible compute clusters, jobs may be interrupted at any micro-step mid-epoch or mid-accumulation window. To guarantee bit-exact reproducibility, the checkpoint must capture the complete Markov state of the training process.

## Key Defects Tested
1. **AMP GradScaler State**: Omission of `scaler.state_dict()` drops dynamic loss scale and growth tracker state.
2. **LR Scheduler State & Timing**: Omission of `scheduler.state_dict()` or incorrect scheduler step placement leads to learning rate schedule misalignment.
3. **Sampler Offset**: Saving only `epoch` count in DataLoader sampler resets the epoch sample offset to 0 upon resume, re-yielding already consumed batches.
4. **Gradient Accumulation Micro-step Bookkeeping**: Omission of micro-step counter shifts gradient accumulation boundaries across resume boundaries.
5. **Accumulated Gradients**: Interruption mid-accumulation drops accumulated parameter `.grad` tensors if not explicitly saved.
6. **RNG Stream Restoration**: Incomplete RNG restoration across PyTorch, NumPy, and Python `random` causes non-deterministic data augmentation or sampling post-resume.

## Testing & Verification
The verifier runs an uninterrupted reference training trajectory and compares it against 6 interruption/resume scenarios (2 seeds x 3 interrupt steps). All loss values and parameter tensors must match bit-for-bit (`torch.equal`). An anti-cheat harness verifies that the unpatched codebase fails the test suite.
