Fix the checkpointing and training resume logic in /app/checkpoint.py and /app/train.py so that a PyTorch training process interrupted and resumed mid-training produces bit-exact identical loss values and final model parameters compared to an uninterrupted reference run.

The codebase in /app contains:
- /app/model.py: Model architecture.
- /app/dataset.py: Synthetic dataset and data sampler.
- /app/checkpoint.py: Checkpoint saving and loading utilities.
- /app/train.py: Training loop with gradient accumulation, AMP autocast, LR scheduling, and checkpoint resumption.

The current implementation in /app/checkpoint.py and /app/train.py suffers from multiple state-completeness and state-ordering defects during checkpoint save/resume. Specifically, when interrupted mid-epoch or mid-accumulation, resuming from a checkpoint fails to reproduce the reference training trajectory.

Requirements:
1. Ensure all relevant training loop state components are preserved in /app/checkpoint.py when saving checkpoints:
   - Model state dict and Optimizer state dict.
   - Learning Rate Scheduler state dict.
   - AMP GradScaler state dict.
   - Sampler state (epoch, consumed samples offset).
   - Micro-step counter for gradient accumulation.
   - Accumulated parameter gradients (param.grad) if interrupted mid-accumulation.
   - Global RNG states (torch, numpy, python random).
2. Ensure /app/train.py correctly restores all saved state components upon resuming, maintaining consistent iteration counts, gradient accumulation boundaries, learning rate schedules, loss scales, and data sampling order.
3. Do not modify /app/model.py.
4. Ensure single-threaded CPU execution and determinism via torch.use_deterministic_algorithms(True).

You have 1800 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
