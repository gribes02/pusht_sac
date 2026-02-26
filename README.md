# PushT — SAC Training for VLA Dataset Collection

A Soft Actor-Critic (SAC) reinforcement learning agent trained on the [PushT](https://github.com/huggingface/gym-pusht) environment. The goal is to train a high-quality policy whose rollouts can be used as a demonstration dataset for a Vision-Language-Action (VLA) model.

The agent controls a circular end-effector and must push a T-shaped block into a target goal zone. The episode is solved when the block overlaps the goal by at least 95%.

---

## Demo

https://github.com/user-attachments/assets/3c944e3a-549c-4453-bf5e-e0ee4fef478c

https://github.com/user-attachments/assets/24296df9-4467-4985-ab24-eeebf0d4a64c

---

## Results

After ~900k environment steps (~3000 episodes), the policy achieves a **Mean-100 reward of ~176**, with individual episodes regularly scoring 200+. The policy reliably solves the task from random initial positions and block orientations.

---

## Project Structure

```
pusht_sac/
├── train.py               # SAC training loop with checkpointing
├── evaluate.py            # Evaluate and record the trained policy
├── collect_dataset.py     # Collect demonstrations for VLA training
├── requirements.txt       # Python dependencies
├── sac/
│   ├── agent.py           # SACAgent — actor/critic update + save/load
│   ├── actor.py           # Gaussian policy network
│   ├── critic.py          # Q-value network
│   ├── replay_buffer.py   # Experience replay
│   ├── losses.py          # Loss utilities
│   └── utils.py           # State normalisation & action scaling
├── env/
│   └── pusht_wrapper.py   # Environment wrapper
├── configs/
│   └── sac_configs.yaml   # Hyperparameter config
└── assets/                # Demo videos
```

---

## Installation

Create and activate a conda environment:
```bash
conda create -n pusht python=3.10 -y
conda activate pusht
```

Install dependencies:
```bash
pip install -r requirements.txt
pip install torch tensorboard
```

For video recording:
```bash
conda install -c conda-forge ffmpeg -y
```

---

## Training

```bash
python train.py
```

Trains for 3000 episodes by default. Checkpoints are saved every 100 episodes to `checkpoints/`. Training auto-resumes from the last checkpoint if one exists. Set `resume=False` in `train.py` to start from scratch.

Monitor training in TensorBoard:
```bash
tensorboard --logdir=logs --port=6006
```

Key metrics logged:
- `Reward/episode` — per-episode reward
- `Reward/mean_100` — rolling mean over the last 100 episodes
- `Loss/actor`, `Loss/critic1`, `Loss/critic2`
- `Alpha` — entropy temperature
- `Q/mean_q1`, `Q/mean_q2` — mean Q-values
- `Policy/log_prob` — policy entropy

---

## Evaluation

```bash
python evaluate.py
```

Loads the checkpoint from `checkpoints/` and runs the deterministic policy. Videos of each episode are saved to `videos/`.

---

## SAC Hyperparameters

| Parameter | Value |
|---|---|
| Hidden dim | 256 |
| Learning rate | 3e-4 |
| Discount (γ) | 0.99 |
| Soft update (τ) | 0.005 |
| Batch size | 256 |
| Replay buffer | 1M transitions |
| Start steps (random exploration) | 5000 |
| Target entropy | −dim(A) = −2 |
| Gradient clip norm | 1.0 |
| Reward scaling | ÷ 100 |

---

## Environment

- **Library:** `gym-pusht`  
- **Obs type:** `state` — `[agent_x, agent_y, block_x, block_y, block_angle]`, normalised to `[-1, 1]`  
- **Action space:** `[x, y]` target position, scaled from `[-1, 1]` → `[0, 512]`  
- **Reward:** Coverage of block in goal zone (0–1 per step, accumulated per episode)  
- **Success:** ≥ 95% block–goal overlap
