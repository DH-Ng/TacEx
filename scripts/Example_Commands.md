# Example Commands
Some example commands for training etc.

To list all TacEx environemnts:
```bash
# Assuming you are in the TacEx root directory
isaaclab -p scripts/reinforcement_learning/list_envs.py
```

## Training

```bash
isaaclab -p ./scripts/reinforcement_learning/rsl_rl/train.py --task TacEx-Ball-Rolling-IK-v0 --num_envs 1024 
```

```bash
isaaclab -p ./scripts/reinforcement_learning/rsl_rl/train.py --task TacEx-Ball-Rolling-Privileged-v0 --num_envs 1024 
```

```bash
isaaclab -p ./scripts/reinforcement_learning/rsl_rl/train.py --task TacEx-Ball-Rolling-Privileged-without-Reach_v0 --num_envs 1024 --enable_cameras
```

## Play
```bash
isaaclab -p ./scripts/reinforcement_learning/rsl_rl/play.py --task TacEx-Ball-Rolling-Tactile-Base-v1 --num_envs 23 --enable_cameras --load_run logs/skrl/ball_rolling/2025-04-08_22-55-53_improved_ppo_torch_base_env_cluster --checkpoint best_agent.pt
```

```bash
isaaclab -p ./scripts/reinforcement_learning/skrl/play.py --task TacEx-Ball-Rolling-Tactile-Base-v1 --num_envs 23 --enable_cameras --checkpoint logs/skrl/ball_rolling/2025-04-15_00-04-33_full_resets_ppo_torch_base_env/checkpoints/best_agent.pt
```


## Other
You can activate tensorboard with
```bash
isaaclab -p -m tensorboard.main serve --logdir /workspace/tacex/logs/rsl_rl/ball_rolling
isaaclab -p -m tensorboard.main serve --logdir /workspace/tacex/logs/skrl/ball_rolling
```

You can debug RL training scripts by (for example) running the command
```bash
#python -m pip install --upgrade debugpy
lab -p -m debugpy --listen 3000 --wait-for-client _your_command_
``` 
and then attaching via VScode debugger.