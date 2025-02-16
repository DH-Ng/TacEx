# Example Commands
Some example commands for training etc. 

## Training

```bash
isaaclab -p ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ball-Rolling-IK-v0 --num_envs 1024 
```

## Other
You can start tensorboard with
```bash
isaaclab -p -m tensorboard.main serve --logdir /workspace/tacex/logs/rsl_rl/ball_rolling
```

You can debug RL training scripts by (for example) running the command
```bash
#python -m pip install --upgrade debugpy
lab -p -m debugpy --listen 3000 --wait-for-client _your_command_
``` 
and then attaching via VScode debugger.