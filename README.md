# Worm Reinforcement Learning Project

A PyBullet-based reinforcement learning project where a segmented worm learns to navigate and collect food using natural undulating movements.

## Overview

This project implements a 6-segment worm robot in PyBullet that learns to move efficiently using reinforcement learning (PPO algorithm). The worm must learn to:
- Generate coordinated undulating movements
- Navigate towards food targets
- Maintain balance and stability
- Develop natural-looking locomotion

## Features

- 6-segment articulated worm with 5 joints
- Physics-based simulation using PyBullet
- PPO (Proximal Policy Optimization) implementation using Stable Baselines3
- Customizable reward system encouraging natural movement
- Real-time visualization of training progress
- Tensorboard integration for monitoring training metrics

## Requirements

- Python 3.8+
- PyBullet
- Stable Baselines3
- Gymnasium
- NumPy
- Tensorboard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/worm_rl.git
cd worm_rl
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install stable-baselines3 pybullet gymnasium numpy tensorboard
```

## Project Structure

```
worm_rl/
├── worm_rl/
│   ├── env/
│   │   └── worm_env.py      # Main environment implementation
│   ├── urdf/
│   │   └── worm.urdf        # Worm robot definition
│   └── train.py             # Training script
├── trained_models/          # Saved models and training logs
├── README.md
└── requirements.txt
```

## Usage

1. Start training:
```bash
python -m worm_rl.train
```

2. Monitor training progress with Tensorboard:
```bash
tensorboard --logdir ./trained_models/tensorboard_logs
```

## Environment Details

### Worm Robot
- 6 segments connected by 5 revolute joints
- Each joint rotates around the Y-axis for side-to-side motion
- Gradient coloring for easy segment identification
- Realistic physics parameters for stable movement

### Observation Space
- Joint positions and velocities (10 values)
- Base position and orientation (7 values)
- Food target position (3 values)

### Action Space
- 5 continuous actions controlling joint torques
- Range: [-1.0, 1.0] for each joint

### Rewards
- Distance to food reward
- Forward progress reward
- Undulation coordination reward
- Height maintenance reward
- Velocity control penalty
- Food collection bonus

## Training Parameters

The PPO agent uses the following key parameters:
- Learning rate: 3e-4
- Batch size: 64
- Training steps: 2M
- Episode length: 2000 steps
- Network architecture: [256, 256] for both policy and value networks

## Customization

You can modify various parameters in the environment:
- `max_steps`: Episode length
- `torque_scale`: Maximum torque applied to joints
- `max_joint_velocity`: Velocity limits for stability
- Reward weights and thresholds
- Physics parameters (friction, damping, etc.)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyBullet for the physics simulation
- Stable Baselines3 for the RL implementation
- The RL community for inspiration and best practices 