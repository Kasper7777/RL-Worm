from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from env.worm_env import WormEnv
import os
import yaml
import torch
import numpy as np
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load training configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(config_path):
        # Default configuration
        config = {
            'training': {
                'total_timesteps': 2_000_000,
                'seed': 42,
                'n_envs': 1
            },
            'ppo': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'policy_kwargs': {
                    'net_arch': {
                        'pi': [256, 256],
                        'vf': [256, 256]
                    }
                }
            },
            'eval': {
                'eval_freq': 10000,
                'n_eval_episodes': 5
            }
        }
        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    return config

def make_env(render=False, seed=0):
    """Create a wrapped, monitored environment."""
    def _init():
        env = WormEnv(render=render)
        env = Monitor(env)
        # Use the new Gymnasium seeding API
        env.reset(seed=seed)
        return env
    return _init

def setup_training_dirs():
    """Create necessary directories for training artifacts."""
    dirs = {
        'base': './trained_models',
        'tensorboard': './trained_models/tensorboard_logs',
        'checkpoints': './trained_models/checkpoints',
        'best_model': './trained_models/best_model',
        'eval_logs': './trained_models/eval_logs'
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def main():
    try:
        # Load configuration
        config = load_config()
        dirs = setup_training_dirs()
        
        # Set seeds for reproducibility
        set_random_seed(config['training']['seed'])
        
        # Create vectorized environment
        logger.info("Initializing environments...")
        env = DummyVecEnv([make_env(render=True, seed=config['training']['seed'])])
        eval_env = DummyVecEnv([make_env(render=False, seed=config['training']['seed'] + 1)])
        
        # Normalize observations and rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        
        # Create PPO agent
        logger.info("Creating PPO agent...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=dirs['tensorboard'],
            device="cuda" if torch.cuda.is_available() else "cpu",
            **config['ppo']
        )
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=dirs['best_model'],
            log_path=dirs['eval_logs'],
            eval_freq=config['eval']['eval_freq'],
            n_eval_episodes=config['eval']['n_eval_episodes'],
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(10000, config['eval']['eval_freq']),
            save_path=dirs['checkpoints'],
            name_prefix="worm_model"
        )
        
        logger.info("Starting training...")
        try:
            model.learn(
                total_timesteps=config['training']['total_timesteps'],
                callback=[eval_callback, checkpoint_callback],
                progress_bar=True
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(dirs['base'], f"worm_ppo_final_{timestamp}")
        model.save(final_model_path)
        env.save(os.path.join(dirs['base'], f"vec_normalize_{timestamp}.pkl"))
        logger.info(f"Final model saved to {final_model_path}")
        
        # Test trained agent
        logger.info("Testing trained agent...")
        mean_reward = 0
        n_eval_episodes = 5
        
        for episode in range(n_eval_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
                
                if done:
                    logger.info(f"Episode {episode + 1} reward: {episode_reward:.2f}")
                    mean_reward += episode_reward
                    break
        
        logger.info(f"Mean reward over {n_eval_episodes} episodes: {mean_reward/n_eval_episodes:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        logger.info("Cleaning up...")
        if 'env' in locals():
            env.close()
        if 'eval_env' in locals():
            eval_env.close()

if __name__ == "__main__":
    main() 