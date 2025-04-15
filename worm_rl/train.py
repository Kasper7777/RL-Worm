from stable_baselines3 import PPO
from env.worm_env import WormEnv
import os
from datetime import datetime
import torch
from stable_baselines3.common.callbacks import EvalCallback
import time

def main():
    # Create directories first
    os.makedirs("./trained_models", exist_ok=True)
    os.makedirs("./trained_models/tensorboard_logs", exist_ok=True)
    os.makedirs("./trained_models/best_model", exist_ok=True)
    os.makedirs("./trained_models/eval_logs", exist_ok=True)

    print("Initializing environment...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            env = WormEnv(render=True)
            eval_env = WormEnv(render=False)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("Failed to initialize environment after multiple attempts")
                return

    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            ),
            activation_fn=torch.nn.ReLU
        ),
        tensorboard_log="./trained_models/tensorboard_logs",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./trained_models/best_model",
        log_path="./trained_models/eval_logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print("\nStarting training...")
    print("You should see a PyBullet window with the worm robot.")
    print("The window will stay open during training.")
    print("Press Ctrl+C to stop training early.\n")

    try:
        # Small delay to ensure window is ready
        time.sleep(1)
        
        model.learn(
            total_timesteps=2000000,
            callback=eval_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Attempting to save current model state...")

    try:
        # Save the final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./trained_models/worm_ppo_{timestamp}"
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\nTesting trained agent...")
    try:
        obs, _ = env.reset()
        total_reward = 0
        food_reached = 0
        
        for step in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if reward > 5.0:
                food_reached += 1
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Food reached: {food_reached} times")
                obs, _ = env.reset()
                total_reward = 0
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        print("\nClosing environments...")
        try:
            env.close()
            eval_env.close()
        except:
            pass

if __name__ == "__main__":
    main() 