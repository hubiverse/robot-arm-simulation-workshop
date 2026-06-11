import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

env_id = "PandaReachDense-v3"
env = gym.make(env_id)
eval_env = gym.make(env_id)

# Set up the EvalCallback to save the best version
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/ppo_reach_best_model/",
    log_path="./logs/ppo_reach_results/",
    eval_freq=20_000,  # Evaluate it every 20,000 steps
    deterministic=True,
    render=False,
)

print(f"Initializing PPO for {env_id}...")

# Configure PPO
model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    verbose=1,
    device="auto"
)

# Train the model for 1,000,000 steps
print("Starting training for 1,000,000 steps...")
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback,
    progress_bar=True
)

# Save the final model
model.save("ppo_panda_reach_dense_1M")
print("Training complete! Backup model saved as 'ppo_panda_reach_dense_1M.zip'")

env.close()
eval_env.close()