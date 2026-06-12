import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback

env_id = "PandaPickAndPlace-v3"
env = gym.make(env_id)

eval_env = gym.make(env_id)

# Set up an EvalCallback to save the best model automatically as it trains
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/results/",
    eval_freq=50_000, # Evaluate every 50,000 steps
    deterministic=True,
    render=False,
)

print(f"Initializing SAC with HER for {env_id}...")

# 3. Configure SAC with Hindsight Experience Replay (HER)
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4, # For every real transition, create 4 fake "successful" ones
        goal_selection_strategy="future",
    ),
    learning_rate=1e-3,
    buffer_size=1_000_000, # Store 1 million steps in memory
    batch_size=256,
    tau=0.05,
    gamma=0.95,           # Discount factor for future rewards
    ent_coef="auto",      # Automatically adjust entropy (randomness)
    policy_kwargs=dict(net_arch=[256, 256, 256]), # A slightly deeper neural net
    verbose=1,
    device="auto"         # Uses GPU if you have one, otherwise CPU
)

# 4. Train the model for 1,000,000 steps
print("Starting training for 1,000,000 steps...")
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback,
    progress_bar=True
)

# 5. Save the final model
model.save("sac_panda_pick_and_place_1M")
print("Training complete! Model saved as 'sac_panda_pick_and_place_1M.zip'")

env.close()
eval_env.close()