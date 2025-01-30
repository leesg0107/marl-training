import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.common.utils import create_env, get_save_path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

def train_sac(steps=500_000, seed=42):
    paths = get_save_path('sac')
    os.makedirs(paths['model'], exist_ok=True)
    os.makedirs(paths['tensorboard'], exist_ok=True)

    env = create_env()
    eval_env = create_env(num_envs=4, num_cpus=1)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(paths['model'], "best"),
        log_path=paths['tensorboard'],
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        action_noise=action_noise,
        ent_coef="auto",
        tensorboard_log=paths['tensorboard'],
        verbose=1
    )

    model.learn(
        total_timesteps=steps,
        callback=eval_callback,
        tb_log_name="SAC"
    )

    model.save(os.path.join(paths['model'], "final_model"))

if __name__ == "__main__":
    train_sac() 