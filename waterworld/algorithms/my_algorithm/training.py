import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.common.utils import create_env, get_save_path
from algorithms.my_algorithm.my_algorithm import MyAlgorithm
from stable_baselines3.common.callbacks import EvalCallback

def train_my_algorithm(steps=500_000, seed=42):
    """MyAlgorithm 학습 함수"""
    paths = get_save_path('my_algorithm')
    os.makedirs(paths['model'], exist_ok=True)
    os.makedirs(paths['tensorboard'], exist_ok=True)

    env = create_env()
    eval_env = create_env(num_envs=4, num_cpus=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(paths['model'], "best"),
        log_path=paths['tensorboard'],
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    model = MyAlgorithm(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        batch_size=256,
        tensorboard_log=paths['tensorboard']
    )

    model.learn(
        total_timesteps=steps,
        callback=eval_callback,
        tb_log_name="MyAlgorithm"
    )

    model.save(os.path.join(paths['model'], "final_model"))

if __name__ == "__main__":
    train_my_algorithm() 