import os
import supersuit as ss
from pettingzoo.sisl import multiwalker_v9

def create_env(num_envs=8, num_cpus=2):
    """공통 환경 생성 함수"""
    env = multiwalker_v9.parallel_env(
        n_walkers=3,
        position_noise=1e-3,
        angle_noise=1e-3,
        forward_reward=1.0,
        terminate_reward=-100.0,
        fall_reward=-10.0,
        shared_reward=True,
        max_cycles=500,
        terminate_on_fall=True,
        remove_on_fall=True
    )
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    return env

def get_save_path(algo_name):
    """알고리즘별 저장 경로 생성"""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {
        'model': os.path.join(base_path, 'results', algo_name),
        'tensorboard': os.path.join(base_path, 'tensorboard', algo_name)
    } 