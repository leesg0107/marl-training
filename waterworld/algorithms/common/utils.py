import os
import supersuit as ss
from pettingzoo.sisl import waterworld_v4

def create_env(num_envs=8, num_cpus=2):
    """공통 환경 생성 함수"""
    env = waterworld_v4.parallel_env()
    
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