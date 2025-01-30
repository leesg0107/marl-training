import os
import supersuit as ss
from pettingzoo.butterfly import pistonball_v6

def create_env(num_envs=8, num_cpus=2):
    """공통 환경 생성 함수"""
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=900,
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