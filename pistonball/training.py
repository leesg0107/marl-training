"""Uses Ray's RLlib to train agents to play Leduc Holdem.

Author: Rohan (https://github.com/Rohan138)
"""

import os
import numpy as np

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from gymnasium.spaces import Box

class PistonballEnvWrapper(MultiAgentEnvCompatibility):
    def reset(self, *, seed=None, options=None):
        obs_tuple = self.env.reset(seed=seed, options=options)
        if isinstance(obs_tuple, tuple):
            obs = obs_tuple[0]  # 첫 번째 요소가 observation
            # 각 에이전트에 대한 빈 info 딕셔너리 생성
            infos = {agent_id: {} for agent_id in range(20)}  # 20 pistons
            return obs, infos
        return obs_tuple

def env_creator(config):
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
    )
    
    # 환경 전처리
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='gymnasium')
    # 커스텀 호환성 래퍼 사용
    env = PistonballEnvWrapper(env)
    return env

if __name__ == "__main__":
    # Ray 초기화 시 옵션 추가
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,
        local_mode=False
    )

    try:
        # 환경 등록
        env_name = "pistonball_v6"
        register_env(env_name, env_creator)

        # 결과 저장 경로 설정
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)

        # PPO 설정
        config = (
            PPOConfig()
            .environment(
                env=env_name,
                disable_env_checking=True
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=2,  # worker 수 감소
                rollout_fragment_length=128,
                num_envs_per_worker=1,  # 명시적으로 설정
            )
            .training(
                train_batch_size=1024,  # 배치 크기 감소
                lr=2.5e-4,
                gamma=0.99,
                lambda_=0.95,
                use_gae=True,
                clip_param=0.2,
                model={
                    "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [256, [11, 11], 1]],
                    "post_fcnet_hiddens": [256, 256],
                }
            )
            .multi_agent(
                policies=["shared_policy"],
                policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy")
            )
            .resources(
                num_gpus=0,  # GPU 사용하지 않음
                num_cpus_per_worker=1,  # worker당 CPU 수 제한
            )
            .debugging(
                log_level="WARNING",  # 로그 레벨 변경
                seed=42  # 재현성을 위한 시드 설정
            )
        )

        # 학습 실행
        tune.run(
            "PPO",
            name="PPO",
            stop={"timesteps_total": 5000000},
            checkpoint_freq=10,
            config=config.to_dict(),
            local_dir=results_dir,
            verbose=1  # 진행 상황 출력
        )

    except Exception as e:
        print(f"Error occurred: {e}")
        raise e

    finally:
        # Ray 종료
        ray.shutdown()