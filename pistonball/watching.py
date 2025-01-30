"""Uses Ray's RLlib to view trained agents playing Leduoc Holdem.

Author: Rohan (https://github.com/Rohan138)
"""

import argparse
import os

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss

parser = argparse.ArgumentParser(
    description="학습된 정책을 체크포인트에서 불러와 렌더링"
)
parser.add_argument(
    "--checkpoint-path",
    help="체크포인트 경로. 예시: `~/ray_results/PPO/PPO_pistonball_v6_XXXXX/checkpoint_000050/checkpoint-50`",
)

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.checkpoint_path is None:
        print("다음 인자가 필요합니다: --checkpoint-path")
        exit(0)

    checkpoint_path = os.path.expanduser(args.checkpoint_path)
    
    ray.init()
    
    # 학습된 에이전트 불러오기
    agent = Algorithm.from_checkpoint(checkpoint_path)

    # 환경 생성
    env = pistonball_v6.env(render_mode="human")
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)

    # 평가 실행
    reward_sums = {agent_id: 0 for agent_id in env.possible_agents}
    env.reset()

    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        reward_sums[agent_id] += reward
        
        if termination or truncation:
            action = None
        else:
            action = agent.compute_single_action(
                observation, 
                policy_id="shared_policy"
            )
        
        env.step(action)
        env.render()

    print("Total rewards:", reward_sums)
    env.close()