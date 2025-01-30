# MARL Training Project

이 프로젝트는 PettingZoo를 사용한 멀티에이전트 강화학습(MARL) 실험을 위한 코드베이스입니다.

## 환경 설정

```bash
pip install -r requirements.txt
```

## 프로젝트 구조
```
.
├── waterworld/          # Waterworld 환경 실험
├── pistonball/         # Pistonball 환경 실험
└── multiwalker/        # Multiwalker 환경 실험
```

## 실행 방법

각 환경별로 다음과 같이 실행할 수 있습니다:

### Waterworld
```bash
# PPO 알고리즘 학습
python waterworld/algorithms/ppo/training.py

# SAC 알고리즘 학습
python waterworld/algorithms/sac/training.py

# 텐서보드 실행
python waterworld/view_tensorboard.py
```

## 주요 기능

- 다양한 MARL 환경 지원 (Waterworld, Pistonball, Multiwalker)
- PPO, SAC 등 다양한 알고리즘 구현
- 텐서보드를 통한 학습 모니터링
- 모델 저장 및 평가 기능