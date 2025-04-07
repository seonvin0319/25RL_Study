# 🧊 FrozenLake-v1: SARSA vs Q-Learning with OpenAI Gym

OpenAI Gym의 `FrozenLake-v1` 환경에서 SARSA (on-policy)와 Q-Learning (off-policy)을 구현하고 비교 실험한 미니 프로젝트이이다.
FrozenLake는 구조가 단순한 환경으로, TD 학습 알고리즘의 학습 방식 차이를 비교하기에 적절하하다.
학습이 지나치게 오래 걸리는 문제를 완화하기 위해, 구멍(Hole)에 빠졌을 때 -1의 패널티를 주는 reward shaping을 적용하였고, 탐험을 충분히 유도하기 위해 ε(epsilon) 값을 상대적으로 크게 설정하였다.

📖 개념 설명 블로그 → https://van-liebling.tistory.com/39

## 📁 프로젝트 구조

```bash
.
├── algorithms/
│   └── td.py                     # SARSA와 Q-learning 업데이트트 구현
│
├── environment/
│   ├── gym_trainer.py            # Gym 환경 초기화, 정책책 학습 및 평가 유틸리티
│   └── gym_test.py               # 저장된 정책 시각화 (gif로 저장) 테스트용 스크립트
│
├── experiment/
│   └── run_dp.py                 # 전체 실험 실행 스크립트트
│
├── results/                      # 학습된 정책 및및 평가 결과 저장
│
└── setup/
    └── dp_arg.yaml               # 실험 파라미터 설정 파일
```

## 🚀 실행 방법

```bash
# [1] 전체 실행
# - FrozenLake 환경 초기화
# - SARSA 및 Q-learning 수행
# - 수렴된 시점의 policy 저장 (.npy)
# - episode 별 reward 결과 저장 (.csv)
# - setting은 setup/dp_arg.yaml 파일에서 변경 가능`

python -m experiment.run_td

# [2] 이미 학습된 정책을 기반으로 시뮬레이션만 수행
# - 저장된 policy(.npy)를 불러와서 실행
# - agent의 움직임을 .gif로 저장

python -m environment.gym_test
```

## 📘 코드 설명

### `algorithms/`

- **`SARSA`**  
  실제 탐험한 policy에 따라 Q의 값을 업데이트하는 on-policy 방식
  Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]

- **`value_iteration.py`**  
  가장 Q값이 높은 greedy policy를 기준으로 업데이트하는 off-policy 방식
  Q(s,a) ← Q(s,a) + α [r + γ max_a Q(s',a) - Q(s,a)]


### `environment/`

- **`gym_trainer.py`**  
  Gym 환경 초기화 및 평가 도구. policy를 여러 episode 동안 실행하여 평균 reward를 계산하고, CSV로 저장 가능.

- **`gym_test.py`**  
  이전에 저장된 policy를 바탕으로 실행하여 agent의 움직임을 시각화하고 `.gif` 파일로 저장하는 테스트용 스크립트.

### `experiment/`

- **`run_dp.py`**  
  실험 전체를 실행하는 메인 스크립트. 설정 로딩 → 알고리즘 실행 → 결과 저장 → 정책 평가.

### `setup/`

- **`dp_arg.yaml`**  
  환경 이름, 렌더링 모드, 하이퍼파라미터터 설정을 저장하는 YAML 파일.

---

## 📂 결과 확인

코드 실행이 끝나면 `results/` 폴더에 다음과 같은 결과 파일이 생성된다:

- **`sarsa_policy.npy`, `q_learning_policy.npy`**  
  수렴 시점의 policy matrix

- **`sarsa_eval.csv`, `q_learning_eval.csv`**  
  episode 별 reward을 기록한 평가 결과 파일

- **`gif/*.gif`**  
  `gym_test.py`를 통해 생성된 시각화 결과 (agent의 action sequence)

---

## 📊 결과 분석

| Metric | SARSA  | Q-Learning |
|--------|------------------|-----------------|
| Episode for Training | 1000 | 1000 |
| Success Ration during Training | `38.8%` | `38.3%` |
| First Success | 14th Episode | 39th Episode |
---

학습 조건: epsilon=0.5, alpha=0.1, gamma=0.99

reward shaping: 구멍(Hole)에 빠질 경우 reward = -1로 패널티 부여

환경 설정: is_slippery=False (deterministic dynamics)

SARSA는 on-policy 특성상 실제 행동을 따라 학습하므로
실패에 대한 반영이 빠르고, 학습 초기에 보수적으로 안정적인 수렴 경향을 보인다.
이로 인해 reward = 1.0을 더 빠르게 얻었으며, 초기 수렴 속도에서 우위를 보였다.

Q-Learning은 off-policy 방식으로, 항상 greedy한 기준으로 업데이트된다.
이는 이론적으로 더 빠르게 optimal policy에 도달할 수 있는 장점이 있지만,
reward를 처음 받기 전까지는 업데이트가 어려워 성공까지의 초기 진입 장벽이 높을 수 있다.

실험에서는 두 알고리즘의 성공률은 거의 비슷하지만,
SARSA가 더 빠르게 학습을 시작하고 평균 reward에서도 조금 더 높은 수치를 기록했다.

## 📚 참고 자료

- [OpenAI Gym Documentation](https://www.gymlibrary.dev/)
- Sutton & Barto, *Reinforcement Learning: An Introduction (2nd Edition)*  
  [공식 웹사이트 (책 전체 무료 제공)](http://incompleteideas.net/book/the-book-2nd.html)

