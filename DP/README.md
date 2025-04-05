# 🧊 FrozenLake-v1: Policy Iteration vs Value Iteration with OpenAI Gym

이 프로젝트는 OpenAI Gym의 `FrozenLake-v1` 환경에서 Dynamic Programming 기법인 **Policy Iteration**과 **Value Iteration**을 구현하고, 두 알고리즘의 성능을 비교한 결과를 정리한 것이다.  
강화학습의 기초 개념을 직접 코드로 구현하고 실습하는 데 중점을 둔 미니 프로젝트다.

📖 개념 설명 블로그 → https://van-liebling.tistory.com/38

## 📁 프로젝트 구조

```bash
.
├── algorithms/
│   ├── policy_iteration.py       # Policy Iteration 알고리즘 구현
│   └── value_iteration.py        # Value Iteration 알고리즘 구현
│
├── environment/
│   ├── gym_trainer.py            # Gym 환경 초기화 및 정책 평가 유틸리티
│   └── gym_test.py               # 정책을 시각화(gif)로 저장하는 테스트용 스크립트
│
├── experiment/
│   └── run_dp.py                 # 전체 실험 실행 (학습 + 저장 + 평가)
│
├── results/                      # 학습된 정책, 가치 함수, 평가 결과 저장 폴더
│
└── setup/
    └── dp_arg.yaml               # 실험 파라미터 설정 파일
```

## 🚀 실행 방법

```bash
# [1] 전체 실행
# - FrozenLake 환경 초기화
# - Policy Iteration 및 Value Iteration 수행
# - 수렴된 시점의 policy와 value 저장 (.npy)
# - episode 별 reward 결과 저장 (.csv)
# - setting은 setup/dp_arg.yaml 파일에서 변경 가능`

python -m experiment.run_dp

# [2] 이미 학습된 정책을 기반으로 시뮬레이션만 수행
# - 저장된 policy(.npy)를 불러와서 실행
# - agent의 움직임을 .gif로 저장

python -m environment.gym_test
```

## 📘 코드 설명

### `algorithms/`

- **`policy_iteration.py`**  
  Policy Evaluation과 Policy Improvement를 반복하여 optimal policy을 구하는 알고리즘 구현.

- **`value_iteration.py`**  
  Optimal Bellman Equation을 이용해 value function을 수렴시키고, 이를 통해 optimal policy을 도출하는 알고리즘 구현.

### `environment/`

- **`gym_trainer.py`**  
  Gym 환경 초기화 및 평가 도구. 수렴한 policy를 여러 episode 동안 실행하여 평균 reward와 수렴 속도를 계산하고, CSV로 저장 가능.

- **`gym_test.py`**  
  이전에 저장된 policy를 바탕으로 3번의 episode 수만큼 실행하여 agent의 움직임을 시각화하고 `.gif` 파일로 저장하는 테스트용 스크립트.

### `experiment/`

- **`run_dp.py`**  
  실험 전체를 실행하는 메인 스크립트. 설정 로딩 → 알고리즘 실행 → 결과 저장 → 정책 평가.

### `setup/`

- **`dp_arg.yaml`**  
  환경 이름, 렌더링 모드, gym_trainer.py의 test episode 수 설정을 저장하는 YAML 파일.

---

## 📂 결과 확인

코드 실행이 끝나면 `results/` 폴더에 다음과 같은 결과 파일이 생성된다:

- **`policy_iteration_policy.npy`, `value_iteration_policy.npy`**  
  수렴 시점의 policy matrix

- **`policy_iteration_value.npy`, `value_iteration_value.npy`**  
  각 상태에 대한 value function

- **`policy_iteration_eval.csv`, `value_iteration_eval.csv`**  
  episode 별 reward을 기록한 평가 결과 파일

- **`gif/test_gif/*.gif`**  
  `gym_test.py`를 통해 생성된 시각화 결과 (agent의 action sequence)

---

## 📊 결과 분석

| Metric | Policy Iteration | Value Iteration |
|--------|------------------|-----------------|
| Convergence Time | 73.4 ms | 70.3 ms |
| Final Value (state 14) | `v[14] = 0.7988` | `v[14] = 0.9411` |
| Policy Update Style | Explicit policy improvement | Value-based greedy extraction |
| Average Reward | 0.7200 (over 1000 episodes) | 0.7400 (over 1000 episodes) |
---
Policy Iteration은 한 번에 정책 전체를 개선하는 구조이기 때문에 빠르게 수렴하며, 실험에서도 매우 짧은 시간 내에 안정적인 정책을 도출했다.

반면, Value Iteration은 상태 가치(value)를 반복적으로 업데이트하면서 수렴해가며, 이론적으로는 더 많은 반복이 필요할 수 있지만, FrozenLake와 같이 작은 환경에서는 실행 시간이 거의 차이가 없었다.

최종 value function 상에서 Value Iteration의 값(v[14])이 더 높게 수렴한 것은 Bellman Optimality Equation을 직접 최적화하는 방식이기 때문으로 해석할 수 있다.

평균 보상 기준으로도 Value Iteration이 약간 우세했지만, 두 알고리즘 모두 0.7 이상의 높은 성능을 보여주었다.

추가적으로, Value Iteration은 **정책 평가와 개선 단계를 분리하지 않아 구현이 간단**하고, 가치 함수를 직접 갱신하기 때문에 **빠르게 수렴하는 경향이 있다.**  
다만, 상태 공간이 클 경우 **모든 상태에 대해 최대값 연산을 반복 수행해야 하므로 계산 비용이 높아질 수 있고**,  
정확한 가치 수렴을 기다려야 하는 경우에는 **정책 반복보다 더 많은 반복이 필요할 수도 있다.**

아래 GIF는 저장된 policy를 기반으로 FrozenLake 환경에서 agent가 실행한 결과를 시각화한 것이다.

![FrozenLake Policy Execution](./results/gif/policy_iteration_episode_2.gif)

## 📚 참고 자료

- [OpenAI Gym Documentation](https://www.gymlibrary.dev/)
- Sutton & Barto, *Reinforcement Learning: An Introduction (2nd Edition)*  
  [공식 웹사이트 (책 전체 무료 제공)](http://incompleteideas.net/book/the-book-2nd.html)

