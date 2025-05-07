import gym
import time

# 환경 생성
env = gym.make("Hopper-v4", render_mode="human")

# 환경 초기화
obs, info = env.reset()

# 100 step 동안 random action 해보기
for _ in range(100):
    action = env.action_space.sample()  # 무작위 액션
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.05)  # 렌더링 부드럽게 보이도록 약간의 딜레이

    if terminated or truncated:
        obs, info = env.reset()

env.close()
