import gymnasium as gym
import gym_pusht
import time


## Visualize environment
if __name__ == "__main__":
    try:
        env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="human")
        observation, info = env.reset()

        for _ in range(1000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
            image = env.render()

            if terminated or truncated:
                observation, info = env.reset()

        env.close()

    except gym.error.Error as e:
        print(f"Gym error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")