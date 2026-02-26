from sac.agent import SACAgent
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from sac.utils import normalize_state, scale_action_to_env
import gym_pusht


if __name__ == "__main__":
    agent = SACAgent(6, 2, 256)
    agent.load("checkpoints")

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)

    state, _ = env.reset()
    norm_state = normalize_state(state)

    for _ in range(1000):
        action = agent.select_action(norm_state, deterministic=True)  # [-1, 1]
        env_action = scale_action_to_env(action)                      # [0, 512]

        observation, reward, terminated, truncated, info = env.step(env_action)
        norm_state = normalize_state(observation)

        if terminated or truncated:
            observation, info = env.reset()
            norm_state = normalize_state(observation)

    env.close()
    print("Videos saved to ./videos/")