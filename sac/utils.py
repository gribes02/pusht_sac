

from math import sin, cos


def normalize_state(s):
    """Normalize the state to be between -1 and 1.
    From the original state space of [0, 512] for position and [0, 2*pi] for angles.
    
    s:
    [agent_x, agent_y, object_x, object_y, object_angle]
    """
    normalized_state = [0] * 6
    normalized_state[0] = (s[0] - 256) / 256  # Normalize x position
    normalized_state[1] = (s[1] - 256) / 256  # Normalize y position
    normalized_state[2] = (s[2] - 256) / 256  # Normalize object x position
    normalized_state[3] = (s[3] - 256) / 256  # Normalize object y position
    theta = s[4]
    x_angle = sin(theta)  # Normalize angle using sine
    y_angle = cos(theta)  # Normalize angle using cosine
    normalized_state[4] = x_angle
    normalized_state[5] = y_angle  # Append the cosine of the angle to the state
    return normalized_state

def scale_action_to_env(a_tanh):
    """Scale the action from [-1, 1] to the environment's action space [0, 512].
    
    action:
    [delta_x, delta_y]
    """
    action_scaled = [0, 0]
    action_scaled[0] = (a_tanh[0] + 1) / 2 * 512  # Scale x movement
    action_scaled[1] = (a_tanh[1] + 1) / 2 * 512  # Scale y movement
    return action_scaled

def scale_action_to_tanh(a_env):
    """Scale the action from the environment's action space to [0, 512].
    The original action space is [-1, 1] for both x and y movements.
    
    action:
    [delta_x, delta_y]
    """
    scaled_action = [0, 0]
    scaled_action[0] = (a_env[0] / 512) * 2 - 1  # Scale x movement
    scaled_action[1] = (a_env[1] / 512) * 2 - 1  # Scale y movement
    return scaled_action