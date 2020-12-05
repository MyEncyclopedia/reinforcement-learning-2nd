import cv2
import numpy as np
from collections import deque
from gym import make, ObservationWrapper, wrappers, Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace




class MaxAndSkipFrameWrapper(Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipFrameWrapper, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class FrameDownsampleWrapper(ObservationWrapper):
    def __init__(self, env):
        super(FrameDownsampleWrapper, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self._width = 84
        self._height = 84

    def observation(self, obs): # (240, 256, 3) => (84, 84, 1)
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class ImageToPyTorchWrapper(ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorchWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=(obs_shape[::-1]), dtype=np.float32)

    def observation(self, obs): # (84, 84, 1) => (1, 84, 84)
        return np.moveaxis(obs, 2, 0)


class FrameBufferWrapper(ObservationWrapper):
    def __init__(self, env, num_steps, dtype=np.float32):
        super(FrameBufferWrapper, self).__init__(env)
        obs_space = env.observation_space
        self._dtype = dtype
        self.observation_space = Box(obs_space.low.repeat(num_steps, axis=0),
                                     obs_space.high.repeat(num_steps, axis=0),
                                     dtype=self._dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self._dtype)
        return self.observation(self.env.reset())

    def observation(self, obs): # (1, 84, 84) => (4, 84, 84)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs
        return self.buffer


class NormalizeFloats(ObservationWrapper):
    def observation(self, obs): # (4, 84, 84)
        return np.array(obs).astype(np.float32) / 255.0


class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, done, info
        

def wrap_environment(env_name: str, action_space: list, monitor=False, iteration=0) -> Wrapper:
    env = make(env_name)
    if monitor:
        env = wrappers.Monitor(env, 'recording/run%s' % iteration, force=True)
    env = JoypadSpace(env, action_space)
    env = MaxAndSkipFrameWrapper(env)
    env = FrameDownsampleWrapper(env)
    env = ImageToPyTorchWrapper(env)
    env = FrameBufferWrapper(env, 4)
    env = NormalizeFloats(env)
    env = CustomReward(env)
    return env
