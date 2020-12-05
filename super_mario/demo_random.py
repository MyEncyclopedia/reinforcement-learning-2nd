
import gym_super_mario_bros
from random import random, randrange
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym import wrappers


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrappers.Monitor(env, 'demo', force=True)

# Play randomly
done = False
env.reset()

step = 0
while not done:
    action = randrange(len(RIGHT_ONLY))
    state, reward, done, info = env.step(action)
    if step > 400:
        env.close()
    print(done, step, info)
    env.render()
    step += 1

env.close()