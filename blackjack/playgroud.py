from gym.envs.toy_text import BlackjackEnv


def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))

def strategy(observation):
    score, dealer_score, usable_ace = observation
    # Stick (action 0) if the score is > 20, hit (action 1) otherwise
    return 0 if score >= 20 else 1

if __name__ == "__main__":
    env = BlackjackEnv()
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            print_observation(observation)
            action = strategy(observation)
            print("Taking action: {}".format( ["Stick", "Hit"][action]))
            observation, reward, done, _ = env.step(action)
            if done:
                print_observation(observation)
                print("Game end. Reward: {}\n".format(float(reward)))
                break