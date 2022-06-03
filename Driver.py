import vizdoom as vzd
from vizdoom import DoomGame
from random import choice
from time import sleep, time

from DQN import DQN_agent


game = vzd.DoomGame()
game.load_config("C:/Users/killi/ViZDoom/scenarios/basic.cfg")
game.init()

actions = [
    [True, False, False],
    [False, True, False],
    [False, False, True]
]

dqn_agent = DQN_agent(3,1)

for i in range(10):
    game.new_episode()
    while not game.is_episode_finished():
        s = game.get_state()
        print(s)
        img = s.screen_buffer
        misc = s.game_variables

        action = dqn_agent.select_action(s)
        reward = game.make_action(action)

    print(f"total reward:     {game.get_total_reward()}")

