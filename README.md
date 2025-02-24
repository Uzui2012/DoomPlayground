# DoomPlayground
Using the VizDoom environment to showcase a variety of reinforcement learning models - making use of the gymnasium wrapper.

First is a simple DQN model agent, training for 1000 episodes on the VizdoomBasic-v0 environment - where the agent can move left, right, or shoot in a square box with 1 demon on the opposite side of the box to the agent. The agent is given +106 reward for killing the demon, -5 per shot, and -1 for each step; for a total of 300 steps (killing the demon terminates that episode). Next is to perform hyperparameter tuning for better results, then training on a variety of environments. 



After the DQN model agent will be the implementation of 2 further models of more recent research application to learn more up-to-date concepts, then finally an implementation of my undergraduate project idea.

---

![DQN Average Score VizdoomBasic Graph](https://github.com/Uzui2012/DoomPlayground/blob/main/rlqlearning/Recordings/avg_scores.png "DQN Average Score VizdoomBasic Graph")
![DQN Running Average Score VizdoomBasic Graph](https://github.com/Uzui2012/DoomPlayground/blob/main/rlqlearning/Recordings/running_avg_scores.png "DQN Running Average Score VizdoomBasic Graph")
![alt text](https://github.com/Uzui2012/DoomPlayground/blob/main/rlqlearning/Recordings/avg_scores.png?raw=true)
![DQN Duration VizdoomBasic Graph](https://github.com/Uzui2012/DoomPlayground/blob/main/rlqlearning/Recordings/duration.png "DQN Duration VizdoomBasic Graph")
![DQN Loss VizdoomBasic Graph](https://github.com/Uzui2012/DoomPlayground/blob/main/rlqlearning/Recordings/loss.png "DQN Loss VizdoomBasic Graph")