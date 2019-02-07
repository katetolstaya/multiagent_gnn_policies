### Description
Reimplementation of [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/pdf/1603.00748v1.pdf) and [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf).

Completely based on [Ilya Kostrikov's Implementation](https://github.com/ikostrikov/pytorch-ddpg-naf)

Extended to run on a [multi-agent flocking environment](https://github.com/katetolstaya/multiagent_ddpg)

### Dependencies
- PyTorch
- TensorboardX
- OpenAI Gym
- [Multi-agent flocking Gym environment](https://github.com/katetolstaya/gym_flock.git)
- TQDM

### Run
Use the default hyperparameters.

#### For NAF:

```
python main.py --algo NAF
```
#### For DDPG

```
python main.py --algo DDPG
```
