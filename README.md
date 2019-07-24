# Learning Multi-Agent Policies using GNNs
## Dependencies
- Python 3
- OpenAI Gym
- PyTorch
- [Gym-Flock](https://github.com/katetolstaya/gym-flock)

## Available algorithms:
- Behavior Cloning as described in [ArXiv](https://arxiv.org/abs/1903.10527) `python3 train.py cfg/cloning.cfg`
- DAGGER imitation learning `python3 train.py cfg/dagger.cfg`

## To test:
- `python3 test_model.py cfg/dagger.cfg`

## Other code:
- `python3 flocking_gym_test.py` provides test code for the Gym Flock environments


