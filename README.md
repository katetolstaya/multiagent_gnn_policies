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

## Citing the Project
To cite this repository in publications:
```shell
@inproceedings{tolstaya2020learning,
  title={Learning decentralized controllers for robot swarms with graph neural networks},
  author={Tolstaya, Ekaterina and Gama, Fernando and Paulos, James and Pappas, George and Kumar, Vijay and Ribeiro, Alejandro},
  booktitle={Conference on Robot Learning},
  pages={671--682},
  year={2020}
}
```
