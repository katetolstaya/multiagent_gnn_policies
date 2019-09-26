import numpy as np


class OUNoise:
    """
    Generates noise from an Ornstein Uhlenbeck process, for temporally correlated exploration. Useful for physical
    control problems with inertia. See https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, n_a, n_agents, scale=0.05, mu=0, theta=0.15, sigma=0.2):  # TODO change default params
        """
        Initialize the Noise parameters.
        :param n_a: Size of the Action space.
        :param n_agents: Number of agents.
        :param scale: Scale of the noise process.
        :param mu: The mean of the noise.
        :param theta: Inertial term for drift..
        :param sigma: Standard deviation of noise.
        """
        self.nA = n_a
        self.n_agents = n_agents
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones((self.n_agents, self.nA)) * self.mu
        self.reset()

    def reset(self):
        """
        Reset the noise process.
        :return:
        """
        self.state = np.ones((self.n_agents, self.nA)) * self.mu

    def noise(self):
        """
        Compute the next noise value.
        :return: The noise value.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(np.shape(x))
        self.state = x + dx
        return self.state * self.scale