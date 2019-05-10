from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'done', 'next_state', 'reward'))

class ReplayBuffer(object):
    """
    Stores training samples for RL algorithms, represented as tuples of (S, A, R, S).
    """

    def __init__(self, max_size=1000):
        """
        Initialize the replay buffer object. Once the buffer is full, remove the oldest sample.
        :param max_size: maximum size of the buffer.
        """
        self.buffer = []
        self.max_size = max_size
        self.curr_size = 0
        self.position = 0

    def insert(self, sample):
        """
        Insert sample into buffer.
        :param sample: The (S,A,R,S) tuple.
        :return: None
        """
        if self.curr_size < self.max_size:
            self.buffer.append(None)
            self.curr_size = self.curr_size + 1

        self.buffer[self.position] = Transition(*sample)
        self.position = (self.position + 1) % self.max_size

    def sample(self, num_samples):
        """
        Sample a number of transitions from the replay buffer.
        :param num_samples: Number of transitions to sample.
        :return: The set of sampled transitions.
        """
        return random.sample(self.buffer, num_samples)

    def clear(self):
        """
        Clears the current buffer.
        :return: None
        """
        self.buffer = []
        self.curr_size = 0
        self.position = 0