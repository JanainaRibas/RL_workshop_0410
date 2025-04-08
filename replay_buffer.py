from collections import deque, namedtuple
import numpy as np
import random
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object."""        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor(np.array([e.state for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([e.action for e in experiences]), dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(np.array([e.reward for e in experiences]), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([e.done for e in experiences]), dtype=torch.float32, device=self.device).unsqueeze(1)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)