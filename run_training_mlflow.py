import gym
import mlflow
import mlflow.pytorch
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from q_net import QNetwork


# Set random seed for reproducibility
r_seed = 42
random.seed(r_seed)
np.random.seed(r_seed)
torch.manual_seed(r_seed)


# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hyperparameters):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = hyperparameters["gamma"]
        self.lr = hyperparameters["lr"]
        self.batch_size = hyperparameters["batch_size"]
        self.device = hyperparameters["device"]
        self.memory = ReplayBuffer(self.action_size, hyperparameters["buffer_size"], self.batch_size, self.device)
        self.epsilon = hyperparameters["initial_epsilon"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.update_every = hyperparameters["update_every"]
        self.target_update_every = hyperparameters["target_update_every"]
        self.lr_decay = hyperparameters["lr_decay"]
        self.tau = hyperparameters["tau"]
        
        # Initialize main Q-Network
        self.m_qnetwork = QNetwork(state_size, action_size, hyperparameters["hidden_size"]).to(self.device)
        # Initialize target Q-Network
        self.t_qnetwork = QNetwork(state_size, action_size, hyperparameters["hidden_size"]).to(self.device)
        # Initialize target network weights
        self.t_qnetwork.load_state_dict(self.m_qnetwork.state_dict())
        # Initialize optimizer
        self.optimizer = optim.Adam(self.m_qnetwork.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=self.lr_decay)

        # Initialize time step (for UPDATE_EVERY steps)
        self.t_step = 0
        
    def act(self, state):
        """Returns action for given state as per current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        # Select action with max Q value
        with torch.no_grad():
            action_values = self.m_qnetwork(state)
        return torch.argmax(action_values).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stores experience in replay memory."""
        self.memory.add(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent based on the experiences in the replay memory."""
        if len(self.memory) < self.batch_size:
            return
        self.t_step += 1

        if self.t_step % self.update_every == 0:
            # Update the Q-Network
            # Sample a batch of experiences from replay memory
            states, actions, rewards, next_states, dones = self.memory.sample()

            q_values = self.m_qnetwork(states).gather(1, actions)
            next_q_values = self.t_qnetwork(next_states).max(1)[0].detach().unsqueeze(1)
            targets = rewards + (1 - dones) * self.gamma * next_q_values

            loss = F.mse_loss(q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Step learning rate scheduler
            self.scheduler.step()

            # Decay epsilon for better exploration-exploitation balance
            self.epsilon = self.epsilon * self.epsilon_decay

        if self.t_step % self.target_update_every == 0:
            # Update the target network
            for target_param, local_param in zip(self.t_qnetwork.parameters(), self.m_qnetwork.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


# Environment Interaction and Training
class LunarLanderTrainer:
    def __init__(self, hyperparameters):
        self.env = gym.make(hyperparameters["env"])
        self.agent = DQNAgent(state_size=self.env.observation_space.shape[0], 
                              action_size=self.env.action_space.n, 
                              hyperparameters=hyperparameters)
        self.episodes = hyperparameters["episodes"]
        self.score_individual = []
        self.average_score = []
        self.last_checkpoint_path = None
        
    def train(self):
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            total_reward = 0
            episode_over = False
            while not episode_over:
                action = self.agent.act(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.agent.store_experience(state, action, reward, next_state, done)
                self.agent.train()
                total_reward += reward
                episode_over = done or truncated
                state = next_state
            
            self.score_individual.append(total_reward)
            if episode % 100 == 0:
                average_score = np.mean(self.score_individual[-100:])
                print(f"Episode {episode}, Average Score: {average_score:.2f}")
                self.average_score.append(average_score)
            
        self.env.close()

        return self.score_individual, self.average_score


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


# Run Training
if __name__ == "__main__":    
    # Define Hyperparameters
    hyperparameter_grid = {
        "gamma": 0.99,
        "lr": [1e-3, 1e-4],
        "lr_decay": 0.95,
        "batch_size": [64, 128],
        "buffer_size": 100000,
        "initial_epsilon": [0.01, 0.1],
        "epsilon_decay": 0.995,
        "hidden_size": [64, 128],
        "episodes": 5000, 
        "update_every": 16,
        "target_update_every": [128, 256],
        "tau": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "env": "LunarLander-v2",
    }

    # Start an MLflow experiment
    experiment_name = hyperparameter_grid["env"]
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run_name = "first_run"
        
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        for lr in hyperparameter_grid["lr"]:
            for batch_size in hyperparameter_grid["batch_size"]:
                for initial_epsilon in hyperparameter_grid["initial_epsilon"]:
                    for hidden_size in hyperparameter_grid["hidden_size"]:
                        for target_update_every in hyperparameter_grid["target_update_every"]:
                            # Update chosen_hyperparameters with grid search parameters
                            with mlflow.start_run(nested=True):
                                current_hyperparameters = hyperparameter_grid.copy()
                                current_hyperparameters.update({
                                    "lr": lr,
                                    "batch_size": batch_size,
                                    "initial_epsilon": initial_epsilon,
                                    "hidden_size": hidden_size,
                                    "target_update_every": target_update_every
                                    })
                                print(f"Running with hyperparameters: {current_hyperparameters}")

                                # Log hyperparameters in MLflow
                                mlflow.log_params(current_hyperparameters)

                                # Create trainer and train the agent
                                trainer = LunarLanderTrainer(current_hyperparameters)
                                ind_score, avg_score = trainer.train()

                                # Log metrics in MLflow
                                for episode in range(len(avg_score)):
                                    mlflow.log_metric("average_score", avg_score[episode], step=episode)
                                for episode in range(len(ind_score)):
                                    mlflow.log_metric("score", ind_score[episode], step=episode)

                                # Log the final model as an artifact
                                mlflow.pytorch.log_model(trainer.agent.m_qnetwork, "model")