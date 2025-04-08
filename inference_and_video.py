import gym
import mlflow.pytorch
import os
import torch

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from q_net import QNetwork
import torch.optim as optim


# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hyperparameters):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = hyperparameters["gamma"]
        self.lr = hyperparameters["lr"]
        self.batch_size = hyperparameters["batch_size"]
        self.device = hyperparameters["device"]
        
        # Initialize main Q-Network
        self.m_qnetwork = QNetwork(state_size, action_size, hyperparameters["hidden_size"]).to(self.device)
        # Initialize target Q-Network
        self.t_qnetwork = QNetwork(state_size, action_size, hyperparameters["hidden_size"]).to(self.device)
        # Initialize target network weights
        self.t_qnetwork.load_state_dict(self.m_qnetwork.state_dict())
        # Initialize optimizer
        self.optimizer = optim.Adam(self.m_qnetwork.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
    def act(self, state):
        """Returns action for given state."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_values = self.m_qnetwork(state)
        return torch.argmax(action_values).item()
    

def run_inference_and_save_video(agent, env_name, checkpoint_path, folder_name, hyperparameters):
    """
    Run inference of the trained agent and save video with hyperparameters in the filename.
    """
    os.makedirs(folder_name, exist_ok=True)
    checkpoint_path = os.path.normpath(checkpoint_path)
    
    # Initialize the video recorder
    video_path = os.path.join(
        folder_name,
        (
            f"lr_{hyperparameters['lr']}_"
            f"bat_{hyperparameters['batch_size']}_"
            f"inite_{hyperparameters['initial_epsilon']}_"
            f"hiddens_{hyperparameters['hidden_size']}_"
            f"episode_{hyperparameters['episodes']}_"
            f"targetup_{hyperparameters['target_update_every']}.mp4"    
        )
    )
    # Initialize the environment with render_mode='rgb_array'
    env = gym.make(env_name, render_mode="rgb_array")
    vid = VideoRecorder(env, path=video_path)
    
    # Load the trained model
    agent.m_qnetwork = mlflow.pytorch.load_model(checkpoint_path)
    
    # Reset the environment
    state, _ = env.reset()  # Gym >=0.26 returns (state, info)
    done = False
    
    # Run the agent in the environment and record the video
    try:
        while not done:
            # Capture each frame for the video
            vid.capture_frame()  
            
            # Get the action from the agent
            action = agent.act(state)
            
            # Perform the action and get the next state
            state, _, terminated, truncated, _ = env.step(action)
            
            # Check if the episode is done
            done = terminated or truncated

    finally:
        # Make sure the video and environment are closed properly
        vid.close()
        env.close()


if __name__ == "__main__":
    # Specify the checkpoint path (after training)
    checkpoint_path = r"mlruns/733636240066742753/a6b8c6e263554ba4bd120d5b3585c5c7/artifacts/model/"  # Update this path with your actual checkpoint path
    
    # Define the hyperparameters for this inference session 
    hyperparameters = {
        "gamma": 0.99,
        "lr": 1e-3,
        "lr_decay": 0.95,
        "batch_size": 128,
        "buffer_size": 100000,
        "initial_epsilon": 0.1,
        "epsilon_decay": 0.995,
        "hidden_size": 128,
        "episodes": 1500,
        "update_every": 16,
        "target_update_every": 256,
        "tau": 1e-3,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "env": "LunarLander-v2",
    }

    # -----------------------------------------------------------------------------------------------------------------------#
    # Define path to save the video 
    folder_name = os.path.join("videos", hyperparameters["env"])

    # Initialize the agent
    env = gym.make(hyperparameters["env"])
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, hyperparameters=hyperparameters)

    # Start the inference and save video
    run_inference_and_save_video(agent, hyperparameters["env"], checkpoint_path, folder_name, hyperparameters)
    print("Inference completed and video saved.")