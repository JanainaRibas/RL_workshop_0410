---

# Solving Lunar Lander with DQN

This project demonstrates how to solve the **Lunar Lander** environment using **Deep Q-Networks (DQN)**. It includes the training process, hyperparameter tuning, result visualization with **MLflow**, and video creation.

---

## Table of Contents

1. [Installation](#installation)
2. [Preparing the Training Script](#preparing-the-training-script)
3. [Training the Model](#training-the-model)
4. [Monitoring with MLflow](#monitoring-with-mlflow)
5. [Creating the Video of the Best Model](#creating-the-video-of-the-best-model)

---

## Installation

Follow the steps below to set up your environment and install the necessary dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/JanainaRibas/RL_workshop_0410.git
cd RL_workshop_0410
```

### 2. Create and Activate a Virtual Environment

**For `venv`:**

```bash
python -m venv my_env
source my_env/bin/activate
```

**For `conda`:**

```bash
conda create -n my_env python=3.8.20
conda activate my_env
```

### 3. Install Dependencies

Once the environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

Alternatively, for `conda` users:

```bash
conda install --file requirements.txt
```

---

## Preparing the Training Script

Before you begin training, you may want to modify the hyperparameters in the `run_training_mlflow.py` script.

### Hyperparameters

The following hyperparameters are defined in the script and can be adjusted for different training configurations:

```python
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
```

- You can modify the values in the lists to adjust the combinations of hyperparameters being tested. For example, to test a single value for learning rate, change:

```python
"lr": 1e-3,
```

- The default number of training episodes is set to 5000, but this can also be modified in the script.

---

## Training the Model

Once your environment is set up, you can start training the model using the following command:

```bash
python run_training_mlflow.py
```

This will run the training process with your specified hyperparameters, using MLflow to track the experiments.

---

## Monitoring with MLflow

### 1. Start the MLflow UI

To monitor the progress of your training and view the results, start the MLflow UI by running:

```bash
mlflow ui
```

This will start a local web server, and you can access the MLflow UI by navigating to the provided link (usually `http://127.0.0.1:5000/`) in your browser.

The MLflow UI will allow you to:

- View the hyperparameters used for each run.
- Compare training performance (e.g., cumulative rewards).
- See artifacts, like checkpoint path.

---

## Creating the Video of the Best Model

After training is complete, you can generate a video of the agent interacting with the environment using the best model checkpoint.

### 1. Update the Checkpoint Path

In the `inference_and_video.py` script, change the value of the hyperparameters to the ones related to the best model, and modify the `checkpoint_path` variable to point to the model. You can get this path from the MLflow UI.

### Locate the Artifacts Section
In the experiment run details page, you will find an Artifacts tab. Click on it to expand the artifacts associated with the selected run. Within this section, look for a directory called model/ or similar. This is where the model checkpoint is stored.

Example:

```python
checkpoint_path = r"mlruns/733636240066742753/a6b8c6e263554ba4bd120d5b3585c5c7/artifacts/model/"
```

### 2. Run the Script

Once the path is updated, run the following command to create the video:

```bash
python inference_and_video.py
```

This will generate a video showing the best-performing agent in action within the Lunar Lander environment.

---

## Additional Notes

- If you are unfamiliar with **MLflow**, you can find more information in the [official documentation](https://www.mlflow.org/docs/latest/index.html).
- You can try other environments from the **OpenAI Gym** library by changing the `env` parameter in the hyperparameter grid. You can find a list of available environments [here](https://gymnasium.farama.org/).

---

