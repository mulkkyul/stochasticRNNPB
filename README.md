
# Stochastic RNNPB

This repository contains the PyTorch implementation of a **Stochastic RNNPB (Recurrent Neural Network with Parametric Biases)** model. The model leverages stochastic parametric biases to perform tasks such as sequence generation and recognition.

<br><br>

## Features

- **Learning Stochastic Representations**: Learns stochastic representations of multidimensional time-series data using parametric biases (`mu` and `logvar`).

- **Sequence Generation**: Generates multidimensional time-series data in both stochastic and deterministic modes.

- **Sequence Recognition**: Recognizes sequences by updating parametric biases to minimize prediction error.

- **Highly Configurable**: Adjustable parameters such as learning rates, beta (KL-divergence weight), hidden size, and more via a `params.json` configuration file. The model can operate in both deterministic and stochastic modes depending on the configuration.

<br><br>

## Repository Structure

```plaintext
.
├── figures            # High-resolution figures used in the paper
├── dataset            # Scripts for preparing the dataset
├── generation.py      # Code for sequence generation using the trained model
├── recognition.py     # Code for recognizing sequences using learned or pre-searched parametric biases
├── train.py           # Code for training the Stochastic RNNPB model
├── myDataloader.py    # Custom dataset handling and preprocessing
├── models.py          # Definition of the Stochastic RNNPB model architecture
├── utils.py           # Utility functions for parameter management, saving/loading models, etc.
├── params.json        # Configuration file for hyperparameters and dataset paths
└── README.md          # This documentation file
```

<br><br>

## Getting Started

### Environment

The code is tested with:

- Python 3.7
- PyTorch 1.10
- Ubuntu
- Other dependencies: Install via `pip install -r requirements.txt` (create this file if needed).

<br><br>

## Usage

### 0. Configuration

Modify `params.json` to adjust hyperparameters:

```json
{
    "random_seed": 1116,
    "train": {
        "learning_rate": 0.001,
        "num_epochs": 1000,
        "batch_size": 128
    },
    "model": {
        "behavior": "stochastic",
        "beta": 0.000001,
        "pb_size": 4,
        "num_layers": 1,
        "hidden_size": 256,
        "data_dim": 17,
        "num_pb": 72
    },
    "path_dataset_train": "./dataset/train/",
    "path_dataset_recognition": "./dataset/recognition/"
}
```

#### Key Parameters:

- `behavior`: "stochastic" or "deterministic"
- `beta`: Weight for KL-divergence in the loss function
- `pb_size`: Size of the parametric bias vector
- `hidden_size`: Number of hidden units in the LSTM
- `data_dim`: Size of the input vector (e.g., number of joints)
- `num_pb`: Number of sequences in the training dataset

<br><br>

### 1. Training the Model

The `train.py` script is used to train the `StochasticLSTMPB` model. It sets up the model, loads the training data, and optimizes the parameters using the Adam optimizer. The training process focuses on learning Parametric Bias (PB) values (mean and variance) for each sequence in the dataset, combining reconstruction loss and KL divergence (scaled by a beta parameter) to guide the updates. It processes the data in batches, computes the loss, and adjusts the model to improve its performance. At the end of training, the script saves the model and optimizer states for future use.

Train the model using the dataset:

```bash
python train.py
```
The model and its parameters will be saved in the `./result/` directory.

<br><br>

### 2. Generating Sequences

The `generation.py` script is used for generating sequences using the `StochasticLSTMPB` model. It loads the trained model and either samples sequences from learned PB (Parametric Bias) values or generates sequences from given PB values (mu and logvar). The script evaluates the model in inference mode and outputs the generated sequences as data frames, making it easy to visualize or analyze the results. You can adjust the sampling behavior to explore different generation scenarios.

Use the trained model for sequence generation:

```bash
python generation.py
```

<br><br>

### 3. Recognizing Sequences

The `recognition.py` script is used to recognize sequences using the `StochasticLSTMPB` model. It loads a trained model and adjusts the Parametric Bias (PB) values (mean and variance) to minimize the discrepancy between the model's output and observed data. The script supports a "pre-search" step to initialize PB values based on prior knowledge or random initialization. During the recognition process, the model iteratively refines PB values using gradient-based optimization to align the generated output with the target sequence. The recognized and target sequences are saved as data frames for analysis or visualization.

Perform sequence recognition:

```bash
python recognition.py
```

<br><br>

## Citation

If you use this code for your research, please cite:
> Jungsik Hwang and Ahmadreza Ahmadi, "A Novel Framework for Learning Stochastic Representations for Sequence Generation and Recognition," ArXiv. 2024.

<br><br>

## License

This project is licensed under the [MIT License](./LICENSE).

<br><br>

## Acknowledgements

This work builds upon the concepts of [RNNPB](https://ieeexplore.ieee.org/abstract/document/1235981) and [Variational Autoencoders](https://arxiv.org/abs/1312.6114) to introduce a stochastic framework for sequence generation and recognition. The [REBL(Robotic Emotional Body Language)-pepper dataset](https://github.com/minamar/rebl-pepper-data) was used to validate the model.

