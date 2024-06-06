# Bisimulation Metrics as Optimal Transport Distances

This repository contains the code associated with the paper "Bisimulation Metrics are Optimal Transport Distances, and Can be Computed Efficiently". The code allows for computing bisimulation metrics using optimal transport distances and provides tools to run experiments on random instances.

## Getting Started

### Cloning the Repository

To get started, clone the repository to your local machine using the following command:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

### Installing Requirements
Make sure you have Python installed on your system. It's recommended to use a virtual environment to manage dependencies. You can install the required packages using pip:

python -m venv venv
source venv/bin/activate
pip install -r requirements
pip install -e .

### Testing the Installation
To verify that everything is set up correctly, you can run the tests included in the repository:

python -m pytest tests/*.py

If the tests pass, you're ready to start experimenting with the code.

## Running the Code
Running Experiments on Random Instances
To run the main script and experiment with different parameters, use:

python main.py

You can modify the arguments to run different experiments. For example:

python main.py -dx 4 -dy 3 --eta 0.1

Computing Distances for Given Instances

The script run_experiments.py allows you to compute distances for specific instances. You can provide a pair of istances 

python run_experiments.py -f1 "../benchmarks/deterministic/miconic/p01.json" -f2 "../benchmarks/deterministic/miconic/p02.json"

or a folder containing multiple instances as input::

python run_experiment.py -f ../benchmarks/stochastic/3_rewards_random

The script will return the distance or a matrix of pairwise distances depending on the input provided.

