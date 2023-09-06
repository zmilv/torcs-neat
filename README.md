## Autonomous driving in TORCS using NEAT

Implementation of the NeuroEvolution of Augmenting Topologies (NEAT) algorithm for autonomously driving a car in The Open Racing Car Simulator (TORCS) using the simulated car's sensor readings.

NEAT evolves neural network architectures and their parameters to optimize the agent's driving performance. Starting with a population of diverse neural networks, the algorithm iteratively breeds and mutates them, preserving topological innovations. Through natural selection and speciation, continuously improving neural structures are developed, ultimately enabling the agent to learn effective car control strategies. 
 

### Environment requirements
* Ubuntu OS
* Python 3
* ```pip install -r requirements.txt```
* Installation of Visual TORCS as listed in the Gym-TORCS project
  (https://github.com/ugo-nama-kun/gym_torcs/tree/master/vtorcs-RL-color)

### Usage
* For training, set the **train** flag to 1 in the main() function of run_neat.py
* For simply running (testing) a saved genome, set the **train** flag to 0
* ```python run_neat.py```
