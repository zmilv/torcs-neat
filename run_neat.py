import atexit
import pickle
from itertools import count

import neat
import numpy as np


NO_OF_GENERATIONS = 50
MAX_STEPS = 1000


generation = -1
car_list = []
fitness_list = []
laptime_list = []


def train_ai(config):
    # Create starting population for the algorithm
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint("checkpoints/checkpoint-49")  # Uncomment to resume from a checkpoint

    # Add reporter for statistical results
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    # Add saving of checkpoints
    p.add_reporter(neat.Checkpointer(5, filename_prefix="checkpoints/checkpoint-"))

    # Save results at program exit
    atexit.register(save_results)

    # Run NEAT
    winner = p.run(run_generation, NO_OF_GENERATIONS)

    # Save the best genome
    print("Saving best genome")
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def run_generation(genomes, config):
    from gym_torcs import TorcsEnv

    # Init NEAT
    nets = []
    ge = []

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)

    # Init TORCS
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    # Main loop
    global generation
    generation += 1

    for i in range(len(nets)):
        print(f"Generation {generation} Car {i}")

        laptime = 0.0
        action = [0, 0, 0, 0]
        for x in range(MAX_STEPS):
            obs, reward, meta, _, laptime_obs = env.step(action)
            if meta:
                break
            action = control_agent(obs, action, nets, i)

            # Update car and fitness
            genomes[i][1].fitness += reward

            if laptime == 0.0 and laptime_obs > 0.0:  # Register standing start lap time
                laptime = laptime_obs

        # Append results list for saving later
        car_list.append(f"G{generation}C{i}")
        car_fitness = genomes[i][1].fitness
        fitness_list.append(int(car_fitness))
        laptime_list.append(laptime)

        print(f"Total fitness: {car_fitness}")

        if np.mod(i, 3) == 0:
            env.reset(
                relaunch=True
            )  # relaunch TORCS every 3 cars because of the memory leak
        else:
            env.reset()

    env.end()  # Shut down TORCS
    print("Finished generation")


def calculate_fitness(obs, obs_prev):
    """Genome fitness calculation. Called by step() in gym_torcs.py"""
    speed = obs["speedX"]
    v_long = speed * np.cos(obs["angle"])
    track_pos = np.abs(obs["trackPos"])

    off_track_coef = 5
    collision_coef = 10

    # Penalty calculation
    penalty = 0
    # Out-of-track detection
    if track_pos > 1:
        if track_pos > 2:
            track_pos = 2
        penalty += off_track_coef * (track_pos - 1)
    # Collision detection
    if obs["damage"] - obs_prev["damage"] > 0:
        print("Damage!")
        penalty += collision_coef

    # Fitness function
    fitness = (1 - penalty) * v_long
    return fitness


def control_agent(obs, action, nets, i):
    # Get sensor readings from observation
    speedX = obs.speedX
    speedY = obs.speedY
    speedZ = obs.speedZ
    angle = obs.angle
    track = obs.track  # 0, 3, 6, 9, 12, 15, 18
    track0 = track[0]
    track1 = track[3]
    track2 = track[6]
    track3 = track[9]
    track4 = track[12]
    track5 = track[15]
    track6 = track[18]
    trackPos = obs.trackPos

    # Input sensor data and get result from network
    output = nets[i].activate(
        (
            speedX,
            speedY,
            speedZ,
            angle,
            track0,
            track1,
            track2,
            track3,
            track4,
            track5,
            track6,
            trackPos,
        )
    )

    # Convert network output to TORCS actions
    steering_output = output[0]
    action[0] = steering_output
    throttle_brake_output = output[1]
    if throttle_brake_output > 0:
        action[1] = throttle_brake_output
        action[2] = 0
    elif throttle_brake_output < 0:
        action[2] = -throttle_brake_output
        action[1] = 0
    else:
        action[1] = 0
        action[2] = 0
    return action


def save_results():
    print("Saving results")
    with open("results.txt", "w") as file:
        file.write("Cars:\n")
        file.write(str(car_list))
        file.write("\n\n")
        file.write("Fitnesses:\n")
        file.write(str(fitness_list))
        file.write("\n\n")
        file.write("Laptimes:\n")
        file.write(str(laptime_list))


def test_ai(config):
    from gym_torcs import TorcsEnv

    with open("genome.pickle", "rb") as f:
        winner = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    u = [0, 0, 0, 0]

    for i in count(0):
        while True:
            obs, reward, meta, _, lap_time = env.step(u)
            if meta:
                break
            u = control_agent(obs, u, [net], 0)
        if np.mod(i, 3) == 0:
            env.reset(
                relaunch=True
            )  # relaunch TORCS every 3 cars because of the memory leak
        else:
            env.reset()


if __name__ == "__main__":
    # Set configuration file
    config_path = "./config-neat.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    train_ai(config)
    # test_ai(config)
