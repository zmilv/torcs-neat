import neat
import pickle
from dependencies import visualize


config_path = "./config-neat.txt"
config = neat.config.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

with open("genome.pickle", "rb") as file:
    genome = pickle.load(file)

node_names = {
    -1: "speedX",
    -2: "speedY",
    -3: "speedZ",
    -4: "angle",
    -5: "track0",
    -6: "track1",
    -7: "track2",
    -8: "track3",
    -9: "track4",
    -10: "track5",
    -11: "track6",
    -12: "trackPos",
    0: "Steering",
    1: "Throttle/Brake",
}

visualize.draw_net(config, genome, True, node_names=node_names, show_disabled=False)
