import ast
from matplotlib import pyplot as plt


# Extract results from the text file
with open("results.txt", "r") as file:
    lines = file.readlines()
for i, line in enumerate(lines):
    line = line.strip()
    if line == "Cars:":
        cars = ast.literal_eval(lines[i + 1].strip())
    elif line == "Fitnesses:":
        fitnesses = ast.literal_eval(lines[i + 1].strip())
    elif line == "Laptimes:":
        laptimes = ast.literal_eval(lines[i + 1].strip())


# Parse results to a list of generations and a dictionaries mapping each generation to its fitness results and lap times
generations = []
result_dict = {}
laptimes_dict = {}
for i, item in enumerate(cars):
    generation = int(item.split("C")[0][1:])
    if generation not in generations:
        generations.append(generation)
        result_dict[generation] = []
        laptimes_dict[generation] = []
    result_dict[generation].append(fitnesses[i])
    laptimes_dict[generation].append(laptimes[i])

# Calculate the maximum and average results from each generation
max_results = []
top5_averages = []
averages = []
for key, results in result_dict.items():
    max_results.append(max(results))
    results.sort(reverse=True)
    top5_results = results[:5]
    top5_average = sum(top5_results) / len(top5_results)
    top5_averages.append(top5_average)
    average = sum(results) / len(results)
    averages.append(average)

# Get data needed for the lap time plot
laptime_generations = generations
best_laptimes = []
for key, results in laptimes_dict.items():
    generation_laptimes = [x for x in results if x != 0.0]
    if not generation_laptimes:
        laptime_generations = [x for x in laptime_generations if x != key]
    else:
        best_laptimes.append(min(generation_laptimes))


# Plot graphs
fig, ax1 = plt.subplots()

ax1.plot(
    generations, max_results, label="Maximum Fitness in generation", color="dodgerblue"
)
ax1.plot(generations, averages, label="Average Fitness in generation", color="orange")
ax1.plot(
    generations,
    top5_averages,
    label="Average Fitness from top 5 genomes in generation",
    color="orchid",
)
ax1.set_xlabel("Generations")
ax1.set_ylabel("Fitness")

ax2 = ax1.twinx()
ax2.scatter(
    laptime_generations,
    best_laptimes,
    label="Best Lap time in generation",
    color="mediumseagreen",
    marker=".",
)
ax2.set_ylabel("Seconds")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2)

plt.title("NEAT Results")
plt.show()
