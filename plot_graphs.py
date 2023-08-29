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

# Parse results to a list of generations and a dictionary mapping each generation to its results
generations = []
result_dict = {}
for i, item in enumerate(cars):
    generation = item.split("C")[0][1:]
    if generation not in generations:
        generations.append(generation)
        result_dict[generation] = []
    result_dict[generation].append(fitnesses[i])

# Calculate the maximum result and average of top 5 results from each generation
max_results = []
top5_averages = []
for key, results in result_dict.items():
    max_results.append(max(results))
    results.sort(reverse=True)
    top5_results = results[:5]
    top5_average = sum(top5_results) / len(top5_results)
    top5_averages.append(top5_average)


# Plot graph
plt.plot(generations, max_results, label="Maximum")
plt.plot(generations, top5_averages, label="Top 5 average")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.legend()
plt.show()
