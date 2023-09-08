import random

# Print all solutions
def print_solutions(solutions):
    for solution in solutions:
        print(solution)

def fitness(solution):
    n = len(solution)
    conflicts = 0

    # Check for conflicts in columns and diagonals
    for i in range(n):
        for j in range(i + 1, n):
            if solution[i] == solution[j] or abs(solution[i] - solution[j]) == abs(i - j):
                conflicts += 1

    # If there are no conflicts, it's a valid solution
    if conflicts == 0:
        return 1.0
    else:
        return 1.0 / (1.0 + conflicts)  # Higher conflicts lead to lower fitness

# Select individuals for crossover
def selection(population, fitness_function):
    total_fitness = sum(fitness_function(solution) for solution in population)

    # Calculate the probability of each individual being selected
    selection_probabilities = [fitness_function(solution) / total_fitness for solution in population]

    # Perform roulette wheel selection to choose the individuals
    selected_population = []
    for _ in range(len(population)):
        selected_individual = roulette_wheel_select(population, selection_probabilities)
        selected_population.append(selected_individual)

    return selected_population

# Select an individual using roulette wheel selection
def roulette_wheel_select(population, probabilities):
    # Spin the roulette wheel and select an individual
    selected_index = 0
    random_value = random.random()

    while random_value > 0:
        random_value -= probabilities[selected_index]
        selected_index += 1

    selected_index -= 1  # Adjust for overshooting

    return population[selected_index]

# Perform crossover on the selected individuals
def crossover(selected_population, population_size):
    offspring_population = []

    # Perform crossover to create new offspring
    while len(offspring_population) < population_size:
        parent1, parent2 = random.sample(selected_population, 2)  # Select two parents randomly

        # Choose a random crossover point
        crossover_point = random.randint(1, len(parent1) - 1)

        # Create offspring by combining genetic material from parents
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        # Add offspring to the new population
        offspring_population.extend([child1, child2])

    # Ensure the offspring population size matches the desired population size
    if len(offspring_population) > population_size:
        offspring_population = offspring_population[:population_size]

    return offspring_population

# Mutate the population
def mutation(offspring_population, mutation_rate):
    mutated_population = []

    for solution in offspring_population:
        if random.random() < mutation_rate:
            # Apply mutation to this solution
            mutated_solution = mutate_solution(solution)
            mutated_population.append(mutated_solution)
        else:
            mutated_population.append(solution)

    return mutated_population

# Mutate a solution by randomly selecting a queen and moving it to a new row in the same column
def mutate_solution(solution):
    # Randomly select a queen and move it to a new row in the same column
    mutated_solution = solution[:]
    queen_to_mutate = random.randint(0, len(solution) - 1)
    new_row = random.randint(0, len(solution) - 1)
    mutated_solution[queen_to_mutate] = new_row
    return mutated_solution

# Generate an initial population, this function will only be called once at the beginning of the algorithm
def initial_population(population_size, n):
    population = []
    for _ in range(population_size):
        solution = list(range(n))
        random.shuffle(solution)
        population.append(solution)
    return population

# Main loop of the genetic algorithm
def genetic_algorithm(n, population_size, max_generations, mutation_rate):
    population = initial_population(population_size, n)
    best_solution = None
    best_fitness = 0

    for generation in range(max_generations):
        # Selection
        selected_population = selection(population, fitness)

        # Crossover
        offspring_population = crossover(selected_population, population_size)

        # Mutation
        mutated_population = mutation(offspring_population, mutation_rate)

        # Replacement
        population = mutated_population # We can use a function for the replacement step if we want to

        # Track the best solution
        for solution in population:
            current_fitness = fitness(solution)
            if current_fitness > best_fitness:
                best_solution = solution
                best_fitness = current_fitness

        # If we find a solution with fitness 1 we can stop
        if best_fitness == 1.0:
            break

    return best_solution, best_fitness, generation + 1

# n is the size of the board, population_size is the number of solutions in each generation,max_generations is the maximum number of generations, the algorithm will stop if this number is reached, mutation_rate is the probability of a mutation occurring
print(genetic_algorithm(n=32, population_size=100, max_generations=10000, mutation_rate=0.1))