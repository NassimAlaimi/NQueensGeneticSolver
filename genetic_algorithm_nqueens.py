# Made by Sandro Sgro, Nassim Alaimi and Aljona Samuelsson.

import random
import time

# Calculate the fitness of a solution, representing the number of clashes.
def fitness(solution):
    n = len(solution)  # Get the size of the board (number of queens).
    clashes = 0  # Initialize the clashes counter.

    # Loop through all pairs of queens on the board.
    for i in range(n):
        for j in range(i + 1, n):
            # Check if there is a clash between queens:
            # 1. If they are on the same diagonal (abs(i - j) == abs(solution[i] - solution[j])).
            # 2. If they are on the same row (solution[i] == solution[j]).
            if abs(i - j) == abs(solution[i] - solution[j]) or solution[i] == solution[j]:
                clashes += 1  # Increment the clash counter.

    return clashes  # Return the total number of clashes for the given solution.

# Generate an initial population of solutions.
def generate_population(pop_size, board_size):
    # Initialize an empty list to store the population.
    population = []

    # Loop to create 'pop_size' number of solutions.
    for _ in range(pop_size):
        # Generate a random permutation of numbers from 1 to 'board_size'.
        # This represents the placement of queens on the chessboard.
        solution = random.sample(range(1, board_size + 1), board_size)
        
        # Append the generated solution to the population list.
        population.append(solution)

    return population  # Return the list of generated solutions (population).

# Select a subset of the population to act as parents based on their fitness.
def select_parents(population, num_parents):
    # Calculate fitness scores for each solution in the population.
    fitness_scores = [fitness_cache.get(tuple(solution), fitness(solution)) for solution in population]

    # Sort the indices of solutions in ascending order of their fitness scores.
    selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:num_parents]

    # Create a list of parent solutions by selecting solutions from the population using the selected indices.
    selected_parents = [population[i] for i in selected_indices]

    return selected_parents  # Return the list of selected parent solutions.


# Perform crossover between two parent solutions to create a child solution.
def crossover(parent1, parent2):
    n = len(parent1)  # Get the size of the parent solutions.
    
    # Choose a random crossover point between 1 and 'n'.
    crossover_point = random.randint(1, n)

    # Initialize an empty list to store the child solution.
    child = [0] * n

    # Copy the first part of the child from 'parent1' up to the crossover point.
    child[:crossover_point] = parent1[:crossover_point]

    # Create a list of available positions from 'parent2' that are not in the child.
    available_positions = [x for x in parent2 if x not in child]

    # Fill in the remaining part of the child with genes from 'parent2'.
    for i in range(n):
        if child[i] == 0:
            if available_positions:
                # If there are available positions from 'parent2', pop one and assign it to the child.
                gene = available_positions.pop()
                child[i] = gene
            else:
                # If all positions from 'parent2' are used, choose a random remaining gene not in the child.
                remaining_genes = [x for x in range(1, n + 1) if x not in child]
                random_gene = random.choice(remaining_genes)
                child[i] = random_gene

    return child  # Return the resulting child solution.

# Mutate a solution with a certain probability.
def mutate(solution, mutation_rate):
    n = len(solution)  # Get the size of the solution (number of queens).

    # Loop through each queen's position in the solution.
    for i in range(n):
        # Check if a random number between 0 and 1 is less than the mutation rate.
        if random.random() < mutation_rate:
            # If the condition is met, generate a random index 'j' within the solution's size.
            j = random.randint(0, n - 1)
            
            # Swap the positions of two queens in the solution to introduce a mutation.
            solution[i], solution[j] = solution[j], solution[i]

# Mutate the first queen in conflict (clashing with another queen).
def mutate_first_conflict_queen(solution):
    n = len(solution)  # Get the size of the solution (number of queens).

    # Iterate through all pairs of queens in the solution.
    for i in range(n):
        for j in range(i + 1, n):
            # Check if there is a clash between queens:
            # 1. If they are on the same diagonal (abs(i - j) == abs(solution[i] - solution[j])).
            # 2. If they are on the same row (solution[i] == solution[j]).
            if abs(i - j) == abs(solution[i] - solution[j]) or solution[i] == solution[j]:
                # Generate a new random position for the queen at index 'i'.
                new_position = random.randint(1, n)

                # Ensure that the new position is different from the current position.
                while new_position == solution[i]:
                    new_position = random.randint(1, n)

                # Update the position of the queen at index 'i' to the new position.
                solution[i] = new_position

                return  # Exit the function after mutating the first conflicting queen.

# Main genetic algorithm function.
def genetic_algorithm(board_size, pop_size, num_generations, mutation_rate, max_same_fitness_iters):
    # Initialize the population with random solutions.
    population = generate_population(pop_size, board_size)

    # Find the initial best solution in the population.
    best_solution = min(population, key=lambda x: fitness_cache.get(tuple(x), fitness(x)))
    best_fitness = fitness_cache.get(tuple(best_solution), fitness(best_solution))

    # Initialize a counter to keep track of consecutive generations with the same best fitness.
    same_fitness_iters = 0

    for generation in range(1, num_generations + 1):
        # Select a subset of the population to act as parents for the next generation.
        parents = select_parents(population, pop_size // 2)
        new_population = parents.copy()

        # Create new child solutions through crossover and mutation.
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        # Find the best solution in the current population.
        current_best = min(population, key=lambda x: fitness_cache.get(tuple(x), fitness(x)))
        current_fitness = fitness_cache.get(tuple(current_best), fitness(current_best))

        # Update the best solution if the current solution has better fitness.
        if current_fitness < best_fitness:
            best_solution = current_best
            best_fitness = current_fitness
            same_fitness_iters = 0 

        # Print the best fitness for the current generation.
        print(f"Generation {generation}: {best_fitness} clashes")

        # Check if the current fitness is the same as the previous generation.
        if current_fitness == best_fitness:
            same_fitness_iters += 1
            if same_fitness_iters >= max_same_fitness_iters:
                # If fitness hasn't improved for a certain number of generations,
                # mutate the first conflicting queen to introduce diversity.
                mutate_first_conflict_queen(best_solution)
                best_fitness = fitness_cache.get(tuple(best_solution), fitness(best_solution))
                same_fitness_iters = 0

        # If a perfect solution is found (fitness equals 0), exit the loop early.
        if best_fitness == 0:
            break

    return best_solution  # Return the best solution found by the genetic algorithm.

# Function to check if a string represents a positive integer.
def is_positive_integer(input_str):
    return input_str.isdigit() and int(input_str) > 0

# Function to check if a string represents a positive decimal number between 0 and 1.
def is_valid_mutation_rate(input_str):
    try:
        rate = float(input_str)
        return 0 <= rate <= 1
    except ValueError:
        return False

# Ask the user to enter board_size until a valid input is provided.
while True:
    board_size = input("Enter board size (a positive integer greater or equal to 4): ")
    if is_positive_integer(board_size) and int(board_size) >= 4:
        board_size = int(board_size)
        break
    else:
        print("Invalid input. Please enter a positive integer for board size.")

# Ask the user to enter pop_size until a valid input is provided.
while True:
    pop_size = input("Enter population size (a positive integer): ")
    if is_positive_integer(pop_size):
        pop_size = int(pop_size)
        break
    else:
        print("Invalid input. Please enter a positive integer for population size.")

# Ask the user to enter num_generations until a valid input is provided.
while True:
    num_generations = input("Enter number of max generations (a positive integer): ")
    if is_positive_integer(num_generations):
        num_generations = int(num_generations)
        break
    else:
        print("Invalid input. Please enter a positive integer for the number of generations.")

# Ask the user to enter mutation_rate until a valid input is provided.
while True:
    mutation_rate = input("Enter mutation rate (a positive float between 0 and 1): ")
    if is_valid_mutation_rate(mutation_rate):
        mutation_rate = float(mutation_rate)
        break
    else:
        print("Invalid input. Mutation rate must be a float between 0 and 1.")

# Ask the user to enter max_same_fitness_iters until a valid input is provided.
while True:
    max_same_fitness_iters = input("Enter max number of iterations with same fitness (a positive integer): ")
    if is_positive_integer(max_same_fitness_iters):
        max_same_fitness_iters = int(max_same_fitness_iters)
        break
    else:
        print("Invalid input. Please enter a positive integer for max same fitness iterations.")

# Cache for fitness calculations to avoid redundant computations.
fitness_cache = {}

# Measure the start time of the genetic algorithm.
start = time.time()

# Run the genetic algorithm to find the best solution.
solution = genetic_algorithm(board_size, pop_size, num_generations, mutation_rate, max_same_fitness_iters)

# Measure the end time of the genetic algorithm.
end = time.time()

# Display the best solution found, its fitness (number of clashes), and the time taken to find the solution.
print(f"Solution: {solution} with {fitness(solution)} clashes in {end - start} seconds")
print("Made by Sandro Sgro, Nassim Alaimi and Aljona Samuelsson.")