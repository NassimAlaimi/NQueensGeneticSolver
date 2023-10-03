import random
import time

def fitness(solution):
    n = len(solution)
    clashes = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == abs(solution[i] - solution[j]) or solution[i] == solution[j]:
                clashes += 1
    return clashes

def generate_population(pop_size, board_size):
    return [random.sample(range(1, board_size + 1), board_size) for _ in range(pop_size)]

def select_parents(population, num_parents):
    # Calculer les fitness pour chaque solution dans la population
    fitness_scores = [fitness_cache.get(tuple(solution), fitness(solution)) for solution in population]
    
    # Sélectionner les indices des parents en fonction des fitness (plus faible est mieux)
    selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:num_parents]
    
    # Retourner les solutions des parents sélectionnés
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
    n = len(parent1)
    crossover_point = random.randint(1, n)
    child = [0] * n
    child[:crossover_point] = parent1[:crossover_point]

    # Créer une liste des positions disponibles pour les gènes de l'enfant
    available_positions = [x for x in parent2 if x not in child]
    
    for i in range(n):
        if child[i] == 0:
            if available_positions:
                # Si des positions sont disponibles, sélectionner un gène parmi elles
                gene = available_positions.pop()
                child[i] = gene
            else:
                # Si la liste est vide, prendre un gène au hasard parmi les gènes non utilisés
                remaining_genes = [x for x in range(1, n + 1) if x not in child]
                random_gene = random.choice(remaining_genes)
                child[i] = random_gene

    return child

def mutate(solution, mutation_rate):
    n = len(solution)
    for i in range(n):
        if random.random() < mutation_rate:
            j = random.randint(0, n - 1)
            solution[i], solution[j] = solution[j], solution[i]

def genetic_algorithm(board_size, pop_size, num_generations, mutation_rate):
    population = generate_population(pop_size, board_size)
    best_solution = min(population, key=lambda x: fitness_cache.get(tuple(x), fitness(x)))
    best_fitness = fitness_cache.get(tuple(best_solution), fitness(best_solution))

    for generation in range(1, num_generations + 1):
        parents = select_parents(population, pop_size // 2)
        new_population = parents.copy()

        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda x: fitness_cache.get(tuple(x), fitness(x)))
        current_fitness = fitness_cache.get(tuple(current_best), fitness(current_best))

        if current_fitness < best_fitness:
            best_solution = current_best
            best_fitness = current_fitness

        print(f"Generation {generation}: {best_fitness} clashes")

        if best_fitness == 0:
            print("Solution trouvée avec zéro conflit ! : ", best_solution)
            break

    return best_solution

board_size = int(input("Taille du tableau: "))
pop_size = int(input("Taille de la population: "))
num_generations = int(input("Nombre de générations: "))
mutation_rate = float(input("Taux de mutation: "))

fitness_cache = {}

start = time.time()
solution = genetic_algorithm(board_size, pop_size, num_generations, mutation_rate)
end = time.time()
print(f"Solution: {solution} with {fitness(solution)} clashes in {end - start} seconds")