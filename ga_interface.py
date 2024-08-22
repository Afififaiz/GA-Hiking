import streamlit as st
import pickle
import matplotlib.pyplot as plt
import random
import os

# Define the path to the GA model
model_path = os.path.expanduser('~/Documents/ga_knapsack_model.pkl')

# Load the GA Model
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

# Ensure 'best_fitness' is a list
if isinstance(model_data.get('best_fitness'), int):
    model_data['best_fitness'] = [model_data['best_fitness']]  # Convert single value to list

# Genetic Algorithm Functions
def initialize_population(num_items, population_size):
    return [[random.choice([0, 1]) for _ in range(num_items)] for _ in range(population_size)]

def calculate_fitness(individual, items, bag_capacity):
    total_value = sum(item["value"] * item_selected for item, item_selected in zip(items, individual))
    total_weight = sum(item["weight"] * item_selected for item, item_selected in zip(items, individual))
    if total_weight > bag_capacity:
        return 0
    else:
        return total_value

def rank_based_selection(fitness_scores, population_size):
    rank_scores = [score for score, _ in sorted(enumerate(fitness_scores), key=lambda x: x[1], reverse=True)]
    total_rank = sum(range(1, population_size + 1))
    probabilities = [rank / total_rank for rank in range(1, population_size + 1)]
    selected_parents = random.choices(range(population_size), weights=probabilities, k=population_size // 2)
    return selected_parents, population_size // 2

def enforce_weight_constraint(individual, items, bag_capacity):
    total_weight = sum(item["weight"] * item_selected for item, item_selected in zip(items, individual))
    while total_weight > bag_capacity:
        mutation_point = random.randint(0, len(items) - 1)
        if individual[mutation_point] == 1:
            individual[mutation_point] = 0
            total_weight -= items[mutation_point]["weight"]
    return individual

def one_point_crossover(parent1, parent2, num_items):
    crossover_point = random.randint(1, num_items - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2

def bit_flip_mutation(individual, items, bag_capacity):
    mutation_point = random.randint(0, len(items) - 1)
    individual[mutation_point] = 1 - individual[mutation_point]
    return enforce_weight_constraint(individual, items, bag_capacity)

def elitism_replacement(population, offspring, items, bag_capacity):
    combined_population = population + offspring
    fitness = [calculate_fitness(individual, items, bag_capacity) for individual in combined_population]
    best_indices = sorted(range(len(fitness)), key=lambda k: fitness[k], reverse=True)[:len(population)]
    return [combined_population[i] for i in best_indices]

# Display the app title
st.title("Interactive Knapsack Problem Solver")

# Sidebar for user inputs
bag_capacity = st.sidebar.slider("Bag Capacity", min_value=10, max_value=50, value=model_data["bag_capacity"])
population_size = st.sidebar.slider("Population Size", min_value=10, max_value=100, value=model_data["population_size"])
num_generations = st.sidebar.slider("Number of Generations", min_value=10, max_value=100, value=model_data["num_generations"])
crossover_rate = st.sidebar.slider("Crossover Rate", min_value=0.0, max_value=1.0, value=model_data["crossover_rate"])
mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=model_data["mutation_rate"])

# Function to run the Genetic Algorithm with custom parameters
def run_genetic_algorithm():
    num_items = len(model_data["items"])
    population = initialize_population(num_items, population_size)
    best_fitness_list = []
    best_individual_list = []

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, num_generations)
    ax.set_ylim(0, max(model_data['best_fitness']) * 1.1 if model_data['best_fitness'] else 1)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness Value')
    ax.set_title('Fitness Evolution Over Generations')

    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(individual, model_data["items"], bag_capacity) for individual in population]
        parents, _ = rank_based_selection(fitness_scores, population_size)

        offspring = []
        for parent1, parent2 in zip(parents[::2], parents[1::2]):
            if random.random() < crossover_rate:
                child1, child2 = one_point_crossover(population[parent1], population[parent2], num_items)
                offspring.extend([child1, child2])
            else:
                offspring.extend([population[parent1], population[parent2]])

        offspring = [bit_flip_mutation(individual, model_data["items"], bag_capacity) if random.random() < mutation_rate else individual for individual in offspring]
        population = elitism_replacement(population, offspring, model_data["items"], bag_capacity)
        fitness_scores = [calculate_fitness(individual, model_data["items"], bag_capacity) for individual in population]

        best_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(best_fitness)]

        best_fitness_list.append(best_fitness)
        best_individual_list.append(best_individual)

        # Update the plot with new data for the animated graph
        line.set_data(range(1, len(best_fitness_list) + 1), best_fitness_list)
        ax.set_xlim(0, max(len(best_fitness_list), 10))
        ax.set_ylim(0, max(best_fitness_list) * 1.1)
        st.pyplot(fig)
        plt.pause(0.1)

    return best_individual_list[-1], best_fitness_list[-1], best_fitness_list

# Run the Genetic Algorithm when the button is clicked
if st.button("Run Optimization"):
    best_individual, best_fitness, fitness_history = run_genetic_algorithm()

    # Display the final fitness evolution as a static graph for all generations
    fig, ax = plt.subplots()
    ax.plot(range(1, len(fitness_history) + 1), fitness_history, marker='o')
    ax.set_xlim(0, len(fitness_history))
    ax.set_ylim(0, max(fitness_history) * 1.1)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness Value')
    ax.set_title('Final Fitness Evolution Over All Generations')
    st.pyplot(fig)

    # Display the best solution
    total_value = sum(item["value"] for item, selected in zip(model_data["items"], best_individual) if selected)
    total_weight = sum(item["weight"] for item, selected in zip(model_data["items"], best_individual) if selected)

    st.subheader("Best Solution")
    st.write(f"Total Value: {total_value}")
    st.write(f"Total Weight: {total_weight}")

    st.write("Items included in the best solution:")
    for item, selected in zip(model_data["items"], best_individual):
        if selected:
            st.write(f"{item['name']} (Weight: {item['weight']}, Value: {item['value']})")
