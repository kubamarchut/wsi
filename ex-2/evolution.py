import numpy as np
from copy import deepcopy


def random_vector(min_x, max_x, n):
    return np.random.uniform(min_x, max_x, size=n)


class Individual:
    def __init__(self, x):
        self.x = x
        self.fitness_score = 0

    def mutate(self, sigma, p, generation, max_generations):
        for i in range(len(self.x)):
            if np.random.rand() < p:
                progress = generation / max_generations
                adaptive_sigma = sigma * (1 - progress)
                self.x[i] = self.x[i] + np.random.normal(0, adaptive_sigma, 1)

    def avg_crossover(self, partner, p):
        if np.random.rand() < p:
            new_x1 = np.mean([self.x, partner.x], axis=0)
            new_x2 = np.mean([self.x, partner.x], axis=0)

            self.x = new_x1
            partner.x = new_x2

    def one_point_crossover(self, partner, p):
        if np.random.rand() < p:
            point = np.random.randint(1, len(self.x))

            new_x1 = np.concatenate((self.x[:point], partner.x[point:]))
            new_x2 = np.concatenate((partner.x[:point], self.x[point:]))

            self.x = new_x1
            partner.x = new_x2

    def __str__(self) -> str:
        return f"{self.x} -> {self.fitness_score:.3E}"


class Population:
    def __init__(self, size, chrom_cnt, min_x, max_x):
        self.size = size
        self.min_x = min_x
        self.max_x = max_x
        self.chrom_cnt = chrom_cnt
        self.individuals = []

    def initialize(self, random=True):
        for _ in range(self.size):
            if random:
                new_chrom = random_vector(self.min_x, self.max_x, self.chrom_cnt)
            else:
                new_chrom = np.ones(self.chrom_cnt) * self.min_x

            new_individual = Individual(new_chrom)
            self.individuals.append(new_individual)

    def fitness(self, objective):
        for individual in self.individuals:
            individual.fitness_score = objective(individual.x)

    def mutate(self, sigma, p, generation, max_generations):
        for individual in self.individuals:
            individual.mutate(sigma, p, generation, max_generations)

    def tournament(self, pool, tournament_size):
        specimens = np.random.choice(pool, tournament_size)
        specimens = sorted(specimens, key=lambda x: x.fitness_score)
        return specimens[0]

    def selection(self, k=2, elitism_size=1):
        self.individuals.sort(key=lambda x: x.fitness_score, reverse=True)
        new_pop = deepcopy(self.individuals[:elitism_size])

        new_pop = [
            deepcopy(self.tournament(self.individuals, k))
            for i in range(len(self.individuals) - elitism_size)
        ]
        self.population = new_pop

    def perform_crossover(self, p):
        for individual in self.individuals:
            partner = np.random.choice(self.individuals)

            individual.avg_crossover(partner, p)

    def __str__(self) -> str:
        output = ""
        for index, individual in enumerate(self.individuals):
            output += f"{index+1}. ind {individual}\n"

        return output


if __name__ == "__main__":
    np.set_printoptions(precision=2)

    population = Population(5, 10, 0, 1)
    population.initialize()
    print(population)
