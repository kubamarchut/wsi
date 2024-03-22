import numpy as np


def random_vector(min_x, max_x, n):
    return np.random.uniform(min_x, max_x, size=n)


class Individual:
    def __init__(self, x):
        self.x = x
        self.fitness_score = 0

    def __str__(self) -> str:
        return f"{self.x} -> {self.fitness_score}"


class Population:
    def __init__(self, size, chrom_cnt, min_x, max_x):
        self.size = size
        self.min_x = min_x
        self.max_x = max_x
        self.chrom_cnt = chrom_cnt
        self.individuals = []

    def initialize(self):
        for _ in range(self.size):
            new_chrom = random_vector(self.min_x, self.max_x, self.chrom_cnt)
            new_individual = Individual(new_chrom)
            self.individuals.append(new_individual)

    def fitness(self, objective):
        for individual in self.individuals:
            individual.fitness_score = objective(individual.x)

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
