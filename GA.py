import numpy as np
import pickle

class myGA:

    def __init__(self):
        print('コンストラクタが呼ばれました')

    def __del__(self):
        print('デストラクタが呼ばれました')

    def predict_fan_rotation_speed(new_population):
        # Predictions using regression (Need to fix in the section of preprocessing so that getting the correct samples)
        KIDnames = ['KID1', 'KID3', 'KID5', 'KID7', 'KID9', 'KID11']
        Predicted_fans = []
        for j in range(len(new_population)):
            tmp=[]
            for i in range(len(KIDnames)):

                filename = 'sc_model.sav'
                loaded_model = pickle.load(open(filename, 'rb'))
                X = loaded_model.transform([new_population[j]])

                #print("Predicted Local fan rotation speed of " + KIDnames[i], Y_pred)
                filename = 'svr_rbf_model_' + KIDnames[i] + '.sav'
                loaded_model = pickle.load(open(filename, 'rb'))
                Y_pred = loaded_model.predict(X)

                tmp.append(Y_pred)
            Predicted_fans.append(tmp)
        Predicted_fans = np.array(Predicted_fans)
        return Predicted_fans

    def cal_pop_fitness(Predicted_fans):
        # Calculating the fitness value of each solution in the current population.
        mean_each_KID = np.mean(Predicted_fans, axis=1)
        return mean_each_KID

    def select_mating_pool(pop, fitness, num_parents):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.

        parents = np.empty((num_parents, pop.shape[1]))

        for parent_num in range(num_parents):
            min_fitness_idx = np.where(fitness == np.min(fitness))
            min_fitness_idx = min_fitness_idx[0][0]
            parents[parent_num, :] = pop[min_fitness_idx, :]
            fitness[min_fitness_idx] = 99999999999

        return parents

    def crossover(parents, offspring_size):
        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.uint8(offspring_size[1] / 2)
        print("crossover_point: ", crossover_point)
        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k + 1) % parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(offspring_crossover):

        # Mutation changes a single gene in each offspring randomly.

        for idx in range(offspring_crossover.shape[0]):
            # The random value to be added to the gene.

            random_value = np.random.uniform(-1.0, 1.0, 1)

            offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value

        return offspring_crossover
