import numpy as np
from GA import myGA
import pickle
import sklearn
import scipy
from sklearn import preprocessing


"""
The y=target is to minimize the local fan rotation speed:
    local fan rotation speed  = MachineLearning(position_of_load)
    where T_minimum < T_cpu < T_maximum 
    What are the best values for the position_of_load?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

num_weights=6
sol_per_pop=8

pop_size=(sol_per_pop, num_weights)

#Creating the initial population

Total_CPU_request=60
dist_rate=np.random.uniform(low=0, high=1, size=pop_size)
Load_jobs=Total_CPU_request ** dist_rate
Load_jobs=Load_jobs.astype(np.int64)
Input=Load_jobs[0]
new_population=[]

for i in range(0, sol_per_pop):
    new_population.append(np.random.permutation(Input))

Exhaust_airtemp=np.array([36.2, 34.9, 35.1]) #ここの部分はどのように与えるか....
#new_one=np.zeros(9, 6)
new_one=[]

for i in range(sol_per_pop):
    tmp=np.append(new_population[i], Exhaust_airtemp)
    new_one.append(tmp)

new_population=np.array(new_one)
print("New population:", new_population)

###########################################################

predict_fan=myGA.predict_fan_rotation_speed(new_population)
print("Predicted fans", predict_fan)
mean_each_KID=myGA.cal_pop_fitness(predict_fan)
print("AVG of KIDs: ", mean_each_KID)


num_generations=5
num_parents_mating=4
new_num_weights=9

parents=myGA.select_mating_pool(new_population, mean_each_KID, num_parents_mating)
print("parents: ", parents)
off_spring_size = (pop_size[0] - parents.shape[0], new_num_weights)

offspring=myGA.crossover(parents, off_spring_size)
print("offspring", offspring)

############################################################


'''
num_generations=5
num_parents_mating=4
new_num_weights=9

for generation in range(num_generations):
    print("Generation: ", generation)
    # Predictions using regression (Need to fix in the section of preprocessing so that getting the correct samples)
    predict_fan=myGA.predict_fan_rotation_speed(new_population)
    #Measuring the fitness of each chromosome in the population
    fitness=myGA.cal_pop_fitness(equation_inputs, new_population)
    parents=myGA.select_mating_pool(new_population, fitness, num_parents_mating)

    #Generating next generation using crossover
    off_spring_size = (pop_size[0] - parents.shape[0], num_weights)
    offspring_crossover=myGA.crossover(parents, off_spring_size)
    #Adding some variations to the offspring using mutation.
    offspring_mutation=myGA.mutation(offspring_crossover)

    #Creating the new population based on the parents and offspring
    new_population[0:parents.shape[0], :]=parents
    new_population[parents.shape[0]:, :]=offspring_mutation

    # The best result in the current iteration.
    print("Best result : ", np.max(np.sum(new_population*equation_inputs, axis=1)))


# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = myGA.cal_pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])

'''