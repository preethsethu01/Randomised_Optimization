import mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
import timeit

def randomHill(problem,init_state,max_attempts,iterations):
    rh_best_state,rh_best_fitness = mlrose.random_hill_climb(problem,max_attempts = max_attempts, max_iters = iterations,restarts=0,init_state = init_state,random_state = 1)
    return rh_best_fitness

def simulatedAnnealing(problem,init_state,max_attempts,iterations):
    #Define decay schedule
    schedule = mlrose.ExpDecay()
    #Solve problem using Simulated Annealing
    sm_best_state,sm_best_fitness = mlrose.simulated_annealing(problem,schedule=schedule,max_attempts=max_attempts,max_iters=iterations,curve=False,random_state=1)
    return sm_best_fitness

def genetic(problem,init_state,max_attempts,iterations):
    genA_best_state,genA_best_fitness = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=max_attempts, max_iters=iterations, curve=False,random_state=1)
    return genA_best_fitness

def mimic(problem,init_state,max_attempts,iterations):
    mimic_best_state,mimic_best_fitness = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=max_attempts, max_iters=iterations, curve=False, random_state=1)
    return mimic_best_fitness

def runalgos(problem,max_attempts,iterations):
    #Define decay schedule
    schedule = mlrose.ExpDecay()
    #Solve problem using Random Hill Climbing
    rh_best_state,rh_best_fitness = mlrose.random_hill_climb(problem,max_attempts = max_attempts, max_iters = iterations,restarts=1,init_state = init_state,random_state = 1)
    #Solve problem using Simulated Annealing
    sm_best_state,sm_best_fitness = mlrose.simulated_annealing(problem,schedule=schedule,max_attempts=max_attempts,max_iters=iterations,curve=False,random_state=1)
    #Solve problem using Gentic Algorithm
    genA_best_state,genA_best_fitness = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=max_attempts, max_iters=iterations, curve=False,random_state=1)
    #Solve problem using MIMIC
    mimic_best_state,mimic_best_fitness = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=max_attempts, max_iters=iterations, curve=False, random_state=1)
    #return rh_best_state,rh_best_fitness,sm_best_state,sm_best_fitness,genA_best_state,genA_best_fitness,mimic_best_state,mimic_best_fitness
    return rh_best_fitness,sm_best_fitness,genA_best_fitness,mimic_best_fitness


def plot_iterations(X,Y):
    plt.figure()
    plt.title("Iterations vs Best Fit")
    plt.ylim(5)
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    labels = ['RandomHill','SimulatedAnnealing','Genetic','MIMIC']
    toplot = []
    for i,x in zip(Y,X):
        for j,label in zip(i,labels):
            toplot.append([x,j,label])
    for k in toplot:
        plt.plot(k[0],k[1],label=k[2])
    print("To Plot")
    print(toplot)
    plt.legend(loc="best")
    plt.show()

def increaseComplex():
    pass
def KColorMax(state,edges,colors):
    print("In KColorMax Function")
    fitness = mlrose.MaxKColor(edges)
    problem = mlrose.DiscreteOpt(length=state.shape[0],fitness_fn=fitness,maximize = False,max_val = colors)
    rh_iterations = [10,25,50,75,100,125,200,250,300]
    max_attempts =10
    rh_fitness = []
    sm_fitness = []
    ga_fitness = []
    mi_fitness = []
    time_iterations = []
    #print("Random Hill Iterations")
    #print(rh_iterations)
    for itera in rh_iterations:
        color_rh_st = timeit.timeit()
        rh_best_fitness = randomHill(problem,state,max_attempts,itera)
        color_rh_et = timeit.timeit()
        time_iterations.append(color_rh_et - color_rh_st)
        color_sm_st = timeit.timeit()        
        sm_best_fitness = simulatedAnnealing(problem,state,max_attempts,itera)
        color_sm_et = timeit.timeit()
        time_iterations.append(color_sm_et - color_sm_st)
        color_ga_st = timeit.timeit()
        ga_best_fitness = genetic(problem,state,max_attempts,itera)
        color_ga_et = timeit.timeit()
        time_iterations.append(color_ga_et - color_ga_st)
        color_mi_st = timeit.timeit()
        mi_best_fitness = mimic(problem,state,max_attempts,itera)
        color_mi_et = timeit.timeit()
        time_iterations.append(color_mi_et - color_mi_st)
        rh_fitness.append(rh_best_fitness)
        sm_fitness.append(sm_best_fitness)
        ga_fitness.append(ga_best_fitness)
        mi_fitness.append(mi_best_fitness)
        #print("rh_best_fitness inside while loop")
        #print(rh_best_fitness)
    return rh_iterations,rh_fitness,sm_fitness,ga_fitness,mi_fitness,time_iterations
        
    

#KColor Initialize
#KColorMax(init_state,edges,3)
#seed = 5
#print("KColorMax Problem")
#for nodes in range(5,10,5):
#    colors_range = random.randint(2,int(nodes/2)+1)
#    states = list(range(0,nodes))
#    random_state = np.random.randint(colors_range,size=nodes)
#    random_edges = list(itertools.combinations(states,2))
   #print(random_edges)
#    for i in range(0,len(random_edges)-nodes):
#        random_edges.pop(random.randint(0,len(random_edges)-int(nodes/2)))
#    input_size = []
#    iterations = []
#    rh_best_fit = []
#    sm_best_fit = []
#    ga_best_fit = []
#    mi_best_fit = []
#    color_time = []
#    input_size.append(nodes)
#    n_iterations,rh_value,sm_value,ga_value,mi_value,time_value = KColorMax(random_state,random_edges,colors_range)
#    iterations.append(n_iterations)
#    rh_best_fit.append(rh_value)
#    sm_best_fit.append(sm_value)
#    ga_best_fit.append(ga_value)
#    mi_best_fit.append(mi_value)
#    color_time.append(time_value)
#    print("Results**********************************")
#    print(input_size)
#    print(iterations)
#    print(rh_best_fit)
#    print(sm_best_fit)
#    print(ga_best_fit)
#    print(mi_best_fit)
#    print(color_time)

#Traveling Salesman
#print("Travelling Salesman")
#for coords in range(5,10,5):
#    dest = list(range(0,coords))
#    coords_list = list(itertools.islice(itertools.combinations(dest,2),coords))
#    print(coords_list)
#    print(len(coords_list))

#   dest_init_state = np.arange(0,len(coords_list))
#    fitness_coords = mlrose.TravellingSales(coords = coords_list)
#    problem_fit = mlrose.TSPOpt(length = len(coords_list), fitness_fn = fitness_coords, maximize=False)
#   tsp_iterations = [10,25,50,75,100,125,200,250,300]
#    tsp_rhfit = []
#    tsp_smfit = []
#    tsp_gafit = []
#    tsp_mifit = []
#    no_of_cities = []
#    no_of_cities.append(coords)
#    for tsp_itera in tsp_iterations:
#        print("Random Hill")
#        tsp_rh_st = timeit.timeit()
#        tsprh_bestfit = randomHill(problem_fit,dest_init_state,10,tsp_itera)
#        tsp_rh_et = timeit.timeit()
#        print(tsp_rh_et - tsp_rh_st)
#        print("Simulated Annealing")
#        tsp_sm_st = timeit.timeit()
#        tspsm_bestfit = simulatedAnnealing(problem_fit,dest_init_state,10,tsp_itera)
#        tsp_sm_et = timeit.timeit()
#        print(tsp_sm_et - tsp_sm_st)
#        print("Genetic Algorithm")
#        tsp_ga_st = timeit.timeit()
#        tspga_bestfit = genetic(problem_fit,dest_init_state,10,tsp_itera)
#        tsp_ga_et = timeit.timeit()
#        print(tsp_ga_et - tsp_ga_st)
#        print("MIMIC")
#        tsp_mi_st = timeit.timeit()
#        tspmi_bestfit = mimic(problem_fit,dest_init_state,10,tsp_itera)
#        tsp_mi_et = timeit.timeit()
#        print(tsp_mi_et - tsp_mi_st)
#        tsp_rhfit.append(tsprh_bestfit)
#        tsp_smfit.append(tspsm_bestfit)
#        tsp_gafit.append(tspga_bestfit)
#        tsp_mifit.append(tspmi_bestfit)
#    print("TSP Result")
#    print(len(coords_list))
#    print(tsp_iterations)
#    print(tsp_rhfit)
#    print(tsp_smfit)
#    print(tsp_gafit)
#    print(tsp_mifit)


#knapsack
print("Travelling Knapsack")
max_weight_pct = 0.6
for items in range(5,10,5):
    ks_weights = random.sample(range(1,items*2),items)
    ks_values = list(range(1,items+1))
    max_weight_pct = 0.6
    weight_range = random.randint(2,int(items/2)+1)
    init_state_ks = np.random.randint(weight_range,size=items)
    fitness_ks = mlrose.Knapsack(ks_weights,ks_values,max_weight_pct)
    problem_ks = mlrose.DiscreteOpt(length = init_state_ks.shape[0], fitness_fn = fitness_ks, maximize=False)
    #ks_iterations = [10,20,50,75,100,125,200,250,300]
    ks_iterations = [1]
    max_attempts = 10
    ks_rh_fitness= []
    ks_sm_fitness = []
    ks_ga_fitness = []
    ks_mi_fitness = []
    for ks_iter in ks_iterations:
        #ks_rh_starttime = timeit.timeit()
        #ks_rh_bestfit  =randomHill(problem_ks,init_state_ks,max_attempts,ks_iter)
        #ks_rh_endtime = timeit.timeit()
        #print("RH Start Time")
        #print(ks_rh_endtime - ks_rh_starttime)
        #ks_sm_startime = timeit.timeit()
        #ks_sm_bestfit =simulatedAnnealing(problem_ks,init_state_ks,max_attempts,ks_iter)
        #ks_sm_endtime = timeit.timeit()
        #print("SM Time")
        #print(ks_sm_endtime - ks_sm_startime)
        #ks_ga_startime = timeit.timeit()
        #ks_ga_bestfit =genetic(problem_ks,init_state_ks,max_attempts,ks_iter)
        #ks_ga_endtime = timeit.timeit()
        #print("GA Time")
        #print(ks_ga_endtime - ks_ga_startime)
        ks_mi_startime = timeit.timeit()
        ks_mi_bestfit =mimic(problem_ks,init_state_ks,max_attempts,ks_iter)
        ks_mi_endtime = timeit.timeit()
        print(ks_mi_endtime -ks_mi_startime )
        #ks_rh_fitness.append(ks_rh_bestfit)
        #ks_sm_fitness.append(ks_sm_bestfit)
        #ks_ga_fitness.append(ks_ga_bestfit)
        #ks_mi_fitness.append(ks_mi_bestfit)
    print("Results")
    print(items)
    #print(ks_iterations)
    #print(ks_rh_fitness)
    #print(ks_sm_fitness)
    #print(ks_ga_fitness)
    print(ks_mi_fitness)


