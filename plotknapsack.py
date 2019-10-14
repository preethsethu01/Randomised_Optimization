import matplotlib.pyplot as plt

def plot_graph(input_size,X,Y):
    plt.figure()
    plt.title(input_size)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.plot(X,Y[0],label="Randomhill")
    plt.plot(X,Y[1],label="SAnnealing")
    plt.plot(X,Y[2],label="GeneticAlg")
    plt.plot(X,Y[3],label="MIMIC")
    plt.legend(loc="best")
    plt.savefig(input_size+'.png')
    plt.show()

X = [10, 25, 50, 75, 100, 125, 200, 250, 300]

Y_5 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] 

plot_graph("Knapsack Input Size =5",X,Y_5)

Y_15 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

plot_graph("Knapsack Input Size =15",X,Y_15)

Y_25 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[110.0, 57.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

plot_graph("Knapsack Input Size =25",X,Y_25)

Y_35 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[208.0, 203.0, 33.0, 33.0, 33.0, 33.0, 33.0, 33.0, 33.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

plot_graph("Knapsack Input Size =35",X,Y_35)

Y_45 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

plot_graph("Knapsack Input Size =45",X,Y_45)


