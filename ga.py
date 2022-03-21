import sys
import argparse
import time
import random
import numpy as np
from ls_2opt import loadTSP
from ls_2opt import funksioni
from ls_2opt import localOptimize

def initPopulation(popsize, n, dm, maxiter):
	
	start_l = []

	solutions = []
	fitness = []

	i = 0
	for i in range(1,popsize+1):
		np.random.seed(i)

		s,cost,init_cost,t = funksioni(n, dm,
						init_strategy='greedy',
						n_strategy='best_improve',
						maxiter=maxiter,
						start_l=start_l)
		solutions.append(s)
		fitness.append(cost)
		if t == maxiter:
			print("Warning: maxiter reached in initializations.")
	# Kthimi i popullates si nje tuple i fitnesit dhe zgjidhjeve
	return (np.array(solutions), np.array(fitness))

def selectParents(population, strategy='tournament', k=2):
	
	if strategy == 'tournament':
		I = [i for i in range(0, population[1].size)]
		parents = []
		for parent in [1, 2]:
			pool = random.sample(I,k)
			
			rel_winner_i = np.argmin(population[1][pool])
			winner_i = pool[rel_winner_i]
			parents.append(population[0][winner_i])
			# Largimi i fituesit qe mos me zgjedh perseri
			I.remove(winner_i)
		return parents[0], parents[1]

	sys.exit("Selection Mechanism for Parents not supported.")

def addEdge(child, parent, dm):
	
	cities = dm.shape[0]
	len_o = len(child)
	l_c = child[len_o - 1]

	r_i = parent.index(l_c)

	if r_i > 0:
		p_c = parent[r_i - 1]
	else:
		p_c = parent[cities - 1]
	if r_i < cities - 1:
		n_c = parent[r_i + 1]
	else:
		n_c = parent[0]

	p_flag = True
	n_flag = True
	if p_c in child:
		p_flag = False
	if n_c in child:
		n_flag = False
	if p_flag and n_flag:
		if dm[l_c,p_c] < dm[l_c,n_c]:
			child.append(p_c)
		else: 
			child.append(n_c)
	elif p_flag:
		child.append(p_c)
	elif n_flag:
		child.append(n_c)
	else:
		return False
	return True

def crossover(p1, p2, strategy='MPX', dm=None):
	
	if strategy == 'OX':
		p1 = p1.tolist()
		p2 = p2.tolist()

		last_city = p1.pop()

		del p2[len(p2)-1]
		k = int(1/3 * len(p1)) 
		offspring = []

		while p1:
			p11_len = min(len(p1),k)
			p11 = p1[:p11_len]
			offspring.extend(p11)
			p1 = p1[p11_len:]

			p2 = [city for city in p2 if city not in p11]

			tmp = p1
			p1 = p2
			p2 = tmp
		offspring.append(last_city)
		return offspring

	if strategy == 'MPX':
		donor = p1.tolist()
		receiver = p2.tolist()

		cities = len(donor) - 1
		del donor[cities]
		del receiver[cities]
		i = np.random.randint(0,cities)
		k = np.random.randint(int(0.2 * cities), int(0.4 * cities))

		offspring = []
		j = (i + k) % cities
		if j > i:
			offspring = donor[i:j+1]
		else:
			offspring = donor[i:] + donor[:j+1]

		while len(offspring) < cities:

			if not addEdge(offspring, receiver, dm):

				if not addEdge(offspring, donor, dm):

					len_o = len(offspring)
					l_c = offspring[len_o - 1]
					r_i = receiver.index(l_c)
					c_i = (r_i + 2) % cities
					while receiver[c_i] in offspring:
						c_i += 1
						c_i = c_i % cities
					offspring.append(receiver[c_i])
		offspring.append(offspring[0])
		return offspring

	sys.exit("Choosen Crossover Strategy not supported.")

def main(argv):
	parser = argparse.ArgumentParser(description='Solve TSP.')
	parser.add_argument('path', help='Path to .tsp file.')
	parser.add_argument('-ps', '--popsize', help='Size of the population.', type=int, default=10)
	parser.add_argument('-mg', '--maxgen', help='Maximum number of generations.', type=int, default=10000)
	parser.add_argument('-mi', '--maxiter', help='Maximum number of iterations in each local search.', type=int, default=1000)
	parser.add_argument('-sel', '--selection', help='Define selection strategy for parents.', choices=['tournament'], default='tournament')
	parser.add_argument('-co', '--crossover', help='Define crossover strategy. ', choices=['OX','MPX'], default='MPX')
	parser.add_argument('-st', '--stop', help='How many generation should produce no better offspring for the algorithm to stop.', type=int, default=-1)
	args = parser.parse_args(argv[1:])
	
	if args.stop < 1:
		args.stop = args.popsize

	dm = loadTSP('instance.tsp')
	n = dm.shape[0]

	start_t = time.process_time()

	population = initPopulation(args.popsize, n, dm, args.maxiter)
	end_t = time.process_time()
	diff_t_init = end_t - start_t
	print("Successfully generated ", args.popsize, " individuals in ", diff_t_init,"s.")
	print("Avg. time per individual: ", diff_t_init / args.popsize)

	start_t = time.process_time()
	best_fitness = min(population[1])
	best_gen = 0
	counter = 0
	worst_fitness = max(population[1])
	worst_index = np.argmax(population[1]) 
	print("Worst fitness: ", worst_fitness)

	for gen in range(0,args.maxgen):

		if counter >= args.stop:
			print("Genetic Algorithm ", gen, ".",
				  "Could not generate a better offspring for ", counter, 
				  " generations.")
			break
		counter += 1
		random.seed(gen)

		p1, p2 = selectParents(population, args.selection)

		child = crossover(p1, p2, args.crossover, dm)

		child, cost, init_cost, t = localOptimize(child,dm,'best_improve',args.maxiter)

		if cost < worst_fitness:

			if not np.isin([cost], population[1])[0]:
				population[0][worst_index] = child
				population[1][worst_index] = cost

				worst_fitness = max(population[1])
				worst_index = np.argmax(population[1]) 

				if cost < best_fitness:
					best_fitness = cost
					best_gen = gen
				print("Generation ", gen, " produced a fitter child with cost: ", cost)
				counter = 0
	best_fitness = min(population[1])
	end_t = time.process_time()
	diff_t = end_t - start_t
	print("Successfully evolved", gen," generations in ", diff_t,"s.")
	print("Avg. time per generation: ", diff_t / gen)
	print("Best solution cost: ", best_fitness, " evolved in generation: ", best_gen)
	print("Avg. solution quality: ", sum(population[1]) / args.popsize)
	print("Generated ", args.popsize, " individuals in ", diff_t_init,"s.")
	print("Avg. time per individual for initialization: ", diff_t_init / args.popsize)

if __name__ == '__main__':
	main(sys.argv)
