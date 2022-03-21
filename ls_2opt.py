import sys
import argparse
import time
import numpy as np
from scipy import spatial

def loadTSP(path):

	city_coord = []
	EUC_2D = False
	EXPLICIT = False
	n = 0
	distance_matrix = 0
	row = 0
	col = 0
	# parse file
	with open(path) as f:
		data = False
		for line in f:
			if data:
				if 'EOF' in line or 'DISPLAY_DATA_SECTION' in line:
					data = False
				elif EUC_2D:

					city_coord.append(line.split(' ')[1:3])
				elif EXPLICIT:

					for el in line.split(' '):
						if el == '':
							continue
						if row == col:
							row = row + 1
							col = 0
							continue
						distance_matrix[row,col] = el
						distance_matrix[col,row] = el
						col = col + 1
			if not data:
				if 'DIMENSION' in line:
					dim_str = line.split(':')
					n = int(dim_str[1])
				if 'EDGE_WEIGHT_TYPE' in line:
					if 'EUC_2D' in line:
						EUC_2D = True
					elif 'EXPLICIT' in line:
						EXPLICIT = True
					else:
						sys.exit("File format not supported.")
				if EXPLICIT and 'EDGE_WEIGHT_FORMAT' in line:
					if 'LOWER_DIAG_ROW' not in line:
						sys.exit("File format not supported.")
				if EUC_2D and 'NODE_COORD_SECTION' in line:
					data = True
				if EXPLICIT and 'EDGE_WEIGHT_SECTION' in line:
					data = True
					distance_matrix = np.zeros((n,n), dtype=np.int32)
	if EUC_2D:
		city_coord_np = np.array(city_coord, dtype=np.float32)

		distance_matrix = spatial.distance_matrix(city_coord_np, city_coord_np, 2)
		np.around(distance_matrix,decimals=0,out=distance_matrix)
	return distance_matrix

def evalTwoOpt(s,i,j,dm):

	if i == j - 1 or i == j or i == j + 1:
		sys.exit("i is not allowed to be j-1, j or j+1! i="+str(i)+" j="+str(j))

	d_cost = dm[s[i],s[j]] + dm[s[i+1], s[j+1]]
	d_cost = d_cost - dm[s[i],s[i+1]] - dm[s[j],s[j+1]]
	return d_cost

def twoOpt(s,i,j):

	s_new = s[:]

	s_new[i+1:j+1] = reversed(s_new[i+1:j+1])
	return s_new

def genNewSolution(s, cost, dm, strategy):
	count = 0
	n = dm.shape[0]
	if strategy == 'best_improve':

		min_d_cost = 0
		min_ij = None

		for i in range(0, n-2):
			for j in range(i+2, n):
				if j == n-1 and i == 0:
					continue
				
				count += 1
				d_cost = evalTwoOpt(s,i,j,dm) 
				if d_cost < min_d_cost:
					min_d_cost = d_cost
					min_ij = (i,j) 
		if min_ij is not None:
			min_s = twoOpt(s,min_ij[0],min_ij[1]) 
		else:
			min_s = s[:] 
		return min_s, cost+min_d_cost
	if strategy == 'first_improve':

		for i in range(0, n-2):
			for j in range(i+2, n):
				if j == n-1 and i == 0:
					continue
				count += 1
				d_cost = evalTwoOpt(s,i,j,dm) #O(1)
				if d_cost < 0:
					return twoOpt(s,i,j), cost+d_cost 
		return s[:], cost

	sys.exit("Neighbourhood-Search-Strategy not supported.")

def eval(s, dm):

	cost = 0
	n = dm.shape[0]
	for i in range(0,n):
		cost += dm[s[i],s[i+1]]
	return cost

def initSolution(n,strategy,dm=None,start_l=None):

	if strategy == 'random':

		cities = [i for i in range(0,n)]
		s = np.random.choice(cities, size=n, replace=False).tolist()
		s.append(s[0])
		return s
	if strategy == 'greedy':

		cities = [i for i in range(0,n)]
		start = np.random.randint(0,n)
		if start_l is not None:
			# Regenerate starting point if already used.
			while start in start_l:
				start = np.random.randint(0,n)
			start_l.append(start)
		s = [start]
		last = start
		while len(s) != n:
			s_ind = np.argsort(dm[last,], kind='heapsort')
			for i in s_ind:
				if i not in s:
					s.append(i)
					last = i
					break
		s.append(start)
		return s

	sys.exit("Initial solution generation strategy not supported.")

def localOptimize(init_solution,dm,n_strategy,maxiter):

	s = init_solution

	init_cost = eval(s, dm)
	cost = init_cost

	t = 0
	for t in range(1, int(maxiter + 1)):

		s_new, cost_new = genNewSolution(s, cost, dm, n_strategy)
		if cost_new == cost:
			break
		s = s_new
		cost = cost_new
	return s, cost, init_cost, t

def funksioni(n,dm,init_strategy,n_strategy,maxiter,start_l=None):
        
	s = initSolution(n,init_strategy,dm,start_l)
	
	return localOptimize(s,dm,n_strategy,maxiter)

def main(argv):
	parser = argparse.ArgumentParser(description='Solve TSP with 2-opt.')
	parser.add_argument('path', help='Path to .tsp file.')
	parser.add_argument('-mi', '--maxiter', help='Maximum number of Iterations.', type=int, default=1000)
	parser.add_argument('-s', '--strategy', help='Define neighbourhood search strategy.', choices=['first_improve','best_improve'], default='best_improve')
	parser.add_argument('-in', '--n_init_solutions', help='Number of initial solutions to test.', type=int, default=1)
	parser.add_argument('-is', '--init_strategy', help='Define initial solution strategy.', choices=['random','greedy'], default='random')
	args = parser.parse_args(argv[1:])

	dm = loadTSP(args.path)
	n = dm.shape[0]

	if args.init_strategy == 'greedy':
		if args.n_init_solutions > n:
			sys.exit("Using nearest neighbour heuristic, there cannot \
					 be more initial solutions then cities!")

	sum_time = 0
	sum_t = 0
	sum_cost = 0
	best_s = 0
	best_cost = 0
	best_time = 0
	best_seed = 0
	best_t = 0
	run = 0
	start_l = []
	for run in range(1,args.n_init_solutions+1):
		print("Run: ", run)
		np.random.seed(run)
		start_t = time.process_time()

		s,cost,init_cost,t = funksioni(n, dm,
						args.init_strategy,
						args.strategy,
						args.maxiter,
						start_l)
		
		end_t = time.process_time()
		diff_t = end_t - start_t

		if run == 1:
			best_s = s
			best_cost = cost
			best_time = diff_t
			best_seed = run
			best_t = t
		elif cost < best_cost:
			best_s = s
			best_cost = cost
			best_time = diff_t
			best_seed = run
			best_t = t
		sum_time += diff_t
		sum_t += t
		sum_cost += cost

		print("Solution with costs ",cost," found in ", diff_t, "s after ", t, " iterations.")
		print("(Initial solution costs: ", init_cost, ")")

	print("Statistics of multistart:")
	print(best_cost, " ... costs of best solution")
	print(best_time, " ... time of best solution")
	print(best_t, " ... iterations till convergence for best solution")
	print(best_seed, " ... seed of best solution")
	print(run, " ... inital solutions generated")
	print(sum_cost/run, " ... average quality of solution")
	print(sum_time/run, " ... average time for solving one tsp instance")
	print(sum_t/run, " ... average iterations till convergence")

if __name__ == "__main__":
	main(sys.argv)
