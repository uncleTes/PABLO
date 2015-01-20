import class_para_tree
import class_global
import utils
import random
import math
import copy

class Bubble(object):
	# Tuples are immutable, lists not
	def __init__(self, center = [None, None], radius = None):
		self.center = center
		self.radius = radius

	def __str__(self):
		return ("center = " + str(self.center[0]) + " " + 
				      str(self.center[1]) + " radius = " + 
				      str(self.radius))

def main():
	# Instantation of a 2D para_tree object
	pabloBB = class_para_tree.Py_Class_Para_Tree_D2()

	# Set 2:1 balance for the octree
	idx = 0
	pabloBB.set_balance(idx, True)

	# Refine globally four level and write the para_tree
	for iteration in xrange(1, 7):
		pabloBB.adapt_global_refine()

	# PARALLEL TEST: Call load_balance, the octree is now distributed over the processes
	pabloBB.load_balance()

	# Define a set of bubbles
	#random.seed() # Current system time is used
	seed = 1418143772
	py_gen = utils.Py_Float_Random(seed, 0, 1)

	nb = 100
	BB, BB0, DY, OM, AA = ([] for i in xrange(0, 5))
	
	for i in xrange(0, nb):
		# random.random() returns floating point number in the range 
		# [0.0, 1.0)
		#randc = [0.8 * (random.random()) + 0.1, (random.random()) - 0.5]
		randc = [0.8 * py_gen.random() + 0.1, 
			py_gen.random() - 0.5]
		#randr = 0.05 * (random.random()) + 0.02
		randr = 0.05 * py_gen.random() + 0.02
		#dy = 0.005 + 0.05 * (random.random())
		dy = 0.005 + 0.05 * py_gen.random()
		#omega = 0.5 * (random.random())
		omega = 0.5 * py_gen.random()
		#aa = 0.15 * (random.random())
		aa = 0.15 * py_gen.random()
		bb = Bubble(center = randc, radius = randr)
		BB.append(bb), DY.append(dy) 
		OM.append(omega), AA.append(aa)

	# Making a deep copy...not only the reference are copied, but all the 
	# objects		
	BB0 = copy.deepcopy(BB)

	t0 = 0
	t = t0
	Dt = 0.5

	# Define vectors of data
	nocts = pabloBB.get_num_octants()

	# Adapt itend times with data injection on new octants
	itstart = 1
	itend = 200
	nrefperiter = 4

	for iteration in xrange(itstart, itend):
		if pabloBB.rank == 0:
			print("iter " + str(iteration))
		
		t += Dt

		for i in xrange(0, nb):
			#BB[i].center[0] = (BB0[i].center[0] +  
			#		   AA[i] * math.cos(OM[i] * t))
			#BB[i].center[1] = (BB[i].center[1] + 
			#		   Dt * DY[i])
			BB[i].center = [BB0[i].center[0] + AA[i] * 
							   math.cos(OM[i]*t),
					BB[i].center[1] + Dt * DY[i]]

		nnodes = class_global.Py_Class_Global_D2().nnodes
	
		dict_parameters = {"iteration" : iteration, 
				   "nrefperiter" : nrefperiter,
				   "nocts" : nocts,
				   "nnodes" : nnodes,
				   "nb" : nb,
				   "BB" : BB}		

		#nocts = pabloBB.for_test_bubbles(iteration, nrefperiter, nocts, nnodes, nb, BB)
		nocts = pabloBB.for_test_bubbles(**dict_parameters)
	
	del pabloBB
	return 0

# "cProfile" used for profiling
if __name__ == "__main__":
	#import cProfile
	wrapper = utils.Py_Wrap_MPI(main)
	#cProfile.run("wrapper.execute(None)", sort = "time")
	result = wrapper.execute(None)
