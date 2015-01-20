import class_para_tree_3D
import class_global_3D
import utils
import random
import math
import copy

class Bubble(object):
	# Tuples are immutable, lists not
	def __init__(self, center = [None, None, None], radius = None):
		self.center = center
		self.radius = radius

	def __str__(self):
		kind_of_object = "Bubble object"
		center = str(self.center)
		radius = str(self.radius)
		return (kind_of_object + ":\n" +
			"center = " + center + "\n" +
			"radius = " + radius)

def main():
	iteration = 0
	idx = 0

	# Instantation of a 3D para_tree object
	pabloBB = class_para_tree_3D.Py_Class_Para_Tree()
	# Set 2:1 balance for the octree
	pabloBB.set_balance(idx, True)
	# Refine globally four level and write the para_tree
	for iteration in xrange(1, 5):
		pabloBB.adapt_global_refine()

	# PARALLEL TEST: Call loadBalance, the octree is now distributed over 
	# the processes
	pabloBB.load_balance()
	
	# Define a set of bubbles
	#random.seed() # Current system time is used
	seed = 1418143772
	# Instantiation of a Py_Float_Random object which will return pseudo-
	# random or random (depending by the seed) float between, in this case,
	# 0 and 1
	py_random_generator = utils.Py_Float_Random(seed, 0, 1)

	nb = 100
	BB, BB0, DZ, OM, AA = ([] for i in xrange(0, 5))
	# Initialiazion of some Bubbles
	for i in xrange(0, nb):
		randc = [(0.8 * py_random_generator.random()) + 0.1,
			 (0.8 * py_random_generator.random()) + 0.1,
			 py_random_generator.random() - 0.5]
		randr = (0.05 * py_random_generator.random()) + 0.04
		dz = 0.005 + (0.05 * py_random_generator.random())
		omega = 0.5 * py_random_generator.random()
		aa = 0.15 * py_random_generator.random()
		bb = Bubble(center = randc, radius = randr)
		BB.append(bb), DZ.append(dz), OM.append(omega), AA.append(aa)
	# Making a deep copy...		
	BB0 = copy.deepcopy(BB)

	t0 = 0
	t = t0
	Dt = 0.5
	# Define vectors of data
	nocts = pabloBB.get_num_octants()
	nghosts = pabloBB.get_num_ghosts()
	oct_data = [99] * nocts
	ghost_data = [0.0] * nghosts
	
	#Adapt itend times with data injection on new octants
	itstart, itend, nrefperiter = 1, 6, 3
	for iteration in xrange(itstart, itend):
		if(pabloBB.rank == 0):
			print("iter " + str(iteration))
		t += Dt

		for i in xrange(0, nb):
			BB[i].center = [BB0[i].center[0],
					BB0[i].center[1],
					BB[i].center[2] + (Dt * DZ[i])]

		
		nnodes = class_global_3D.Py_Class_Global().nnodes
		
		parameters = {"iteration" : iteration, 
			      "nrefperiter" : nrefperiter,
			      "nocts" : nocts,
			      "nnodes" : nnodes,
			      "nb" : nb,
			      "BB" : BB}		

		nocts = pabloBB.for_test_bubbles(**parameters)
	del pabloBB
	return 0	

if __name__ == "__main__":
	wrapper = utils.Py_Wrap_MPI(main)
	result = wrapper.execute(None)
