#from mpi4py import MPI
import class_para_tree
import class_octant
import utils
import math

def main():
	# instantation of a 2D para_tree object with default constructor
	pablo1 = class_para_tree.Py_Class_Para_Tree_D2()
	#pablo1.set_balance(0, False)

	# Compute the connectivity and write the para_tree
	pablo1.compute_connectivity()
	pablo1.write("Pablo1_iter0")

	# Refine globally one level and write the para_tree
	pablo1.adapt_global_refine()
	pablo1.update_connectivity()
	pablo1.write("Pablo1_iter1")

	# Define a center point
	xc = yc = 0.5

	# Set NO 2:1 balance in the right side of domain
	nocts = pablo1.get_num_octants()

	for i in xrange(0, nocts):
		(x, y, z) = pablo1.get_center(i)

		if (x > xc):
			pablo1.set_balance(i, False)

	# Define a radius
	radius = 0.4

	# Simple adapt() nref1 times in upper area of domain
	nref1 = 6
	for iter in xrange(0, nref1):
		nocts = pablo1.get_num_octants()

		for i in xrange(0, nocts):
			octant = pablo1.get_octant(i)
			(x, y, z) = pablo1.get_center(octant, from_octant = True)
			
			# Set refinement marker = 1 for octants inside a circle
			if ((pow((x - xc), 2.0) + pow((y - yc), 2.0) 
				< pow(radius, 2.0)) 
				and (y < yc)):
				pablo1.set_marker(octant, 1)

		# Adapt octree, update connectivity and write
       		pablo1.adapt()
       		pablo1.update_connectivity()
		pablo1.write("Pablo1_iter"+ str(iter + 2))
	
	# While adapt() nref2 times in lower area of domain
	# (Useful if you work with center of octants)
	nref2 = 5
	iter = 0
	done = True

	while iter <= nref2:
		done = True
		while done:
			nocts = pablo1.get_num_octants()
			for i in xrange(0, nocts):
				octant = pablo1.get_octant(i)
				py_octant = class_octant.Py_Class_Octant_D2(octant, python_octant = True)
				(x, y, z) = pablo1.get_center(octant, from_octant = True)
				if ((pow((x - xc), 2.0) + pow((y - yc), 2.0) 
					< pow(radius,2.0)) 
					and (y > yc) 
					and iter <= nref2 
					and py_octant.level <= iter + 1):

					# Set refinement marker=1 for octants inside a circle
					pablo1.set_marker(octant, 1)
			
			done = pablo1.adapt()
			pablo1.update_connectivity()
			pablo1.write("Pablo1_iter" + str(iter+nref1+2))

		iter += 1

	# Globally refine one level, update the connectivity and write the para_tree
	pablo1.adapt_global_refine()
	pablo1.update_connectivity()
	pablo1.write("Pablo1_iter"+ str(iter + nref1 + 3))


	del pablo1	
	return 0

	
if __name__ == "__main__":
	wrapper = utils.Py_Wrap_MPI(main)
	result = wrapper.execute(None)
