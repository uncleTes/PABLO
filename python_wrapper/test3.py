import class_para_tree
import utils
import class_global

def main():
	iteration = 0
	# Instantation of a 2D para_tree object
	pablo3 = class_para_tree.Py_Class_Para_Tree_D2()

	# Compute the connectivity and write the para_tree
	pablo3.compute_connectivity()
	pablo3.write("Pablo3_iter" + str(iteration))

	# Refine globally two level and write the para_tree
	for iteration in xrange(1, 3):
		pablo3.adapt_global_refine()
		pablo3.update_connectivity()
		pablo3.write("Pablo3_iter"+ str(iteration))

	
	# Define a center point and a radius
	xc = yc = 0.5
	radius = 0.4
	nnodes = class_global.Py_Class_Global_D2().nnodes

	# Simple adapt() 6 times the octants with at least one node inside the 
	# circle
	for iteration in xrange(3, 9):
		nocts = pablo3.get_num_octants()
		for i in xrange(0, nocts):
			# Set NO 2:1 balance for every octant
			pablo3.set_balance(i, False)
			nodes = pablo3.get_nodes(i)
			for j in xrange(0, nnodes):
				(x, y) = nodes[j][:2]
				if (((x-xc) * (x-xc)) + ((y-yc) * (y-yc)) <= 
				    (radius * radius)):
					pablo3.set_marker(i, 1, from_index = True)
		# Adapt octree
		pablo3.adapt()

		# Update the connectivity and write the para_tree
		pablo3.update_connectivity();
		pablo3.write("Pablo3_iter" + str(iteration))

	# Coarse globally one level and write the para_tree
	iteration = 9
	pablo3.adapt_global_coarse()
	pablo3.update_connectivity()
	pablo3.write("Pablo3_iter" + str(iteration))

	# Define a center point and a radius
	xc = yc = 0.35
	radius = 0.15

	# Simple adapt() 5 times the octants with at least one node inside the 
	# circle
	for iteration in xrange(10, 15):
		nocts = pablo3.get_num_octants()
		for i in xrange(0, nocts):
			pablo3.set_balance(i, False)
			nodes = pablo3.get_nodes(i)
			for j in xrange(0, nnodes):
				(x, y) = nodes[j][:2]
				# Set refinement marker=-1 (coarse it one time) 
				# for octants inside a circle
				if (((x-xc) * (x-xc)) + ((y-yc) * (y-yc)) <= 
				    (radius * radius)):
					pablo3.set_marker(i, -1, from_index = True)
		
		# Adapt octree, update connectivity and write
		pablo3.adapt()
		pablo3.update_connectivity()
		pablo3.write("Pablo3_iter" + str(iteration))

	del pablo3
	return 0

if __name__ == "__main__":
	wrapper = utils.Py_Wrap_MPI(main)
	result = wrapper.execute(None)
