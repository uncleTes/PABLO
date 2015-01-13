import class_para_tree
import utils
import class_global

def main():
	# Instantation of a 2D para_tree object
	pablo2 = class_para_tree.Py_Class_Para_Tree_D2()

	# Compute the connectivity and write the para_tree
	pablo2.compute_connectivity()
	pablo2.write("Pablo2_iter0")

	# Refine globally two level and write the para_tree
	for iteration in xrange(1, 3):
        	pablo2.adapt_global_refine()
        	pablo2.update_connectivity()
        	pablo2.write("Pablo2_iter" + str(iteration))

	# Define a center point and a radius
	xc = yc = 0.5
	radius = 0.4

	# Simple adapt() 6 times the octants with at least one node inside the circle
	for iteration in xrange(3, 9):
        	nocts = pablo2.get_num_octants()
		for i in xrange(0, nocts):
			nodes = pablo2.get_nodes(i)
			nnodes = class_global.Py_Class_Global_D2().nnodes

			for j in xrange(0, nnodes):
                        	(x, y) = (nodes[j][0], nodes[j][1])

				if ((pow((x - xc), 2.0) + pow((y - yc), 2.0) 
					<= pow(radius, 2.0))):
					pablo2.set_marker(i, 1, from_index = True)
		# Adapt octree
		pablo2.adapt()

		# Update the connectivity and write the para_tree.*/
		pablo2.update_connectivity()
		pablo2.write("Pablo2_iter" + str(iteration))

	del pablo2

	return 0

if __name__ == "__main__":
	wrapper = utils.Py_Wrap_MPI(main)
	result = wrapper.execute(None)
