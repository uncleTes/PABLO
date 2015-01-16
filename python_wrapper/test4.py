import class_para_tree
import utils
import class_global

def main():
	iteration = 0
	dim = 2
	# Instantation of a 2D para_tree object
	pablo4 = class_para_tree.Py_Class_Para_Tree_D2()

	# Refine globally four level and write the para_tree
	for iteration in xrange(1, 5):
		pablo4.adapt_global_refine()

	# Define a center point and a radius
	xc, yc = 0.5, 0.5
	radius = 0.25
	nnodes = class_global.Py_Class_Global_D2().nnodes
	nnfaces = class_global.Py_Class_Global_D2().nfaces
	

	# Define vectors of data
	nocts = pablo4.get_num_octants()
	nghosts = pablo4.get_num_ghosts()
	oct_data = [0.0] * nocts
	ghost_data = [0.0] * nghosts

	# Assign a data to the octants with at least one node inside the circle
	for i in xrange(0, nocts):
		nodes = pablo4.get_nodes(i)
		for j in xrange(0, nnodes):
			(x, y) = nodes[j][:2]
			if (((x-xc) * (x-xc)) + ((y-yc) * (y-yc)) <= 
			    (radius * radius)):
				oct_data[i] = 1.0

	# Assign a data to the ghost octants (NONE IT IS A SERIAL TEST) with at least 
	# one node inside the circle
	for i in xrange(0, nghosts):
		ghost_octant = pablo4.get_ghost_octant(i)
		nodes = pablo4.get_nodes(ghost_octant, from_octant = True)
		for j in xrange(0, nnodes):
			(x, y) = nodes[j][:2]
			if (((x-xc) * (x-xc)) + ((y-yc) * (y-yc)) <= 
			    (radius * radius)):
				ghost_data[i] = 1.0

	# Update the connectivity and write the para_tree
	iteration = 0
	pablo4.update_connectivity()
	pablo4.write_test("Pablo4_iter" + str(iteration), oct_data)

	# Smoothing iterations on initial data
	start = 1
	for iteration in xrange(start, start+25):
		oct_data_smooth = [0.0] * nocts
		neigh, neigh_t, isghost, isghost_t = ([] for i in xrange(0, 4))
		for i in xrange(0, nocts):
			neigh = []
			isghost = []
			# Find neighbours through edges (codim=1) and nodes
			#  (codim=2) of the octants
			for codim in xrange(1, dim+1):
				if (codim == 1):
					nfaces = nnfaces
				elif (codim == 2):
					nfaces = nnodes
				
				for iface in xrange(0, nfaces):
					(neigh_t, isghost_t) = pablo4.find_neighbours(i, iface, codim, 
							       neigh_t,isghost_t)
					neigh.extend(neigh_t)
					isghost.extend(isghost_t)

			# Smoothing data with the average over the one ring 
			# neighbours of octants
			oct_data_smooth[i] = oct_data[i]/(len(neigh) + 1)
			for j in xrange(0, len(neigh)):
				if isghost[j]:
					oct_data_smooth[i] += ghost_data[neigh[j]]/(len(neigh) + 1)
				else:
					oct_data_smooth[i] += oct_data[neigh[j]]/(len(neigh) + 1)

		# Update the connectivity and write the para_tree
		pablo4.update_connectivity()
		pablo4.write_test("Pablo4_iter" + str(iteration), oct_data_smooth)

		oct_data = oct_data_smooth

	del pablo4
	return 0

if __name__ == "__main__":
	wrapper = utils.Py_Wrap_MPI(main)
	result = wrapper.execute(None)
