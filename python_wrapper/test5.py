import class_para_tree
import utils
import class_global

def main():
	iteration = 0
	# Instantation of a 2D para_tree object
	pablo5 = class_para_tree.Py_Class_Para_Tree_D2()
	idx = 0

	# Set NO 2:1 balance for the octree
	pablo5.set_balance(idx, False)

	# Refine globally five level and write the para_tree
	for iteration in xrange(1, 6):
		pablo5.adapt_global_refine()

	# Define a center point and a radius
	xc, yc = 0.5, 0.5
	radius = 0.25

	# variables from class "Class_Global"
	nnodes = class_global.Py_Class_Global_D2().nnodes
	nchildren = class_global.Py_Class_Global_D2().nchildren

	# Define vectors of data
	nocts = pablo5.get_num_octants()
	nghosts = pablo5.get_num_ghosts()
	oct_data = [0.0] * nocts
	ghost_data = [0.0] * nghosts

	# Assign a data (distance from center of a circle) to the octants 
	# with at least one node inside the circle
	for i in xrange(0, nocts):
		nodes = pablo5.get_nodes(i)
		center = pablo5.get_center(i)
		for j in xrange(0, nnodes):
			(x, y) = nodes[j][:2]
			if (((x-xc) * (x-xc)) + ((y-yc) * (y-yc)) <= 
			    (radius * radius)):
				oct_data[i] = (((center[0]-xc) * (center[0]-xc)) + 
					       ((center[1]-yc) * (center[1] -yc)))
				if (center[0] <= xc):
					# Set to refine to the octants in the left 
					# side of the domain
					pablo5.set_marker(i, 1, from_index = True)
				else:

					# Set to coarse to the octants in the right 
					# side of the domain
					pablo5.set_marker(i, -1, from_index = True)

	# Update the connectivity and write the para_tree
	iteration = 0
	pablo5.update_connectivity()
	pablo5.write_test("Pablo5_iter" + str(iteration), oct_data)

	# Adapt two times with data injection on new octants
	start = 1
	for iteration in xrange(1, start+2):
		for i in xrange(0, nocts):
			nodes = pablo5.get_nodes(i)
			center = pablo5.get_center(i);
			for j in xrange(0, nnodes):
				(x, y) = nodes[j][:2]
				if (((x-xc) * (x-xc)) + ((y-yc) * (y-yc)) <= 
				    (radius * radius)):
					if (center[0] <= xc):
						# Set to refine to the octants in 
						# the left side of the domain 
						# inside a circle
						pablo5.set_marker(i, 1, from_index = True)
					else:
						# Set to coarse to the octants in the right side of the domain inside a circle
						pablo5.set_marker(i, -1, from_index = True)

		# Adapt the octree and map the data in the new octants
		mapper = []
		#print("before = " + str(mapper))
		# the same as (result, mapper) = pablo5.adapt_mapper(mapper)
		(result, mapper) = pablo5.adapt_mapper(mapper)
		#print("after = " + str(mapper))
		nocts = pablo5.get_num_octants()
		oct_data_new = [0.0] * nocts

		# Assign to the new octant the average of the old children if 
		# it is new after a coarsening; while assign to the new octant 
		# the data of the old father if it is new after a refinement.
		for i in xrange(0, nocts):
			if (pablo5.get_is_new_c(i)):
				for j in xrange(0, nchildren):
					oct_data_new[i] += oct_data[mapper[i]+j]/nchildren
			else:
				oct_data_new[i] += oct_data[mapper[i]]
		
		# Update the connectivity and write the para_tree
		pablo5.update_connectivity()
		pablo5.write_test("Pablo5_iter" + str(iteration), oct_data_new)

		oct_data = oct_data_new

	del pablo5
	return 0

if __name__ == "__main__":
	wrapper = utils.Py_Wrap_MPI(main)
	result = wrapper.execute(None)
