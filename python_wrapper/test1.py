from mpi4py import MPI
import class_para_tree
import utils

def main():
	# instantation of a 2D para_tree object with default constructor
	pablo1 = class_para_tree.Py_Class_Para_Tree_D2()

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

	del pablo1
	
	return 0
	
if __name__ == "__main__":
	wrapper = utils.Py_Wrap_MPI(main)
	result = wrapper.execute(None)
