#from mpi4py import MPI
import class_para_tree
import utils

def main():

	# instantation of a 2D para_tree object with default constructor
	ptreedefault = class_para_tree.Py_Class_Para_Tree_D2()

	# write the para_tree in physical domain
	ptreedefault.write("Pablo0_default")
        # write the para_tree in logical domain.*/
	ptreedefault.write_logical("Pablo0_default_logical")

	#level0 = MAX_LEVEL_2D
	X = 10.0
	Y = 20.0
	Z = 0.0
	L = 250.0

	# instantation of a 2D para_tree object with custom constructor
        ptreecustom = class_para_tree.Py_Class_Para_Tree_D2(X, Y, Z, L)
                
	# write the para_tree in physical domain
	ptreecustom.write("Pablo0_custom")
	# write the para_tree in logical domain
	ptreecustom.write_logical("Pablo0_custom_logical")

	del ptreedefault
	del ptreecustom

	return 0

if __name__ == "__main__":

	wrapper = utils.Py_Wrap_MPI(main)
	result = wrapper.execute(None)
