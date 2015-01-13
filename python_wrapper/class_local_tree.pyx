cdef extern from *:
	ctypedef void* D2 "2"

cdef extern from "Class_Local_Tree.hpp":
	cdef cppclass Class_Local_Tree[T]:
		Class_Local_Tree() except +
		
		# Compute the connectivity of octants and store the coordinates
		# of nodes
		void computeConnectivity()

		# Clear the connectivity of octants
		void clearConnectivity()

		# Update the connectivity of octants
		void updateConnectivity()

cdef class Py_Class_Local_Tree_D2:
	cdef Class_Local_Tree[D2]* thisptr

	def __cinit__(self):
		self.thisptr = new Class_Local_Tree[D2]()

	def __dealloc__(self):
		del self.thisptr

	#def compute_connectivity(self):
	#	self.thisptr.computeConnectivity()

	#def clear_connectivity(self):
	#	self.thisptr.clearConnectivity()

	#def update_connectivity(self):
	#	self.thisptr.updateConnectivity()
