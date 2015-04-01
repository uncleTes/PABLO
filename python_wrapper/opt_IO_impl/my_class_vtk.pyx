from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int8_t, uintptr_t

include"../class_para_tree.pyx"

cdef extern from "My_Class_VTK.hpp":
	cdef cppclass My_Class_VTK[N, C, G, D, dim]:
		My_Class_VTK(
                        const N& nodes_,
			const N& ghostNodes_,
			const C& connectivity_,
			const C& ghostConnectivity_,
			D& data_,
			G& grid_,
			string dir_, 
			string name_, 
			string cod_, 
			int ncell_, 
			int npoints_, 
			int nconn_
			) except +

		void printVTK()

cdef class Py_Class_VTK:
	cdef My_Class_VTK[u32vector2D, 
			u32vector2D,
			Class_Para_Tree[D2],
			dvector,
			D2]* thisptr

	#def __cinit__(self, 
	#		u32vector2D nodes,
	#		u32vector2D ghost_nodes,
	#		u32vector2D connectivity,
	#		u32vector2D ghost_connectivity,
	#		dvector data,
	#		octree,
	#		string directory,
	#		string file_name,
	#		string file_type,
	#		int n_cells,
	#		int n_points,
        #                int n_conn):
	#	self.thisptr = new My_Class_VTK[u32vector2D,
	#					u32vector2D,
	#					Class_Para_Tree[D2],
	#					dvector,
	#					D2](nodes,
	#						ghost_nodes,
	#						connectivity,
	#						ghost_connectivity,
        #                                                data,
	#						(<Py_Class_Para_Tree_D2>octree).thisptr[0],
	#						directory,
	#						file_name,
	#						file_type,
	#						n_cells,
	#						n_points,
	#						n_conn)

	def __cinit__(self, *args):
		self.thisptr = new My_Class_VTK[u32vector2D, 
						u32vector2D, 
						Class_Para_Tree[D2],
						dvector,
						D2](<u32vector2D>args[0], 
						<u32vector2D>args[1], 
						<u32vector2D>args[2], 
						<u32vector2D>args[3], 
						<dvector>args[4],
						#<Class_Para_Tree[D2]&>args[5],
						(<Py_Class_Para_Tree_D2>args[5]).thisptr[0], 
						<string>args[6], 
						<string>args[7], 
						<string>args[8], 
						<int>args[9],
						<int>args[10],
						<int>args[11])


	def __dealloc__(self):
		del self.thisptr

	def print_vtk(self):
		self.thisptr.printVTK()

