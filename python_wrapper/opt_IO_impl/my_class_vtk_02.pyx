from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int8_t, uintptr_t
import numpy as np
cimport numpy as np

include"../class_para_tree.pyx"

cdef extern from "My_Class_VTK_02.hpp":
	cdef cppclass My_Class_VTK_02[G, D, dim]:
		My_Class_VTK_02(
			D* data_,
			G& grid_,
			string dir_, 
			string name_, 
			string cod_, 
			int ncell_, 
			int npoints_, 
			int nconn_
			) except +

		void printVTK()

		void Add_Data(string name_, 
		              int comp_, 
		              string type_, 
		              string loc_, 
		              string cod_)


cdef class Py_Class_VTK:
	cdef My_Class_VTK_02[Class_Para_Tree[D2],
			double,
			D2]* thisptr

	def __cinit__(self, 
			np.ndarray[double, ndim = 2, mode = "c"] data,
			octree,
			string directory,
			string file_name,
			string file_type,
			int n_cells,
			int n_points,
                        int n_conn):
		self.thisptr = new My_Class_VTK_02[Class_Para_Tree[D2],
						double,
						# http://stackoverflow.com/questions/22055196/how-to-pass-numpy-array-to-cython-function-correctly
						# For one dimensional numpy
						# array see:
						# https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
						# http://trac.cython.org/ticket/838
						D2](&data[0, 0],
							(<Py_Class_Para_Tree_D2>octree).thisptr[0],
							directory,
							file_name,
							file_type,
							n_cells,
							n_points,
							n_conn)

	def __dealloc__(self):
		del self.thisptr

	def print_vtk(self):
		self.thisptr.printVTK()

	def add_data(self,
		     string dataName,
		     int dataDim,
		     string dataType,
		     string pointOrCell,
		     string fileType):
		self.thisptr.Add_Data(dataName,
			              dataDim,
			              dataType,
			              pointOrCell,
			              fileType)

