#from mpi4py import MPI
#cdef extern from "mpi-compat.h": pass
#cimport mpi4py.MPI as MPI
#from mpi4py.MPI cimport Intracomm as IntracommType

from libcpp.string cimport string

#cdef MPI.Comm WORLD = MPI.COMM_WORLD
#cdef IntracommType SELF = MPI.COMM_SELF

cdef extern from "Class_Log.hpp":
	cdef cppclass Class_Log:
		Class_Log(string filename) except +
	
		string filename

		void writeLog(string msg)

cdef class Py_Class_Log:
	cdef Class_Log* thisptr

	def __init__(self, file_name):
		self.thisptr = new Class_Log(file_name)

	def __dealloc__(self):
		del self.thisptr

	def write_log(self, message):
		self.thisptr.writeLog(message)
