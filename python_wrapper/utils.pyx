from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free
import sys

cdef extern from *:
	ctypedef int RTI "int" # RTI = ReturnTypeInteger
	ctypedef void* PVP "void*" # PV = ParametersVoidPointer

ctypedef RTI (*method)(PVP parameters, void* userData)

cdef extern from "utils.hpp":
	cdef cppclass WrapMPI[RT, P]: # RT = ReturnType; P = Parameters
		WrapMPI(method _method, 
			void* userData, 
			int argc, 
			char* argv[]) except +

		RT execute(P parameters)

cdef RTI callBack(PVP parameters, void* method):
	if (<object>parameters is None):
		return (<object>method)()
	else:
		return (<object>method)(<object>parameters)

cdef class Py_Wrap_MPI:
	cdef WrapMPI[RTI, PVP]* thisptr;

	def __cinit__(self, py_method):
		cdef char** c_argv= <char**>malloc(sizeof(char*) * 
						   len(sys.argv))

		if (c_argv is NULL):
			raise MemoryError()

		try:
			for index, arg in enumerate(sys.argv):
				c_argv[index] = arg

			self.thisptr = new WrapMPI[RTI, PVP](callBack, 
						     	     <void*>py_method,
						     	     len(sys.argv), 
						     	     c_argv)
		finally:
			free(c_argv)

	def __dealloc__(self):
		if (self.thisptr):
			del self.thisptr

	def execute(self, parameters):
		return self.thisptr.execute(<PVP>parameters)

cdef extern from "utils.hpp":
	bool fileExists(string& fileName)
		
def file_exists(file_name):
	return fileExists(file_name)

cdef extern from "utils.hpp":
	cdef cppclass FloatRandom:
		FloatRandom(long int c_seed, int min, int max)
		float random()

cdef class Py_Float_Random:
	cdef FloatRandom* thisptr;

	def __cinit__(self, seed, min, max):
		self.thisptr = new FloatRandom(seed, min, max)

	def __dealloc__(self):
		if (self.thisptr):
			del self.thisptr

	def random(self):
		return self.thisptr.random()
