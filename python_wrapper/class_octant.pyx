from libc.stdint cimport uint32_t, uint8_t, int8_t, uintptr_t
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free

cdef extern from *:
	ctypedef void* D2 "2"

cdef extern from "Class_Octant.hpp":
	cdef cppclass Class_Octant[T]:
		Class_Octant() except +
		Class_Octant(Class_Octant[T]& octant) except +

		uint32_t x
		uint32_t y
		uint8_t level
		int8_t marker
		#bool info[12]

		uint32_t getX() const
		uint32_t getY() const
		double* getCenter()
		uint32_t getSize() const
		void setMarker(int8_t marker)
		void setBalance(bool balance)
		bool getNotBalance() const
		
		# Get the level of an octant
		# return ---> level of octant
		uint8_t getLevel() const

cdef class Py_Class_Octant_D2:
	cdef Class_Octant[D2]* thisptr

	def __cinit__(self, uintptr_t oct_ptr, python_octant = False):

		if (python_octant):
			self.thisptr = new Class_Octant[D2]((<Class_Octant[D2]*><void*>oct_ptr)[0])
		else:
			if (oct_ptr is None):
				self.thisptr = new Class_Octant[D2]()
			elif (type(oct_ptr) is Py_Class_Octant_D2):
				self.thisptr = new Class_Octant[D2]((<Py_Class_Octant_D2>oct_ptr).thisptr[0])
			else:
				print("Wrong type of intialization, dude." + 
				      " Please retry.")
				del self

	def __dealloc__(self):
		del self.thisptr

	def get_level(self):
		return <uint8_t>self.thisptr.getLevel()

	def get_size(self):
		return self.thisptr.getSize()

	def set_marker(self, int8_t marker):
		self.thisptr.setMarker(marker)
               
	# Logic coordinates
	def get_center(self):
		cdef vector[double] center
		py_center = []

		center = <vector[double]>self.thisptr.getCenter()

		for i in xrange(0, 3):
			py_center.append(center[i])
	
		return py_center

	property x:
		def __get__(self):
			return self.thisptr.getX()

	property y:
		def __get__(self):
			return self.thisptr.getY()

	property level:
		def __get__(self):
			return self.thisptr.getLevel()

	property get_not_balance:
			def __get__(self):
				return self.thisptr.getNotBalance()
