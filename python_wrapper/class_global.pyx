from libc.stdint cimport uint8_t, uint32_t

cdef extern from *:
	ctypedef void* D2 "2"

cdef extern from "Class_Global.hpp":
	cdef cppclass Class_Global[T]:
		# Length of the logical domain
		const uint32_t max_length
		# Number of children of an octant
		const uint8_t  nchildren
		# Number of faces of an octant
		const uint8_t  nfaces
		# Number of nodes of an octant                
		const uint8_t  nnodes
		# Number of nodes per face of an octant
		const uint8_t  nnodesperface
		# Bytes occupation of an octant
		const uint8_t  octantBytes
		# Bytes occupation of the index of an octant
		const uint8_t  globalIndexBytes
		# Bytes occupation of the refinement marker of an octant
		const uint8_t  markerBytes
		# Bytes occupation of the level of an octant
		const uint8_t  levelBytes
		# Bytes occupation of a boolean
		const uint8_t  boolBytes
		# Index of the face of an octant neighbour through the i-th
		# face of the current octant
        #        const uint8_t  oppface[4]
	#	# Local indices of faces sharing the i-th node of an octant
        #        const uint8_t  nodeface[4][2]
	#	# Local indices of nodes of the i-th face of an octant
        #        const uint8_t  facenode[4][2]
	#	# Components (x,y,z) of the normals per face (z=0 in 2D)
        #        const int8_t   normals[4][3] 

cdef class Py_Class_Global_D2:
	cdef Class_Global[D2]* thisptr

	def __cinit__(self):
		self.thisptr = new Class_Global[D2]()

	def __dealloc__(self):
		if (self.thisptr is not NULL):
			del self.thisptr

	property nnodes:
		def __get__(self):
			return self.thisptr.nnodes 

	property nfaces:
		def __get__(self):
			return self.thisptr.nfaces

	property nchildren:
		def __get__(self):
			return self.thisptr.nchildren
