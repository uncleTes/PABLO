from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef extern from *:
	ctypedef void* D2 "2"

include "class_octant.pyx"
include "class_local_tree.pyx"
include "class_log.pyx"

cdef extern from "Class_Map.hpp":
	cdef cppclass Class_Map[T]:
		Class_Map() except +

# ---------------------TYPEDEFS---------------------
ctypedef public vector[Class_Octant[D2]] OctantsType 
ctypedef public vector[uint32_t]         u32vector
ctypedef public vector[double]           dvector
ctypedef public vector[vector[uint32_t]] u32vector2D
ctypedef public vector[vector[uint64_t]] u64vector2D
ctypedef public vector[vector[double]]   dvector2D
ctypedef public vector[int]              ivector
ctypedef public vector[vector[int]]      ivector2D
# --------------------------------------------------

cdef extern from "Class_Para_Tree.hpp":
	cdef cppclass Class_Para_Tree[T]:
		Class_Log log
		
		# ---------------------------MEMBERS---------------------------
		# Global array containing position of the first possible octant
		# in each processor
		uint64_t* partition_first_desc

		# Global array containing position of the last possible octant
		# in each processor
		uint64_t* partition_last_desc

		# Global array containing global index of the last existing
		# octant in each processor                 
		uint64_t* partition_range_globalidx

		# Global number of octants in the parallel octree
		uint64_t global_num_octants

		# Local indices of border octants per process                            
		map[int, vector[uint32_t]] bordersPerProc

		# Number of processes of the job
		int nproc

		# Global max existing level in the parallel octree
		uint8_t max_depth

		# ---------------------distributed members---------------------
		# Local rank of process
		int rank

		# Local tree in each processor
		Class_Local_Tree[T] octree
		
		# ----------------------auxiliary members----------------------
        	# MPI error flag
		int error_flag

		# True if the octree is the same on each processor, False if the
		# octree is distributed
		bool serial

		# -------------------------map member--------------------------
		# Transoformation map from logical to physical domain
		Class_Map[T] trans
		# -------------------------------------------------------------

		# -------------------------CONSTRUCTORS------------------------
		# Default constructor of Para_Tree. It builds one octant with 
		# node 0 in the Origin (0,0,0) and side of length 1
		Class_Para_Tree(string logfile) except +

		# Constructor of Para_Tree with input parameters. It builds one
		# octant with:
         	# param[in] X ---> coordinate X of node 0
         	# param[in] Y ---> coordinate Y of node 0
         	# param[in] Z ---> coordinate Z of node 0
         	# param[in] L ---> side length of the octant
		Class_Para_Tree(double X, 
				double Y, 
				double Z, 
				double L,
				string logfile) except +
		
		# Constructor of Para_Tree for restart a simulation with input 
		# parameters. For each process it builds a vector of octants. 
		# The input parameters are:
         	# param[in] X ---> physical coordinate X of node 0
         	# param[in] Y ---> physical coordinate Y of node 0
         	# param[in] Z ---> physical coordinate Z of node 0
         	# param[in] L ---> physical side length of the domain
         	# param[in] XY ---> coordinates of octants (node 0) in logical
		#                   domain
         	# param[in] levels ---> level of each octant
		Class_Para_Tree(double X, 
				double Y, 
				double Z,
				double L, 
				ivector2D& XY, 
				ivector& levels,
				string logfile) except +
		# -------------------------------------------------------------

		# ----------------------GET/SET METHODS------------------------
		# -------------------Octant get/set methods--------------------
		# Get the coordinates of an octant, i.e. the coordinates of its 
		# node 0:
		# param[in] oct ---> pointer to target octant
		# return ---> coordinate X of node 0
		double getX(Class_Octant[T]* oct)
		
		# Get the coordinates of an octant, i.e. the coordinates of its 
		# node 0:
		# param[in] oct ---> pointer to target octant
		# return ---> coordinate Y of node 0
		double getY(Class_Octant[T]* oct)

		# Get the coordinates of an octant, i.e. the coordinates of its 
		# node 0.
		# param[in] oct ---> pointer to target octant
		# return coordinate Z of node 0
		double getZ(Class_Octant[T]* oct)

		# Get the size of an octant, i.e. the side length:
		# param[in] oct ---> pointer to target octant
		# return ---> size of octant
		double getSize(Class_Octant[T]* oct)

		# Get the area of an octant (for 2D case the same value of 
		# getSize):
		# param[in] oct ---> pointer to target octant
		# return ---> size of octant
		double getArea(Class_Octant[T]* oct)

		# Get the volume of an octant:
		# param[in] oct ---> pointer to target octant
		# return ---> volume of octant
		double getVolume(Class_Octant[T]* oct)

		# Compute the connectivity of octants and store the coordinates
		# of nodes
		void computeConnectivity()

		# Adapt the octree mesh refining all the octants by one level
		bool adaptGlobalRefine()

		# Update the connectivity of octants
		void updateConnectivity()		

		#---------------------------------------------------------------
		# ------------------Local tree get/set methods------------------
		# Get the local number of octants:
		# return ---> local number of octants.
		uint32_t getNumOctants() const
		
		void write(string)
		void writeLogical(string)

# Wrapper Python for class Class_Para_Tree<2>
cdef class  Py_Class_Para_Tree_D2:
	# Pointer to the object Class_Para_Tree<2>
	cdef Class_Para_Tree[D2]* thisptr

	# ------------------------------Constructor-----------------------------
	# different number of arguments can be passed, so different 
	# Class_Para_Tree<2> constructors can be called
	def __cinit__(self, *args):
		number_of_parameters = len(args)
		
		if (number_of_parameters == 0):
			self.thisptr = new Class_Para_Tree[D2]("PABLO.log")
		elif (number_of_parameters == 1):
			self.thisptr = new Class_Para_Tree[D2](args[0])
		elif (number_of_parameters == 4):
			self.thisptr = new Class_Para_Tree[D2](args[0],
								args[1],
								args[2],
								args[3],
								"PABLO.log")
		elif (number_of_parameters == 5):
			self.thisptr = new Class_Para_Tree[D2](args[0],
								args[1],
								args[2],
								args[3],
								args[4])
		elif (number_of_parameters == 6):
			self.thisptr = new Class_Para_Tree[D2](args[0],
								args[1],
								args[2],
								args[3],
								args[4],
								args[5],
								"PABLO.log")
		elif (number_of_parameters == 7):
			self.thisptr = new Class_Para_Tree[D2](args[0],
								args[1],
								args[2],
								args[3],
								args[4],
								args[5],
								args[6])
		else:
			print("Wrong number of parameters, dude. Please Retry.")
			# Delete references to the instance, and once they are 
			# all gone, the object is reclaimed
			del self
	
	# ------------------------------Destructor------------------------------
	def __dealloc__(self):
		# delete the pointer to the internal object Class_Para_Tree<2>
		del self.thisptr

	# -------------------------------Properties-----------------------------
	#property trans:
	#	def __get__(self):
	#		return <Class_Map_D2>self.thisptr.trans

	property rank:
		def __get__(self):
			return self.thisptr.rank

	property serial:
		def __get__(self):
			return self.thisptr.serial

	# ---------------------------Get/set methods----------------------------
	#def get_x(self, oct):
	#	return self.trans.map_x(oct.x)

	#def get_y(self, oct):
	#	return self.trans.map_y(oct.y)

	#def get_z(self, oct):
	#	return self.trans.map_z(oct.z)

	#def get_size(self, oct):
	#	return self.trans.map_size(oct.get_size())

	#def get_area(self, oct):
	#	return self.trans.map_size(oct.get_area())

	#def get_volume(self, oct):
	#	return self.trans.map_area(oct.get_volume())

	# ----------------------Local tree get/set methods---------------------
	def get_num_octants(self):
		return self.thisptr.getNumOctants()
	
	def write(self, file_name):
		self.thisptr.write(file_name)

	def write_logical(self, file_name):
		self.thisptr.writeLogical(file_name)

	def compute_connectivity(self):
		self.thisptr.computeConnectivity()

	def adapt_global_refine(self):
		self.thisptr.adaptGlobalRefine()
	
	def update_connectivity(self):
		self.thisptr.updateConnectivity()
