from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int8_t, uintptr_t
import class_octant
cimport mpi4py.MPI as MPI
from mpi4py.mpi_c cimport *

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

		MPI_Comm comm

		# -------------------------map member--------------------------
		# Transoformation map from logical to physical domain
		Class_Map[T] trans
		# -------------------------------------------------------------

		# -------------------------CONSTRUCTORS------------------------
		# Default constructor of Para_Tree. It builds one octant with 
		# node 0 in the Origin (0,0,0) and side of length 1
		Class_Para_Tree(string logfile, MPI_Comm mpi_comm) except +

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
				string logfile,
				MPI_Comm mpi_comm) except +
		
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

		# Get the coordinates of the center of an octant:
		# param[in] oct ---> Pointer to target octant
		# param[out] center ---> coordinates of the center of octant
		void getCenter(Class_Octant[T]* oct, vector[double]& center)

		# Get the coordinates of the center of an octant.
		# param[in] idx ---> local index of target octant
		# return ---> center coordinates of the center of octant
		vector[double] getCenter(uint32_t idx)

		# Get the coordinates of the center of an octant
		# param[in] oct ---> target octant
		# return ---> center coordinates of the center of octant
		vector[double] getCenter(Class_Octant[T]* oct)
	
		# Compute the connectivity of octants and store the coordinates
		# of nodes
		void computeConnectivity()

		void computeGhostsConnectivity()

		# Adapt the octree mesh refining all the octants by one level
		bool adaptGlobalRefine()
		
		# Adapt the octree mesh coarsening all the octants by one level
		bool adaptGlobalCoarse()

		# Update the connectivity of octants
		void updateConnectivity()

		void updateGhostsConnectivity()

		dvector getNodeCoordinates(uint32_t inode)

		#---------------------------------------------------------------
		# ------------------Local tree get/set methods------------------
		# Get the local number of octants:
		# return ---> local number of octants.
		uint32_t getNumOctants() const
		
		void write(string)
		void writeLogical(string)
	
	
		# Write the physical octree mesh in .vtu format with data for 
		# test in a user-defined file; if the connectivity is not stored, 
		# the method temporary computes it.
		# The method doesn't write the ghosts on file:
		# param[in] filename ---> Seriously?....
		void writeTest(string filename, vector[double] data)

		# Set the balancing condition of an octant:
		# param[in] idx ---> local index of target octant.
		# param[in] balance ---> has octant to be 2:1 balanced in 
		#                        adapting procedure?
		void setBalance(uint32_t idx, bool balance)

		void setBalance(Class_Octant[T]* oct, bool balance)

		bool getBound(Class_Octant[T]* octant, uint8_t iface)

		uint64_t getGhostGlobalIdx(uint32_t idx)

		uint64_t getGlobalIdx(uint32_t idx)

		# Get an octant as pointer to the target octant:
		# param[in] idx ---> local index of target octant.
		# return ---> pointer to target octant.
		Class_Octant[T]* getOctant(uint32_t idx)
	
		# Get a ghost octant as pointer to the target octant:
		# param[in] idx ---> local index (in ghosts structure) of 
		# target ghost octant;
		# return ---> pointer to target ghost octant.
		Class_Octant[T]* getGhostOctant(uint32_t idx) 

		# Set the balancing condition of an octant:
		# param[in] oct ---> pointer to target octant.
		# param[in] balance ---> has octant to be 2:1 balanced in 
		#                        adapting procedure?
		void setMarker(Class_Octant[T]* oct, int8_t marker)

		# Set the refinement marker of an octant.
		# param[in] idx ---> local index of target octant.
		# param[in] marker ---> refinement marker of octant 
		#                       (n=n refinement in adapt, 
		#                        -n=n coarsening in adapt, 
		#                        default=0).
		void setMarker(uint32_t idx, int8_t marker)

		uint8_t getMarker(Class_Octant[T]* oct)

		# Get the refinement marker of an octant:
		# param[in] idx ---> local index of target octant;
		# return ---> marker of octant
		uint8_t getMarker(uint32_t idx)

		# Adapt the octree mesh with user setup for markers and 2:1 
		# balancing conditions
		bool adapt()

		# Adapt the octree mesh with user setup for markers and 2:1 
		# balancing conditions.
		# Track the changes in structure octant by a mapper:
		# param[out] mapidx ---> mapper from new octants to old octants;
		#                        mapidx[i] = j -> the i-th octant after 
		#                        adapt was in the j-th position before 
		#                        adapt; if the i-th octant is new after 
		#                        refinement the j-th old octant was the 
		#                        father of the new octant;
		#                        if the i-th octant is new after coarsening 
		#                        the j-th old octant was the first child 
		#                        of the new octant.
		bool adapt(u32vector& mapidx)
	
		# Get if the octant is new after coarsening:
		# param[in] idx ---> local index of target octant;
		# return ---> is octant new?
		bool getIsNewC(uint32_t idx)

		bool getIsGhost(uint32_t idx)

		
		bool getBalance(Class_Octant[T]* oct)
		bool getBalance(uint32_t idx)

		void balance21(bool first)

		# Get the coordinates of the nodes of an octant:
		# param[in] idx ---> local index of target octant;
		# return ---> nodes coordinates of the nodes of octant.
		dvector2D getNodes(uint32_t idx)

		# Get the coordinates of the nodes of an octant:
		# param[in] oct ---> pointer to target octant;
		# return --->nodes coordinates of the nodes of octant.
		dvector2D getNodes(Class_Octant[T]* oct)

		const u32vector2D& getNodes()

		const u32vector2D& getGhostNodes()

		const u32vector2D& getConnectivity()

		const u32vector2D& getGhostConnectivity()

		uint32_t getNumNodes()

		
		# Distribute Load-Balancing the octants of the whole tree over
		# the processes of the job following the Morton order.
		# Until loadBalance is not called for the first time the mesh 
		# is serial
		void loadBalance()

		# Get the local number of ghost octants:
		# return ---> local number of ghost octants.
		uint32_t getNumGhosts() const

		
		# Get the level of an octant:
		# param[in] idx ---> local index of target octant;
		# return ---> level of octant.
		uint8_t getLevel(uint32_t idx)

		# Finds neighbours of octant through iface in vector octants.
		# Returns a vector (empty if iface is a bound face) with the 
		# index of neighbours in their structure (octants or ghosts) and
		# sets isghost[i] = true if the i-th neighbour is ghost in the 
		# local tree:
		# param[in] idx ---> index of current octant;
		# param[in] iface ---> index of face/edge/node passed through 
		# for neighbours finding;
		# param[in] codim ---> codimension of the iface-th entity 1=edge, 
		# 2=node;
		# param[out] neighbours ---> vector of neighbours indices in 
		# octants/ghosts structure;
		# param[out] isghost ---> vector with boolean flag; true if the 
		# respective octant in neighbours is a ghost octant.
		void findNeighbours(uint32_t idx,
				    uint8_t iface,
				    uint8_t codim,
				    u32vector& neighbours,
				    vector[bool]& isghost)

		Class_Octant[T]* getPointOwner(u32vector& point)
		Class_Octant[T]* getPointOwner(dvector& point)
		uint32_t getPointOwnerIdx(dvector& point)
 
# Wrapper Python for class Class_Para_Tree<2>
cdef class  Py_Class_Para_Tree_D2:
	# Pointer to the object Class_Para_Tree<2>
	cdef Class_Para_Tree[D2]* thisptr
	cdef MPI_Comm mpi_comm

	# ------------------------------Constructor-----------------------------
	# different number of arguments can be passed, so different 
	# Class_Para_Tree<2> constructors can be called
	def __cinit__(self, *args):
		number_of_parameters = len(args)
		
		if (number_of_parameters == 0):
			mpi_comm = MPI_COMM_WORLD
			self.thisptr = new Class_Para_Tree[D2]("PABLO.log", mpi_comm)
		elif (number_of_parameters == 1):
			mpi_comm = (<MPI.Comm>args[0]).ob_mpi
			self.thisptr = new Class_Para_Tree[D2]("PABLO.log", mpi_comm)
		elif (number_of_parameters == 2):
			mpi_comm = (<MPI.Comm>args[1]).ob_mpi
			self.thisptr = new Class_Para_Tree[D2](args[0], mpi_comm)
		elif (number_of_parameters == 4):
			mpi_comm = MPI_COMM_WORLD
			self.thisptr = new Class_Para_Tree[D2](args[0],
								args[1],
								args[2],
								args[3],
								"PABLO.log",
								mpi_comm)
		elif (number_of_parameters == 5):
			mpi_comm = (<MPI.Comm>args[4]).ob_mpi
			self.thisptr = new Class_Para_Tree[D2](args[0],
								args[1],
								args[2],
								args[3],
								"PABLO.log",
								mpi_comm)
		elif (number_of_parameters == 6):
			mpi_comm = (<MPI.Comm>args[5]).ob_mpi
			self.thisptr = new Class_Para_Tree[D2](args[0],
								args[1],
								args[2],
								args[3],
								args[4],
								mpi_comm)
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
	property rank:
		def __get__(self):
			return self.thisptr.rank

	property serial:
		def __get__(self):
			return self.thisptr.serial

	property global_num_octants:
		def __get__(self):
			return self.thisptr.global_num_octants

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

	def write_test(self, string file_name, vector[double] data):
		cdef string c_file_name = file_name
		cdef vector[double] c_data = data
		self.thisptr.writeTest(c_file_name, c_data)

	def write_logical(self, file_name):
		self.thisptr.writeLogical(file_name)

	def compute_connectivity(self):
		self.thisptr.computeConnectivity()

	def compute_ghosts_connectivity(self):
		self.thisptr.computeGhostsConnectivity()

	def adapt_global_refine(self):
		result = self.thisptr.adaptGlobalRefine()
		return result
	
	def update_connectivity(self):
		self.thisptr.updateConnectivity()

	def update_ghosts_connectivity(self):
		self.thisptr.updateGhostsConnectivity()

	def adapt_global_coarse(self):
		self.thisptr.adaptGlobalCoarse()

	def get_center_from_index(self, uint32_t idx):
		cdef uint32_t c_idx = idx
		#cdef vector[double] center = self.thisptr.getCenter(c_idx)
		return self.thisptr.getCenter(c_idx)

		#return center

	def get_center(self, uintptr_t idx, from_octant = False):
		cdef vector[double] center
		py_center = []

		if (from_octant):
			center = self.thisptr.getCenter(<Class_Octant[D2]*><void*>idx)
		else:
			center = self.thisptr.getCenter(<uint32_t>idx)

		for i in xrange(0, 3):
			py_center.append(center[i])
	
		return py_center

	def set_balance(self, idx, bool balance):
		cdef Class_Octant[D2]* octant
		
		if (type(idx) is not int):
			# ??? Should not be octant = <Class_Octant[D2]*><void*>idx ???
			oct = <Class_Octant[D2]*><void*>idx
			self.thisptr.setBalance(<Class_Octant[D2]*>octant, balance)
		else:
			self.thisptr.setBalance(<uint32_t>idx, balance)
	
	def set_marker_from_index(self, uint32_t idx, int8_t marker):
		cdef uint32_t c_idx = idx
		cdef int8_t c_marker = marker

		self.thisptr.setMarker(c_idx, c_marker)

	def set_marker(self, uintptr_t octant, int8_t marker, bool from_index = False):
		if (not from_index):
			self.thisptr.setMarker(<Class_Octant[D2]*><void*>octant, <int8_t>marker)

		else:
			self.thisptr.setMarker(<uint32_t>octant, <int8_t>marker)

	def get_marker(self, uintptr_t octant, bool from_index = False):
		if (not from_index):
			return self.thisptr.getMarker(<Class_Octant[D2]*><void*>octant)
		else:
			return self.thisptr.getMarker(<uint32_t>octant)

	#def get_marker(self, octant):
	#	cdef Class_Octant[D2]* oct
	#	oct =  <Class_Octant[D2]*><void*>octant
	#	return self.thisptr.getMarker(oct)


	
	def get_balance(self, idx):
		cdef Class_Octant[D2]* oct
		
		if (type(idx) is not int):
			oct = <Class_Octant[D2]*><void*>idx
			return self.thisptr.getBalance(<Class_Octant[D2]*>oct)
		else:
			return self.thisptr.getBalance(<uint32_t>idx)

	def get_octant(self, uint32_t idx):
		cdef Class_Octant[D2]* octant

		octant = self.thisptr.getOctant(idx)

		py_oct = <uintptr_t>octant
		return py_oct

	def get_bound(self, uintptr_t octant, uint8_t iface):
		return self.thisptr.getBound(<Class_Octant[D2]*><void*>octant,
						iface)

	def get_ghost_global_idx(self, uint32_t idx):
		return self.thisptr.getGhostGlobalIdx(idx)

	def get_global_idx(self, uint32_t idx):
		return self.thisptr.getGlobalIdx(idx)

	def get_point_owner_logical(self, u32vector& point):
		cdef Class_Octant[D2]* octant

		octant = self.thisptr.getPointOwner(<u32vector&>point)

		py_oct = <uintptr_t>octant

		return py_oct

	def get_point_owner_physical(self, dvector& point):
		cdef Class_Octant[D2]* octant

		octant = self.thisptr.getPointOwner(<dvector&>point)

		py_oct = <uintptr_t>octant

		return py_oct

	def get_point_owner_idx(self, dvector& point):
		return self.thisptr.getPointOwnerIdx(<dvector&>point)

	def get_ghost_octant(self, uint32_t idx):
		cdef Class_Octant[D2]* octant
		
		octant = self.thisptr.getGhostOctant(idx)

		py_oct = <uintptr_t>octant

		return py_oct

	def adapt_mapper(self, u32vector& mapidx):
		result = self.thisptr.adapt(<u32vector&>mapidx)

		# the same as return (result, mapidx)
		return result, mapidx

	def adapt(self):
		return self.thisptr.adapt()

	def get_nodes(self):
		return self.thisptr.getNodes()

	def get_ghost_nodes(self):
		return self.thisptr.getGhostNodes()

	def get_connectivity(self):
		return self.thisptr.getConnectivity()

	def get_ghost_connectivity(self):
		return self.thisptr.getGhostConnectivity()

	def get_num_nodes(self):
		return self.thisptr.getNumNodes()

	def get_nodes(self, uintptr_t idx, from_octant = False):
		if (not from_octant):
			return self.thisptr.getNodes(<uint32_t>idx)

		return self.thisptr.getNodes(<Class_Octant[D2]*><void*>idx)

	#def get_nodes(self, uint32_t idx):
	#	cdef uint32_t c_idx = idx
	#	return self.thisptr.getNodes(c_idx)

	def load_balance(self):
		self.thisptr.loadBalance()

	def get_num_ghosts(self):
		return self.thisptr.getNumGhosts()

	def get_level(self, uint32_t idx):
		cdef uint32_t c_idx = idx
		return self.thisptr.getLevel(c_idx)

	def find_neighbours(self, uint32_t idx,
				uint8_t iface,
				uint8_t codim,
				u32vector& neighbours,
				vector[bool]& isghost):
		self.thisptr.findNeighbours(idx, iface, codim, neighbours,
						isghost)

		return (neighbours, isghost)

	def get_is_new_c(self, uint32_t idx):
		return self.thisptr.getIsNewC(idx)

	def get_is_ghost(self, uint32_t idx):
		return self.thisptr.getIsGhost(idx)
	
	def for_test_bubbles(self, int iteration, int nrefperiter, int nocts, int nnodes, int nb, BB):
		cdef int c_nocts = nocts
		cdef int c_nrefperiter = nrefperiter
		cdef int c_nnodes = nnodes
		cdef int c_nb = nb
		cdef int ib
		cdef int i
		cdef int j
		cdef int iref
		cdef vector[double] center
		cdef dvector2D nodes
		cdef double radius
		cdef double radius_sqr
		cdef double xc
		cdef double yc
		cdef double x_2
		cdef double y_2
		cdef double x_1
		cdef double y_1
		
		
		cdef bool inside

		for iref in xrange(0, c_nrefperiter):
			for i in xrange(0, c_nocts):
				inside = False
				nodes = self.thisptr.getNodes(i)
				center = self.thisptr.getCenter(i)
				level = self.thisptr.getLevel(i)
				ib = 0
			 	
				while (not inside and ib < c_nb):
					(xc, yc) = BB[ib].center
					(x_2, y_2) = (center[0]-xc, center[1]-yc)
					radius = BB[ib].radius
					radius_sqr = radius*radius
					for j in xrange(0, c_nnodes):
						(x, y) = (nodes[j][0], nodes[j][1])
						(x_1, y_1) = (x-xc, y-yc)
						if ((((x_1)*(x_1) +
						     (y_1)*(y_1)) <=
						     1.15*(radius_sqr) and
						     ((x_1)*(x_1) + 
						     (y_1)*(y_1)) >=
						     0.85*(radius_sqr)) or
						    (((x_2)*(x_2) +
						     (y_2)*(y_2)) <=
						     1.15*(radius_sqr) and
						     ((x_2)*(x_2) + 
						     (y_2)*(y_2)) >=
						     0.85*(radius_sqr))):
							if (level < 9):
								# Set to refine inside the sphere
								#pabloBB.set_marker(i, 1, from_index = True)
								self.thisptr.setMarker(<uint32_t>i, 1)
							else:
								self.thisptr.setMarker(<uint32_t>i, 0)
							
							inside = True
					ib += 1
				
				if (level > 6 and (not inside)):
					# Set to coarse if the octant has a level higher than 5
					#pabloBB.set_marker(i, -1, from_index = True)
					self.thisptr.setMarker(<uint32_t>i, -1)

			adapt = self.thisptr.adapt()

			# PARALLEL TEST: (Load)Balance the octree over the 
			# processes with communicating the data
			self.thisptr.loadBalance()

			c_nocts = self.thisptr.getNumOctants()

		
		self.thisptr.updateConnectivity()

		self.thisptr.write("PabloBubble_iter" + str(iteration))

		return c_nocts
