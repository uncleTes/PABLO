# ------------------------------------IMPORT------------------------------------
# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
# A guide to analyzing Python performance:
# http://www.huyng.com/posts/python-performance-analysis/
import numbers
import math
import collections
import BaseClass2D
import ExactSolution2D
import numpy
from petsc4py import PETSc 
from mpi4py import MPI
import class_global
import utilities
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
glob = class_global.Py_Class_Global_D2()

# ------------------------------------------------------------------------------
class Laplacian2D(BaseClass2D.BaseClass2D):   
    """Class which evaluates the laplacian onto a 2D grid.
    
    Attributes:
        _comm (MPI.Intracomm) : intracommunicator which identify the
                                process where evaluate the laplacian.
        _octree (class_para_tree.Py_Class_Para_Tree_D2) : PABLO's ParaTree.
        _comm_w (MPI.Intracomm) : global intracommunicator.
        _pen (float or int) : penalization value.
        _over_l (boolean) : flag inndicating if we are in an overlapped or 
                            full immersed case.
        _f_bound (list of lists) : foreground boundaries (boundaries of the
                                   grids over the background ones).
        _b_bound (list of numbers) : background boundaries.
        _proc_g (int) : grid for which the current process is doing all the 
                        work.
        _N_oct (int) : total number of octants in the communicator.
        _n_oct (int) : local number of octants in the process.
        _edge (number) : length of the edge of the grid.
        _grid_l (list) : list of processes working on the current grid.
        _tot_oct (int) : total number of octants in the whole problem.
        _oct_f_g (list) : list of octants for each grid presents into the 
                          problem.
        _h (number): edge's length of the edge of the octants of the grid.
        _rank_w (int) : world communicator rank.
        _rank (int) : local communicator rank.
        _masked_oct_bg_g (int) : number of masked octants on the background 
                                 grid."""

    # --------------------------------------------------------------------------
    def __init__(self, 
                 kwargs = {}):
        """Initialization method for the \"Laplacian2D\" class.

        Arguments:
            kwargs (dictionary) : it must contains the following keys (in 
                                  addition to the ones of \"BaseClass2D\"):
                                  - \"edge\".
                                     
            Raises:
                AssertionError : if \"edge" is not greater than 0.0, then the 
                                 exception is raised and catched, launching an 
                                 \"MPI Abort\", launched also if attributes 
                                 \"f_bound\" or \"b_bound\" are \"None\"."""

        # http://stackoverflow.com/questions/19205916/how-to-call-base-classs-init-method-from-the-child-class
        super(Laplacian2D, self).__init__(kwargs)
        # If some arguments are not presents, function \"setdefault\" will set 
        # them to the default value.
        # Penalization.
        self._pen = kwargs.setdefault("penalization", 
                                      0)
        # Over-lapping.
        self._over_l = kwargs.setdefault("overlapping",
                                         False)
        # \"[[x_anchor, x_anchor + edge, 
        #     y_anchor, y_anchor + edge]...]\" = penalization boundaries (aka
        # foreground boundaries).
        self._f_bound = kwargs.setdefault("foreground boundaries",
                                          None)
        # \"[x_anchor, x_anchor + edge, 
        #    y_anchor, y_anchor + edge]\" = background boundaries.
        self._b_bound = kwargs.setdefault("background boundaries",
                                          None)
        # Checking existence of penalization boundaries and background 
        # boundaries. The construct \"if not\" is useful whether 
        # \"self._f_bound\" or \"self._b_bound\" are None ore with len = 0.
        if ((not self._f_bound) or 
            (not self._b_bound)):
            msg = "\"MPI Abort\" called during initialization "
            extra_msg = " Penalization or bakground boundaries or both are " + \
                        "not initialized. Please check your \"config file\"."
            self.log_msg(msg    , 
                         "error",
                         extra_msg)
	    self._comm_w.Abort(1) 
        # The grid of the current process: process' grid.
        self._proc_g = kwargs["process grid"]
        # Total number of octants into the communicator.
        self._N_oct = self._octree.global_num_octants
        # Local number of octants in the current process of the communicator.
        self._n_oct = self._octree.get_num_octants()
        # Length of the edge of the grid.
        self._edge = kwargs["edge"]
        self._grid_l = kwargs["grid processes"]
        try:
            assert self._edge > 0.0
        except AssertionError:
            msg = "\"MPI Abort\" called during initialization "
            extra_msg = " Attribute \"self._edge\" equal or smaller than 0."
            self.log_msg(msg    ,
                         "error",
                         extra_msg)
	    self._comm_w.Abort(1)
        # Total number of octants presents in the problem.
        self._tot_oct = kwargs["total octants number"]
        # Number of octants for each grid. It is a list.
        self._oct_f_g = kwargs["octants for grids"]
        # Length of the edge of an octree.
        self._h = self._edge / numpy.sqrt(self._N_oct)
        # Getting the rank of the current process inside the world communicator
        # and inside the local one.
        self._rank_w = self._comm_w.Get_rank()
        self._rank = self._comm.Get_rank()
        # Initializing exchanged structures.
        self.init_e_structures()
    # --------------------------------------------------------------------------
   
    # --------------------------------------------------------------------------
    # Returns the center of the face neighbour.
    def neighbour_centers(self           ,
                          _centers       ,
                          _edges_or_nodes,
                          _values):
        """Function which returns the centers of neighbours, depending on 
           for which face we are interested into.
           
           Arguments:
               _centers (tuple or list of tuple) : coordinates of the centers 
                                                   of the current octree.
               _edges_or_nodes (int between 1 and 2 or list) : numbers
                                                               indicating if 
                                                               the neighbour 
                                                               is from edge 
                                                               or node.
               _values (int between 0 and 3 or list) : faces for which we are  
                                                       interested into knowing
                                                       the neighbour's center.
                                            
           Returns:
               a tuple or a list containing the centers evaluated."""

        h = self._h
        centers = _centers
        values = _values
        edges_or_nodes = _edges_or_nodes        
        # Checking if passed argumetns are lists or not. If not, we have to do
        # something.
        try:
            len(centers)
            len(values)
            len(edges_or_nodes)
        # \"TypeError: object of type 'int' has no len()\", so are no lists but
        # single elements.
        except TypeError:
            t_center, t_value, t_e_o_n = centers, values, edges_or_nodes
            centers, values, edges_or_nodes = ([] for i in range(0, 3))
            centers.append(t_center)
            values.append(t_value)
            edges_or_nodes.append(t_e_o_n)

        if ((len(values) != 1) and
            (len(values) != len(centers))):
            msg = "\"MPI Abort\" called " 
            extra_msg = " Different length of \"edges\" (or \"nodes\") and " +\
                        "\"centers\"."
            self.log_msg(msg    ,
                         "error",
                         extra_msg)
            self._comm_w.Abort(1)

	# Evaluated centers.
        eval_centers = []
        # Face or node.
        for i, value in enumerate(values):
            (x_center, y_center) = centers[i]
            if not isinstance(value, numbers.Integral):
                value = int(math.ceil(e_o_n))
            try:
                # Python's comparison chaining idiom.
                assert 0 <= value <= 3
            except AssertionError:
                msg = "\"MPI Abort\" called " 
                extra_msg = " Faces numeration incorrect."
                self.log_msg(msg    ,
                             "error",
                             extra_msg)
                self._comm_w.Abort(1)
            # If no exception is raised, go far...
            else:
                edge = False
                node = False
                if edges_or_nodes[i] == 1:
                    edge = True
                elif edges_or_nodes[i] == 2:
                    node = True
                if ((value % 2) == 0):
		    if (value == 0):
                    	x_center -= h
                        if node:
                            y_center -= h
		    else:
                        if edge:
                    	    y_center -= h
                        if node:
                            x_center -= h
                            y_center += h
                else:
		    if (value == 1):
                    	x_center += h
                        if node:
                            y_center -= h
		    else:
                    	y_center += h
                        if node:
                            x_center += h
                            
                eval_centers.append((x_center, y_center))

        msg = "Evaluated centers of the neighbours"
        if len(centers) == 1:
            msg = "Evaluated center of the neighbour"
            return eval_centers[0]
                
        self.log_msg(msg   ,
                     "info")

        return eval_centers
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Evaluate boundary conditions. 
    def eval_b_c(self   ,
                 centers,
                 faces):
        """Method which evaluate boundary condition on one octree or more,
           depending by the number of the \"center\" passed by.
           
           Arguments:
               centers (tuple or list of tuple) : coordinates of the center/s
                                                  of the octree on the boundary.
               faces (int between 0 and 3 or list of int) : the face of the 
                                                            current octree for 
                                                            which we are
                                                            interested
                                                            into knowing the 
                                                            neighbour's center.
                                                           
           Returns:
               boundary_values (int or list) : the evaluated boundary condition
                                               or a list of them.
               c_neighs (tuple or list of tuples) : the centers where evaluate
                                                    the boundary conditions."""

        edges_or_nodes = []
        just_one_neighbour = False
        for i in xrange(0, len(centers)):
            # Evaluating boundary conditions for edges.
            edges_or_nodes.append(1)
        # Centers neighbours.
        c_neighs = self.neighbour_centers(centers       ,
                                          edges_or_nodes,
                                          faces)
        # \"c_neighs\" is only a tuple, not a list.
        if not isinstance(c_neighs, list):
            just_one_neighbour = True
            c_neigh = c_neighs
            c_neighs = []
            c_neighs.append(c_neigh)
        x_s = [c_neigh[0] for c_neigh in c_neighs] 
        y_s = [c_neigh[1] for c_neigh in c_neighs]

        boundary_values = ExactSolution2D.ExactSolution2D.solution(x_s, 
                                                   		   y_s)

        msg = "Evaluated boundary conditions"
        if just_one_neighbour: 
            msg = "Evaluated boundary condition"
        self.log_msg(msg   ,
                     "info")
        return (boundary_values, c_neighs)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Get octants's ranges for processes, considering also the ones masked by
    # the foreground grids, whom PETSc does not count using 
    # \"getOwnershipRange\".
    def get_ranges(self):
        """Method which evaluate ranges of the octree, for the current process.
                                                           
           Returns:
               the evaluated octants' range."""

        grid = self._proc_g
        n_oct = self._n_oct
        o_ranges = self._b_mat.getOwnershipRange()
        is_background = True
        if grid:
            is_background = False
        # If we are on the background grid, we need to re-evaluate the ranges
        # of octants owned by each process, not simply adding the masked ones
        # (as we will do for the poreground grids), to the values given by PETSc
        # function \"getOwnershipRange\" on the matrix. The idea is to take as
        # start of the range the sum of all the octants owned by the previous
        # processes of the current process, while for the end of the 
        # take the same range of processes plus the current one, obviously 
        # subtracting the value \"1\", because the octants start from \"0\".
        if is_background:
            # Local rank.
            rank_l = self._rank
            # Range's start.
            r_start = 0
            # Range's end.
            r_end = self._s_counts[0] - 1
            # Octants for process.
            octs_f_process = self._s_counts
            for i in xrange(0, rank_l):
                r_start += octs_f_process[i]
                r_end += octs_f_process[i + 1]
            new_ranges = (r_start, r_end)
            
        else:
            # Masked octants
            masked_octs = self._masked_oct_bg_g
            new_ranges = (o_ranges[0] + masked_octs, 
                          o_ranges[1] + masked_octs)

        msg = "Evaluated octants' ranges"
        extra_msg = "with ranges " + str(new_ranges) 
        self.log_msg(msg   ,
                     "info",
                    extra_msg)

        return new_ranges
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Set boundary conditions.
    def set_b_c(self):
        """Method which set boundary conditions for the different grids."""
    
	log_file = self.logger.handlers[0].baseFilename
        penalization = self._pen
        b_bound = self._b_bound
        grid = self._proc_g
        n_oct = self._n_oct
        nfaces = glob.nfaces
        h = self._h
        h2 = h * h
        is_background = True
        o_ranges = self.get_ranges()

        # If we are onto the grid \"0\", we are onto the background grid.
        if grid:
            is_background = False

        b_indices, b_values = ([] for i in range(0, 2))# Boundary indices/values
        b_centers, b_faces = ([] for i in range(0, 2)) # Boundary centers/faces
        for octant in xrange(0, n_oct):
            # Global index of the current local octant \"octant\".
            g_octant = o_ranges[0] + octant
            m_g_octant = self.mask_octant(g_octant)
            # Check if the octant is not penalized.
            if (m_g_octant != -1):
                py_oct = self._octree.get_octant(octant)
                center  = self._octree.get_center(octant)[:2]

                for face in xrange(0, nfaces):
                    # If we have an edge on the boundary.
                    if self._octree.get_bound(py_oct, 
                                              face):
                        b_indices.append(m_g_octant)
                        b_faces.append(face)
                        b_centers.append(center)
            
        (b_values, c_neighs) = self.eval_b_c(b_centers,
                                             b_faces)
        # Converting from numpy array to python list.
	b_values = b_values.tolist()

        # Grids not of the background: equal to number >= 1.
        if grid:
            for i, center in enumerate(c_neighs):
                # Check if foreground grid is inside the background one.
                check = utilities.check_oct_into_square(center     ,
                                            	        b_bound    ,
                                                        h          ,
                                                        0.0        ,
                                          	        self.logger,
                                          	        log_file)
                if check:
                    # Can't use list as dictionary's keys.
                    # http://stackoverflow.com/questions/7257588/why-cant-i-use-a-list-as-a-dict-key-in-python
                    # https://wiki.python.org/moin/DictionaryKeys
                    key = (grid        , # Grid (0 is for the background grid)
                           b_indices[i], # Masked global index of the octant
                           b_faces[i]  , # Boundary face
                           h)            # Edge's length
                    # We store the center of the cell on the boundary.
                    self._edl.update({key : center})
                    # The new corresponding value inside \"b_values\" would be
                    # \"0.0\", because the boundary value is given by the 
                    # coefficients of the bilinear operator in the \"extension\"
                    # matrix.
                    b_values[i] = 0.0
	    
        b_values[:] = [b_value * (-1.0 / h2) for b_value in b_values]
        insert_mode = PETSc.InsertMode.ADD_VALUES
        self._rhs.setValues(b_indices,
                            b_values ,
                            insert_mode)

        self.assembly_petsc_struct("rhs")
        
        msg = "Set boundary conditions"
        extra_msg = "of grid \"" + str(self._proc_g) + "\""
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Creates a layer around the foreground grids, reducing their area.
    def apply_overlap(self,
                      overlap):
        """Method which apply a layer onto the foreground grids to reduce the
           \"penalized\" area and so do less iterations to converge.
           
           Arguments:
               overlap (number) : size of the layer to apply.

           Returns:
               p_bound (list of lists) : list of the new \"penalized\" grids."""

        f_bound = self._f_bound
        # \"p_bound\" is a new vector of vectors which contains the 
        # effective boundaries to check for penalization using an 
        # overlapping region for the grids, used into D.D.
        # Penalization boundaries.
        p_bound = []
        # Reducing penalization boundaries using the overlap.
        for boundary in f_bound:
            # Temporary boundary
            t_bound = []
            for index, point in enumerate(boundary):
                t_bound.append(point + overlap) if (index % 2) == 0 else \
                t_bound.append(point - overlap)
            p_bound.append(t_bound)
        
        msg = "Applied overlap"
        extra_msg = "with overlap " + str(overlap)
        self.log_msg(msg   ,
                     "info",
                     extra_msg)

        return p_bound
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Apply mask to the octants, not bein considering the octants of the 
    # background grid covered by the ones of the foreground meshes.
    def mask_octant(self, 
                    g_octant):
        """Method which evaluate the global index of the octant, considering
           the masked ones, dued to the grids' overposition.
           
           Arguments:
               g_octant (int) : global index of the octant.

           Returns:
               m_g_octant (int) : masked global index of the octant."""

        grid = self._proc_g
        if grid:
            # Masked global octant.
            m_g_octant = g_octant - self._masked_oct_bg_g
        else:
            m_g_octant = self._ngn[g_octant]
        
        return m_g_octant
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Creates masking system for the octants of the background grid covered by 
    # the foreground meshes, and also determines the number of non zero elements
    # to allocate for each row in the system's matrix. Be aware of that, for the
    # moment, this last count is exact for the background grid but for the 
    # foreground ones it consider the worst case (for the two levels gap).
    def create_mask(self, 
                    o_n_oct = 0):
        """Method which creates the new octants' numerations and initialize non
           zero elements' number for row in the matrix of the system.
           
           Arguments:
               o_n_oct (int) : number of octants overlapped.

           Returns:
               (d_nnz, o_nnz) (tuple) : two lists containting the diagonal
                                        block's and non diagonal block's number
                                        of non zero elements."""

        log_file = self.logger.handlers[0].baseFilename
        logger = self.logger
        penalization = self._pen
        f_bound = self._f_bound
        grid = self._proc_g
        n_oct = self._n_oct
        octree = self._octree
        nfaces = glob.nfaces
        h = self._h
        comm_l = self._comm
        rank_l = comm_l.Get_rank()
        h2 = h * h
        is_background = False
        overlap = o_n_oct * h
        p_bound = []
        if not grid:
            is_background = True
            p_bound = self.apply_overlap(overlap)

        # Lists containing number of non zero elements for diagonal and non
        # diagonal part of the coefficients matrix, for row. 
        d_nnz, o_nnz = ([] for i in range(0, 2))
        new_oct_count = 0
        for octant in xrange(0, n_oct):
            d_count, o_count = 0, 0
            neighs, ghosts = ([] for i in range(0, 2))
            g_octant = octree.get_global_idx(octant)
            py_oct = octree.get_octant(octant)
            center  = octree.get_center(octant)[:2]
            # Check to know if an octant is penalized.
            is_penalized = False
            # Background grid.
            if is_background:
                is_penalized = utilities.check_oct_into_squares(center ,
                                                  	        p_bound,
                                                                h      ,
                                                                0.0    ,
                                                  	        logger ,
                                                  	        log_file)
            if is_penalized:
                self._nln[octant] = -1
                key = (grid    ,
                       g_octant,
                       h)
                # If the octant is covered by the foreground grids, we need
                # to store info of the stencil it belongs to to push on the
                # relative rows of the matrix, the right indices of the octants
                # of the foreground grid owning the penalized one.
                stencil = []
                stencil.append((g_octant, center))
                self._edl.update({key : stencil})
            else:
                self._nln[octant] = new_oct_count
                new_oct_count += 1
                d_count += 1
            # Not add to foreground count.
            not_add_fg_count = False
            for face in xrange(0, nfaces):
                # Check to know if a neighbour of an octant is penalized.
                is_n_penalized = False
                # Not boundary face.
                if not self._octree.get_bound(py_oct, 
                                              face):
                    (neighs, ghosts) = octree.find_neighbours(octant, 
                                                              face  , 
                                                              1     , 
                                                              neighs, 
                                                              ghosts)
                    if not ghosts[0]:
                        index = octree.get_global_idx(neighs[0])
                        n_center = self._octree.get_center(neighs[0])[:2]
                    else:
                        index = self._octree.get_ghost_global_idx(neighs[0])
                        py_ghost_oct = self._octree.get_ghost_octant(neighs[0])
                        n_center = self._octree.get_center(py_ghost_oct, 
                                                         True)[:2]

                    if is_background:
                        # Is neighbour penalized.
                        is_n_penalized = utilities.check_oct_into_squares(n_center,
                                                      	                  p_bound ,
                                                                          h       ,
                                                                          0.0     ,
                                                  	                  logger  ,
                                                  	                  log_file)
                    if not is_penalized:
                        if is_n_penalized:
                            # Being the neighbour penalized, it means that it 
                            # will be substituted by 4 octant being part of 
                            # the foreground grids, so being on the non diagonal
                            # part of the grid.
                            o_count += 4
                        else:
                            if ghosts[0]:
                                o_count += 1
                            else:
                                d_count += 1
                    else:
                        if not is_n_penalized:
                            self._edl.get(key).append((index, n_center))
                else:
                    # Adding elements for the octants of the background to use
                    # to interpolate stencil values for boundary conditions of 
                    # the octants of the foreground grid. This is the worst
                    # scenario.
                    if not is_background:
                        if not not_add_fg_count:
                            # Worst cases with two levels of difference.
                            o_count += 12 # It could be 16 if the neighbours of
                                          # the background grid would be covered
                                          # by other foreground grids.
                            d_count += 8
                            # Add to the counters only for one face, because in
                            # case of two faces on the boundary, elements
                            # considered will be the same, They will change
                            # coefficients' values, not coefficients' number.
                            not_add_fg_count = True
            if not is_penalized:
                d_nnz.append(d_count)
                o_nnz.append(o_count)
                self._centers_not_penalized.append(center)

        self.spread_new_background_numeration(is_background)

        msg = "Created mask"
        self.log_msg(msg   ,
                     "info")

        return (d_nnz, o_nnz)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # The new masked global numeration for the octants of the background grid 
    # has to be spread to the other meshes.
    def spread_new_background_numeration(self,
                                         is_background):
        n_oct = self._n_oct
        comm_l = self._comm
        comm_w = self._comm_w
        rank_l = comm_l.Get_rank()
        tot_not_masked_oct = numpy.sum(self._nln != -1)
        tot_masked_oct = n_oct - tot_not_masked_oct
        # Elements not penalized for grid.
        el_n_p_for_grid = numpy.empty(comm_l.size,
                                      dtype = int)
        comm_l.Allgather(tot_not_masked_oct, 
                         el_n_p_for_grid)
        # Counting the number of octants not penalized owned by all the previous
        # grids to know the offset to add at the global numeration of the octree
        # because although it is global, it is global at the inside of each
        # octant, not in the totality of the grids.
        oct_offset = 0
        for i in xrange(0, len(el_n_p_for_grid)):
            if i < rank_l:
                oct_offset += el_n_p_for_grid[i]
        # Adding the offset at the new local numeration.
        self._nln[self._nln >= 0] += oct_offset
        
        if is_background:
            # Send counts. How many element have to be sent by each process.
            self._s_counts = []
            self._s_counts.extend(comm_l.allgather(self._nln.size))
            displs = [0] * len(self._s_counts)
            offset = 0
            for i in range(1, len(self._s_counts)):
                offset += self._s_counts[i-1]
                displs[i] = offset
            comm_l.Gatherv(self._nln                                       ,
                           [self._ngn, self._s_counts, displs, MPI.INT64_T],
                           root = 0)
        comm_w.Barrier()
        # Broadcasting the vector containing the new global numeration of the
        # background grid \"self._ngn\" to all processes of the world 
        # communicator.
        comm_w.Bcast(self._ngn,
                     root = 0)

        msg = "Spread new global background masked numeration"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Initialize diagonal matrices of the block matrix.
    def init_mat(self,
                 (e_d_nnz, e_o_nnz),
                 o_n_oct = 0):
        """Method which initialize the diagonal parts of the monolithic matrix 
           of the system.
           
           Arguments:
               o_n_oct (int) : number of octants overlapped."""

	log_file = self.logger.handlers[0].baseFilename
        logger = self.logger
        penalization = self._pen
        f_bound = self._f_bound
        grid = self._proc_g
        comm_w = self._comm_w
        rank_w = self._rank_w
        octree = self._octree
        tot_oct = self._tot_oct
        is_background = True
        # Range deplacement.
        h = self._h
        h2 = h * h
        overlap = o_n_oct * h
        p_bound = []
        oct_offset = 0
        if grid:
            for i in range(0, grid):
                oct_offset += self._oct_f_g[i]
            is_background = False
        else:
            p_bound = self.apply_overlap(overlap)

        (d_nnz, o_nnz) = (e_d_nnz, e_o_nnz)
        n_oct = self._n_oct
        nfaces = glob.nfaces
        sizes = self.find_sizes()
        self._b_mat = PETSc.Mat().createAIJ(size = (sizes, sizes),
                                            nnz = (d_nnz, o_nnz) ,
                                            comm = comm_w)
        # If an element is being allocated in a place not preallocate, then 
        # the program will stop.
        self._b_mat.setOption(self._b_mat.Option.NEW_NONZERO_ALLOCATION_ERR, 
                              True)
        
        o_ranges = self.get_ranges()
        for octant in xrange(0, n_oct):
            indices, values = ([] for i in range(0, 2)) # Indices/values
            neighs, ghosts = ([] for i in range(0, 2))
            g_octant = o_ranges[0] + octant
            # Masked global octant.
            m_g_octant = self.mask_octant(g_octant)
            py_oct = octree.get_octant(octant)
            center = octree.get_center(octant)[:2]
            # Check to know if an octant on the background is penalized.
            is_penalized = False
            # Background grid.
            if is_background:
                is_penalized = utilities.check_oct_into_squares(center ,
                                                  	        p_bound,
                                                                h      ,
                                                                0.0    ,
                                                  	        logger ,
                                                  	        log_file)
            if not is_penalized:
                indices.append(m_g_octant)
                values.append((-4.0 / h2))

                for face in xrange(0, nfaces):
                    is_n_penalized = False
                    if not octree.get_bound(py_oct, 
                                            face):
                        (neighs, ghosts) = octree.find_neighbours(octant, 
                                                                  face  , 
                                                                  1     , 
                                                                  neighs, 
                                                                  ghosts)

                        if not ghosts[0]:
                            n_center = octree.get_center(neighs[0])[:2]
                            index = neighs[0] + o_ranges[0]
                            # Masked index.
                            m_index = self.mask_octant(index)
                        else:
                            index = octree.get_ghost_global_idx(neighs[0])
                            py_ghost_oct = octree.get_ghost_octant(neighs[0])
                            n_center = octree.get_center(py_ghost_oct, 
                                                         True)[:2]
                            m_index = self.mask_octant(index)
                            m_index = m_index + oct_offset
                        
                        if is_background:
                            # Is neighbour penalized.
                            is_n_penalized = utilities.check_oct_into_squares(n_center,
                                                      	                      p_bound ,
                                                                              h       ,
                                                                              0.0     ,
                                                      	                      logger  ,
                                                      	                      log_file)
                        if not is_n_penalized:
                            indices.append(m_index)
                            values.append(1.0 / h2)

                self._b_mat.setValues(m_g_octant, # Row
                                      indices   , # Columns
                                      values)     # Values to be inserted
    
        # We have inserted argument \"assebly\" equal to 
        # \"PETSc.Mat.AssemblyType.FLUSH_ASSEMBLY\" because the final assembly
        # will be done after inserting the prolongation and restriction blocks.
        self.assembly_petsc_struct("matrix",
                                   PETSc.Mat.AssemblyType.FLUSH_ASSEMBLY)
        
        msg = "Initialized diagonal parts of the monolithic  matrix"
        extra_msg = "with sizes \"" + str(self._b_mat.getSizes()) + \
                    "\" and type \"" + str(self._b_mat.getType()) + "\""
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Assembly PETSc's structures, like \"self._b_mat\" or \"self._rhs\".
    def assembly_petsc_struct(self       ,
                              struct_type,
                              assembly_type = None):
        """Function which inglobe \"assemblyBegin()\" and \"assemblyEnd()\" 
           \"PETSc\" function to avoid problem of calling other functions
           between them.
        
           Arguments:
                struct_type (string) : what problem's structure we want to
                                       assembly.
                assembly_type (PETSc.Mat.AssemblyType) : type of assembly; it
                                                         can be \"FINAL\" or
                                                         \"FLUSH\"."""

        if struct_type == "matrix":
            self._b_mat.assemblyBegin(assembly = assembly_type)
            self._b_mat.assemblyEnd(assembly = assembly_type)
        elif struct_type == "rhs":
            self._rhs.assemblyBegin()
            self._rhs.assemblyEnd()
        else:
            msg = "\"MPI Abort\" called during initialization "
            extra_msg = " PETSc struct " + str(struct_type) +\
                        "not recognized."
            self.log_msg(msg    , 
                         "error",
                         extra_msg)
	    self._comm_w.Abort(1) 
        
        msg = "Assembled PETSc structure " + str(struct_type)
        extra_msg = "with assembly type " + str(assembly_type)
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Finds local and global sizes used to initialize \"PETSc \" structures.
    def find_sizes(self):
	"""Method which find right sizes for \"PETSc\" structures.
 
	   Returns:
	        sizes (tuple) : sizes for \"PETSc\" data structure."""

        grid = self._proc_g
        n_oct = self._n_oct
        rank_l = self._rank
        not_masked_oct_bg_g = numpy.size(self._ngn[self._ngn != -1])
        self._masked_oct_bg_g = self._ngn.size - not_masked_oct_bg_g
        tot_oct = self._tot_oct - self._masked_oct_bg_g 
        if not grid:
            # Not masked local octant background grid.
            not_masked_l_oct_bg_g = numpy.size(self._nln[self._nln != -1])
        sizes = (n_oct if grid else not_masked_l_oct_bg_g, 
                 tot_oct)

        msg = "Found sizes for PETSc structure"
        extra_msg = "with sizes " + str(sizes)
        self.log_msg(msg   ,
                     "info",
                     extra_msg)

        return sizes
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Interpolating the solution is necessary to be able to use the \"vtm\"
    # typology of files. For the moment, the value interpolated is substituted
    # by the fixed value \"0.0\".
    def interpolate_solution(self):
        """Function which creates a \"new\" solution array, pushing in the
           octants covered by foreground meshes the values interpolated from the
           neighbours around them.

           Returns:
                inter_sol (PETSc.Vec) : the interpolated solution."""

        grid = self._proc_g
        octree = self._octree
        o_ranges = self.get_ranges()
        n_oct = self._n_oct
        tot_oct = self._tot_oct
        # Upper bound octree's id contained.
        up_id_octree = o_ranges[0] + n_oct
        # Octree's ids contained.
        ids_octree_contained = range(o_ranges[0], 
                                     up_id_octree)
        # Interpolated solution.
        inter_sol = self.init_array("interpolated solution",
                                    False)

        for i in xrange(0, tot_oct):
            if i in ids_octree_contained:
                sol_index = self.mask_octant(i)
                if (sol_index != -1):
                    sol_value = self._sol.getValue(sol_index)
                    inter_sol.setValue(i, sol_value)

        inter_sol.assemblyBegin()
        inter_sol.assemblyEnd()
        
        msg = "Interpolated solution"
        self.log_msg(msg   ,
                     "info")
    
        return inter_sol
    # --------------------------------------------------------------------------
   
    # --------------------------------------------------------------------------
    # Initializes a \"PTESc\" array, being made of zeros or values passed by.
    def init_array(self             ,
                   # Array name.
                   a_name = ""      ,
                   petsc_size = True,
                   array = None):
	"""Method which initializes an array or with zeros or with a 
	   \"numpy.ndarray\" passed as parameter.

	   Arguments:
		a_name (string) : name of the array to initialize, being written
				  into the log. Default value is \"\".
                petsc_size (boolean): if \"True\", it impose the use of the 
                                      sizes used by \"PETSc\". Otherwise, the
                                      local and global numbers of octants from
                                      the octree are used. 
		array (numpy.ndarray) : possible array to use to initialize the
					returned array. Default value is 
					\"None\".

	   Returns:
		a PETSc array."""
	
        if not petsc_size:
            n_oct = self._n_oct
            tot_oct = self._tot_oct
            sizes = (n_oct, tot_oct)
        else: 
            sizes = self.find_sizes()
        # Temporary array.
        t_array = PETSc.Vec().createMPI(size = sizes,
                                        comm = self._comm_w)
        t_array.setUp()

        if array is None:
            t_array.set(0)
        else:
            try:
                assert isinstance(array, numpy.ndarray)
                # Temporary PETSc vector.
                t_petsc = PETSc.Vec().createWithArray(array       ,
                                                      size = sizes,
                                                      comm = self._comm_w)
                t_petsc.copy(t_array)
            except AssertionError:
                msg = "\"MPI Abort\" called during array's initialization"
                extra_msg = "Parameter \"array\" not an instance of " + \
                            "\"numpy.ndarray\"."
                self.log_msg(msg    ,
                             "error",
                             extra_msg)
                self._comm_w.Abort(1)
        msg = "Initialized \"" + str(a_name) + "\""
        self.log_msg(msg,
                     "info")
        return t_array
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Initializes \"rhs\".
    def init_rhs(self, 
                 numpy_array):
	"""Method which intializes the right hand side.
            
           Arguments:
                numpy_array (numpy.array) : array to use to initialize with it
                                            the \"rhs\"."""

        self._rhs = self.init_array("right hand side",
                                    True             ,
                                    numpy_array)
        
        msg = "Initialized \"rhs\""
        self.log_msg(msg,
                     "info")
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Initializes \"sol\".
    def init_sol(self):
        """Method which initializes the solution."""

        self._sol = self.init_array("solution",
                                    True)
        
        msg = "Initialized \"rhs\""
        self.log_msg(msg,
                     "info")
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Solving...
    def solve(self):
        """Method which solves the system."""
        # Creating a "KSP" object.
        ksp = PETSc.KSP()
        pc = PETSc.PC()
        ksp.create(self._comm_w)
        ksp.setOperators(self._b_mat,
                         self._b_mat)

        pc = ksp.getPC()
        # Setting tolerances.
        tol = 1.e-13
        ksp.setTolerances(rtol = tol            , 
                          atol = tol            , 
                          divtol = PETSc.DEFAULT, # Let's PETSc use DEAFULT
                          max_it = PETSc.DEFAULT) # Let's PETSc use DEAFULT
        ksp.setFromOptions()
        pc.setFromOptions()
        # Solve the system.
        ksp.solve(self._rhs, 
                  self._sol)
        # How many iterations are done.
        it_number = ksp.getIterationNumber()

        msg = "Evaluated solution"
        extra_msg = "Using \"" + str(it_number) + "\" iterations."
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Initialize exchanged structures.	
    def init_e_structures(self):
	"""Method which initializes structures used to exchange data between
	   different grids."""

        n_oct = self._n_oct
        N_oct_bg_g = self._oct_f_g[0]
        # The \"self._edl\" will contains local data to be exchanged between
	# grids of different levels.
	# Exchanged data local.
        self._edl = {} 
        # The \"self._edg\" will contains the excahnged data between grids of
	# different levels.
	# Exchanged data global.
        self._edg = []

        self._eml = {}
        self._emg = [] 
        
        self._nln = numpy.empty(n_oct,
                                dtype = numpy.int64)
        self._ngn = numpy.empty(N_oct_bg_g,
                                dtype = numpy.int64)
        self._mdl_f = {}
        self._mdl_b = {}
        self._mdg_f = {}
        self._mdg_b = []

        self._centers_not_penalized = []

        msg = "Initialized exchanged structures"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Creating the restriction and prolongation blocks inside the monolithic 
    # matrix of the system.
    def update_values(self, 
                      intercomm_dictionary = {}):
        """Method wich update the system matrix.

           Arguments:
                intercomm_dictionary (python dict) : contains the
                                                     intercommunicators."""

        log_file = self.logger.handlers[0].baseFilename
        grid = self._proc_g
        n_oct = self._n_oct
        comm_w = self._comm_w
        comm_l = self._comm
        rank_w = self._rank_w
        rank_l = self._rank
        is_background = True
        if grid:
            is_background = False
        o_ranges = self.get_ranges()
        # Upper bound octree's id contained.
        up_id_octree = o_ranges[0] + n_oct
        # Octree's ids contained.
        ids_octree_contained = range(o_ranges[0], 
                                     up_id_octree)
        # Calling \"allgather\" to obtain data from the corresponding grid,
        # onto the intercommunicators created, not the intracommunicators.
        # http://www.mcs.anl.gov/research/projects/mpi/mpi-standard/mpi-report-1.1/node114.htm#Node117
        # http://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/
        # http://www.linux-mag.com/id/1412/
        for key, intercomm in intercomm_dictionary.items():
            # Extending a list with the lists obtained by the other processes
            # of the corresponding intercommunicator.
            self._edg.extend(intercomm.allgather(self._edl))

        if not is_background:
            self.update_fg_grids(o_ranges,
                                 ids_octree_contained)

        comm_w.Barrier()

        for key, intercomm in intercomm_dictionary.items():
            self._mdg_b.extend(intercomm.allgather(self._mdg_f))

        if is_background:
            self.update_bg_grids(o_ranges,
                                 ids_octree_contained)
        
        comm_w.Barrier()

        self.assembly_petsc_struct("matrix",
                                   PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)

        self.assembly_petsc_struct("rhs")
        
        msg = "Updated monolithic matrix"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Sub_method of \"update_values\".
    def update_fg_grids(self    ,
                        o_ranges,
                        ids_octree_contained):
        """Method which update the non diagonal blocks relative to the 
           foreground grids.

           Arguments:
                o_ranges (tuple) : the initial and final octants managed by the
                                   current process.
                ids_octree_contained (list) : list of the indices of the octants
                                              contained in the current process."""

        octree = self._octree
        comm_l = self._comm
        # \"self._edg\" will be a list of same structures of data,
        # after the \"allgather\" call; these structures are dictionaries.
        for index, dictionary in enumerate(self._edg):
            for key, stencil in dictionary.items():
                if key[0] == 0:
                    (x_center, y_center) = stencil[0][1]
                    local_idx = octree.get_point_owner_idx((x_center,
                                                            y_center))
                    h2 = key[2] * key[2]
                    global_idx = local_idx + o_ranges[0]

                    if global_idx in ids_octree_contained:
                        center_cell_container = octree.get_center(local_idx)[:2]
                        location = utilities.points_location((x_center,
                                                              y_center),
                                                             center_cell_container)
                        neigh_centers, neigh_indices = ([] for i in range(0, 2)) 
                        (neigh_centers, 
                         neigh_indices)  = self.find_right_neighbours(location ,
                                                                      local_idx,
                                                                      o_ranges[0])
                        bil_coeffs = utilities.bil_coeffs((x_center, 
                                                           y_center),
                                                          neigh_centers)

                        self._mdl_f.update({(key[1], (x_center, y_center)) : 
                                            [neigh_centers, 
                                             neigh_indices, 
                                             bil_coeffs]})

                        bil_coeffs = [coeff * (1.0 / h2) for coeff in bil_coeffs]

                        row_indices = [octant[0] for octant in stencil[1:]]

                        self.apply_rest_prol_ops(row_indices  ,
                                                 neigh_indices,
                                                 bil_coeffs   ,
                                                 neigh_centers)
        self._mdg_f = comm_l.gather(self._mdl_f, 
                                    root = 0)
        
        msg = "Updated prolongation blocks"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Sub_method of \"update_values\".
    def update_bg_grids(self    ,
                        o_ranges,
                        ids_octree_contained):
        """Method which update the non diagonal blocks relative to the 
           backgrounds grids.

           Arguments:
                o_ranges (tuple) : the initial and final octants managed by the
                                   current process.
                ids_octree_contained (list) : list of the indices of the octants
                                              contained in the current process."""

        log_file = self.logger.handlers[0].baseFilename
        octree = self._octree
        comm_l = self._comm
        b_bound = self._b_bound
        for index, dictionary in enumerate(self._edg):
            for key, item in dictionary.items():
                (x_center, y_center) = item
		h2 = key[3] * key[3]
                local_idx = octree.get_point_owner_idx((x_center,
                                                        y_center))
    
                global_idx = local_idx + o_ranges[0]

                if global_idx in ids_octree_contained:
                    center_cell_container = octree.get_center(local_idx)[:2]
                    location = utilities.points_location((x_center,
                                                          y_center),
                                                          center_cell_container)
                    neigh_centers, neigh_indices = ([] for i in range(0, 2)) 
                    # New neighbour indices, new bilinear coefficients, 
                    # new centers.
                    n_n_i, n_b_c, n_c = ([] for i in range(0, 3)) 
                    (neigh_centers, neigh_indices)  = self.find_right_neighbours(location   ,
                                                                                 local_idx  ,
                                                                                 o_ranges[0],
                                                                                 True)
                    bil_coeffs = utilities.bil_coeffs((x_center, 
                                                       y_center),
                                                      neigh_centers)
                    # Substituting bilinear coefficients for \"penalized\" 
                    # octants with the coefficients of the foreground grid
                    # owning the penalized one.
                    for i, index in enumerate(neigh_indices):
                        if not isinstance(index, basestring):
                            if (self._ngn[index] == -1):
                                got_m_values = False
                                for j, listed in enumerate(self._mdg_b):
                                    if not not listed: 
                                        for k, dictionary in enumerate(listed):
                                            m_values = dictionary.get((index, 
                                                                       (neigh_centers[i][0],
                                                                        neigh_centers[i][1])))
                                            if m_values is not None:
                                                n_n_i.extend(m_values[1])
                                                n_b_c.extend([(m_value * bil_coeffs[i]) for m_value in m_values[2]])
                                                n_c.extend(m_values[0])
                                                got_m_values = True
                                                break
                                    if got_m_values:
                                        break
                            else:
                                masked_index = self._ngn[index]
                                n_n_i.append(masked_index)
                                n_b_c.append(bil_coeffs[i])
                                n_c.append(neigh_centers[i])
                        else:
                            n_n_i.append(index)
                            n_b_c.append(bil_coeffs[i])
                            n_c.append(neigh_centers[i])
                            
                    n_b_c= [coeff * (1.0 / h2) for coeff in n_b_c]
                    self.apply_rest_prol_ops(key[1],
                                             n_n_i ,
                                             n_b_c ,
                                             n_c)

        msg = "Updated restriction blocks"
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Returns the right neighbours for an octant, being them of edges or nodes.
    def find_right_neighbours(self          , 
                              location      , 
                              current_octant,
                              start_octant  ,
                              is_background = False):
        """Method which compute the right 4 neighbours for the octant 
           \"current_octant\", considering first the label \"location\" to
           indicate in what directions go to choose the neighborhood.

           Arguments:
                location (string) : indicates what cardinal quadrant choice
                                    (\"nordovest\", \"nordest\", \"sudovest\",
                                     \"sudest\").
                current_octant (int) : local index of the current octant.
                start_octant (int) : global index of the first contained octant
                                     in the process.
                is_background (bool) : indicates if we are or not on the 
                                       background grid. On this choice depends
                                       how the indices of the neighbours will
                                       be evaluatd.

           Returns:
                (centers, indices) (tuple of lists) : tuple containing the lists
                                                      of centers and indices of
                                                      the neighbours."""

        py_oct = self._octree.get_octant(current_octant)
        ordered_points = {}
        centers = []
        indices = []
        grid = self._proc_g
        # Ghosts' deplacement.
        g_d = 0
        if grid:
            for i in range(0, grid):
                g_d = g_d + self._oct_f_g[i]

        if location == "nordovest":
            # Adding 1) the number of node, 2) (codim, number of face/node).
            ordered_points.update({0 : (1, 0)})
            ordered_points.update({1 : None})
            ordered_points.update({2 : (2, 2)})
            ordered_points.update({3 : (1, 3)})
        elif location == "nordest":
            ordered_points.update({0 : None})
            ordered_points.update({1 : (1, 1)})
            ordered_points.update({2 : (1, 3)})
            ordered_points.update({3 : (2, 3)})
        elif location == "sudovest":
            ordered_points.update({0 : (2, 0)})
            ordered_points.update({1 : (1, 2)})
            ordered_points.update({2 : (1, 0)})
            ordered_points.update({3 : None})
        elif location == "sudest":
            ordered_points.update({0 : (1, 2)})
            ordered_points.update({1 : (2, 1)})
            ordered_points.update({2 : None})
            ordered_points.update({3 : (1, 1)})
        # Using \"sorted\" to be sure that values of the dict 
	# \"ordered_points\" are ordered by keys.
        for q_point in sorted(ordered_points.keys()):
            edge_or_node = ordered_points[q_point]
            if edge_or_node is None:
                centers.append(self._octree.get_center(current_octant)[:2])
                index = current_octant
                m_index = self.mask_octant(index + start_octant)
                if is_background:
                    m_index = index + start_octant
                indices.append(m_index)
            else:
                neighs, ghosts = ([] for i in range(0, 2))
                (neighs, 
	         ghosts) = self._octree.find_neighbours(current_octant ,
                                                        edge_or_node[1],
                                                        edge_or_node[0],
                                                        neighs         ,
                                                        ghosts)
                # Check if it is really a neighbour of edge or node. If not,
                # it means that we are near the boundary and so...
                if len(neighs) is not 0:
                    # Neighbour is into the same process, so is local.
                    if not ghosts[0]:
                        cell_center = self._octree.get_center(neighs[0])[:2]
                        centers.append(cell_center)
                        index = neighs[0]
                        m_index = self.mask_octant(index + start_octant)
                        if is_background:
                            m_index = index + start_octant
                        indices.append(m_index)
                    else:
                        # In this case, the quas(/oc)tree is no more local
                        # into the current process, so we have to find it
                        # globally.
                        index = self._octree.get_ghost_global_idx(neighs[0])
                        # \".index\" give us the index of 
                        # \"self._global_ghosts\" that contains the index
                        # of the global ghost quad(/oc)tree previously
                        # found and stored in \"index\".
                        py_ghost_oct = self._octree.get_ghost_octant(neighs[0])
                        cell_center = self._octree.get_center(py_ghost_oct, 
                                                              True)[:2]
                        centers.append(cell_center)
                        m_index = self.mask_octant(index)
                        if is_background:
                            m_index = index
                        indices.append(m_index + g_d)
                # ...we need to evaluate boundary values.
                else:
                    border_center = self._octree.get_center(current_octant)[:2]
                    center = self.neighbour_centers(border_center  ,
                                                    edge_or_node[0],
                                                    edge_or_node[1])

                    centers.append(center)
                    indices.append("outside_bg")

        return (centers, indices)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Apply restriction/prolongation operators.
    def apply_rest_prol_ops(self         ,
                            row_indices  ,
                            col_indices  ,
                            col_values   ,
                            centers):
        """Method which applies the right coefficients at the right neighbours
           in the prolongaion and restriction blocks.

           Arguments: 
                row_indices (list) : indices of the rows where to apply the
                                     coefficients.
                col_indices (list) : indices of the columns where to apply the
                                     coefficients.
                col_values (list) : elements to insert into \"col_indices\"."""

        grid = self._proc_g
        is_background = True
        if grid:
            is_background = False
        insert_mode = PETSc.InsertMode.ADD_VALUES
        n_rows = 1 if is_background else len(row_indices)
        to_rhs = []
        # Exact solutions.
        e_sols = []

        for i, index in enumerate(col_indices):
            # If the neighbour is outside the background boundary, the exact
            # solution is evaluated.
            if index == "outside_bg":
                to_rhs.append(i)
                e_sol = ExactSolution2D.ExactSolution2D.solution(centers[i][0],
                                                                 centers[i][1])
                e_sols.append(e_sol)

        for i in range(0, n_rows):
            row_index = row_indices if is_background else row_indices[i]
            co_indices = col_indices
            co_values = col_values
            if not is_background:
                row_index = self._ngn[row_index]
            # If \"not_rhs\" is not empty.
            if not not to_rhs:
                bil_coeffs = [col_values[j] for j in to_rhs]
                for i in range(0, len(to_rhs)):
                    self._rhs.setValues(row_index                       ,
                                        (-1 * bil_coeffs[i] * e_sols[i]),
                                        insert_mode)
                
                co_indices = [col_indices[j] for j in 
                               range(0, len(col_indices)) if j not in to_rhs]
                co_values = [col_values[j] for j in 
                              range(0, len(col_values)) if j not in to_rhs]
                
            self._b_mat.setValues(row_index  ,
                                  co_indices ,
                                  co_values  ,
                                  insert_mode)
        
        msg = "Applied prolongation and restriction operators."
        self.log_msg(msg   ,
                     "info")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def evaluate_norms(self, 
                       exact_solution,
                       solution):
        """Function which evals the infinite and L2 norms of the error.

           Arguments:
                exact_solution (numpy.array)
                solution (numpy.array) : computed solution.

           Returns:
                (norm_inf, norm_L2) (tuple of int): evaluated norms."""

        h = self._h
        octant_area = (self._h * self._h)
        numpy_difference = numpy.subtract(exact_solution,
                                          solution)
        norm_inf = numpy.linalg.norm(numpy_difference,
                                     # Type of norm we want to evaluate.
                                     numpy.inf)
        norm_L2 = numpy.linalg.norm(numpy_difference,
                                    2) * h

        msg = "Evaluated norms"
        extra_msg = "with (norm_inf, norm_L2) = " + str((norm_inf, norm_L2))
        self.log_msg(msg   ,
                     "info",
                     extra_msg)

        return (norm_inf, norm_L2)
    # --------------------------------------------------------------------------
        
    
    @property
    def comm(self):
        return self._comm

    @property
    def octree(self):
        return self._octree

    @property
    def N(self):
        return self._N_oct

    @property
    def n(self):
        return self._n_oct

    @property
    def mat(self):
        return self._mat

    @property
    def rhs(self):
        return self._rhs

    @property
    def sol(self):
        return self._sol

    @property
    def res(self):
        return self._res

    @property
    def h(self):
        return self._h

    @property
    def not_pen_centers(self):
        return self._centers_not_penalized
