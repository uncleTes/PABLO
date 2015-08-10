import numbers
import math

class Laplacian2D(object):   
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
        _edge (number) : length of the edge of the grid."""

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
        self.set_intercomm_structures()
        # If some arguments are not presents, function "setdefault" will set 
        # them to the default value.
        # Penalization.
        self._pen = kwargs.setdefault("penalization", 
                                      0)
        # Over-lapping.
        self._over_l = kwargs.setdefault("overlapping",
                                         False)
        # [[x_anchor, x_anchor + edge, 
        #   y_anchor, y_anchor + edge]...] = penalization boundaries (aka
        # foreground boundaries).
        self._f_bound = kwargs.setdefault("foreground boundaries",
                                          None)
        # [x_anchor, x_anchor + edge, 
        #  y_anchor, y_anchor + edge] = background boundaries.
        self._b_bound = kwargs.setdefault("background boundaries",
                                          None)
        # Checking existence of penalization boundaries and background 
        # boundaries. The construct \"if not\" is useful whether 
        # \"self._f_bound\" or \"self._b_bound\" are None ore with len = 0.
        if ((not self._f_bound) or 
            (not self._b_bound)):
            msg = "\"MPI Abort\" called during initialization "
            extra_msg = " Penalization or bakground boundaries or both are " +
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
        try:
            assert self._edge > 0.0
        except AssertionError:
            msg = "\"MPI Abort\" called during initialization "
            extra_msg = " Attribute \"self._edge\" equal or smaller than 0."
            self.log_msg(msg    ,
                         "error",
                         extra_msg)
            self._comm_w.Abort(1)

        # Length of the edge of an octree.
        self._h = self._edge / numpy.sqrt(self._N_oct)
   
    # Returns the center of the face neighbour.
    def neighbour_centers(self  ,
                         centers,
                         faces):
        """Function which returns the centers of neighbours, depending on 
           for which face we are interested into.
           
           Arguments:
               centers (tuple or list of tuple) : coordinates of the centers of 
                                                  the current octree.
               faces (int between 0 and 3 or list) : faces for which we are  
                                                     interested into knowing
                                                     the neighbour's center.
                                            
           Returns:
               a tuple or a list containing the centers evaluated."""

        if ((len(faces) != 1) and
            (len(faces) != len(centers)):
            msg = "\"MPI Abort\" called " 
            extra_msg = " Different length of \"faces\" and \"centers\"."
            self.log_msg(msg    ,
                         "error",
                         extra_msg)
            self._comm_w.Abort(1)

        h = self._h
        centers = []
        for i, face in enumerate(faces):
            (x_center, y_center) = centers[i]
            if not isinstance(face, numbers.Integral):
                face = int(math.ceil(face))
            try:
                # Python's comparison chaining idiom.
                assert 0 <= face <= 3
            except AssertionError:
                msg = "\"MPI Abort\" called " 
                extra_msg = " Faces numeration incorrect."
                self.log_msg(msg    ,
                             "error",
                             extra_msg)
                self._comm_w.Abort(1)
            else:
                if ((face % 2) == 0):
                    (x_center = x_center - h) if (face == 0) else \
                    (y_center = y_center - h)
                else:
                    (x_center = x_center + h) if (face == 1) else \
                    (y_center = y_center + h)

                centers.append((x_center, y_center))

        if len(centers) == 1:
            return centers[0]
                
        return centers

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
               the evaluated boundary condition or a list of them."""

        # Centers neighbours.
        c_neighs = self.neighbour_centers(centers,
                                          faces)

        x_s = [c_neigh[0] for c_neigh in c_neighs] 
        y_s = [c_neigh[1] for c_neigh in c_neighs] 

        boundary_values = ExactSolution2D.solution(x_s, 
                                                   y_s)

        return boundary_values

    # Overlap adds.
    def over_adds(b_centers,
                  b_faces  ,
                  b_values ,
                  b_indices):
        f_bound = self._f_bound
        neigh_centers = self.neighbour_centers(b_centers,
                                               b_faces)
        for i, neigh_center in enumerate(neigh_centers):
            # Check on the current extra border octant of the background grid if
            # is overlapped by foreground grids.
            check = utilities.check_into_squares(neigh_center,
                                                 f_bound     ,
                                                 self.logger ,
                                                 log_file)
            if check and b_faces[i] == 1:
                key = (grid, b_indices[i], "ghost_boundary")
                self.__temp_data_local.update({key : neigh_center})
                b_values[i] = self.__inter_extra_array_ghost_boundary.getValue(b_indices[i])

    
    # Set boundary conditions.
    def set_b_c(self):
        penalization = self._pen
        b_bound = self._b_bound
        grid = self._proc_g
        n_oct = self._n_oct
        nfaces = glob.nfaces
        # \"getOwnershipRange()\" gives us the local ranges of the matrix owned
        # by the current process.
        o_ranges = self._mat.getOwnershipRange()
        h = self._h
        h2 = h * h
        is_background = False
        # If we are onto the grid \"0\", we are onto the background grid.
        if not grid:
            is_background = True

        b_indices, b_values = ([] for i in range(0, 2))# Boundary indices/values
        b_centers, b_faces = ([] for i in range(0, 2)) # Boundary centers/faces
        for octant in xrange(0, n_oct):
            # Global index of the current local octant \"octant\".
            g_octant = o_ranges[0] + octant
            py_oct = self._octree.get_octant(octant)
            center  = self._octree.get_center(octant)[:2]

            for face in xrange(0, nfaces):
                # If we have an edge on the boundary.
                if self._octree.get_bound(py_oct, 
                                          face):
                    b_indices.append(g_octant)
                    b_faces.append(face)
                    b_centers.append(center)
            
        b_values = self.eval_b_c(b_centers,
                                 b_faces)

        if is_background:
            if overlapping:
                self.over_adds(b_centers,
                               b_faces  ,
                               b_values ,
                               b_indices) 
        # Grids not of the background: equal to number >= 1.
        else:
            for i, center in enumerate(b_centers):
                # Check if foreground grid is inside the background one.
                check = check_into_square(center     ,
                                          b_bound    ,
                                          self.logger,
                                          log_file)
                if check:
                    # Can't use list as dictionary's keys.
                    # http://stackoverflow.com/questions/7257588/why-cant-i-use-a-list-as-a-dict-key-in-python
                    # https://wiki.python.org/moin/DictionaryKeys
                    key = (grid        , # Grid (0 is for the background grid)
                           b_indices[i], # Global index of the octant
                           faces[i]    , # Boundary face
                           h)            # Edge's length
                    # We store the centers of the cells on the boundary.
                    self.__temp_data_local.update({key : center})
                    b_values[i] = self.__inter_extra_array.getValue(b_indices[i])
                    # Residual evaluation...
                    sol_value = self._sol.getValue(b_indices[i])
                    self._residual_local.update({tuple(center) : sol_value})
                else:
                    b_values[i] = self.eval_b_c(center,
                                                b_faces[i])

        b_values[:] = [b_value * (-1/h2) for b_value in b_values]
        insert_mode = PETSc.InsertMode.ADD_VALUES
        self.__rhs.setValues(b_indices,
                             b_values ,
                             insert_mode)
        # ATTENTION!! Non using these functions will give you an unassembled
        # vector PETSc.
        self.__rhs.assemblyBegin()
        self.__rhs.assemblyEnd()
        msg = "Set boundary conditions"
        extra_msg = "of grid \"" + str(self._proc_g) + "\""
        self.log_msg(msg   ,
                     "info",
                     extra_msg)

    # Init global ghosts.
    def init_g_g(self):
        # Number of local ghosts (if present).
        g_n_oct = self._octree.get_num_ghosts()
        self._global_ghosts = []

        for g_octant in xrange(0, g_n_oct):
            # Getting global index for ghost octant.
            gg_idx = self._octree.get_ghost_global_idx(g_octant)
            # Saving all global indeces for the ghost octants, for each single
            # process. This is useful for PETSc.
            self._global_ghosts.append(gg_idx)
   
