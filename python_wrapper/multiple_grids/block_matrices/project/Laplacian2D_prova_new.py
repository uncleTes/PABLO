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
    def neighbour_centers(self   ,
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

    def apply_overlap(self,
                      overlap):
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

        return p_bound
   
    # Init matrix.
    def init_mat(self,
                 # Overlap octants' number.
                 o_n_oct):
        penalization = self._penalization
        grid = self._proc_g
        self._mat = PETSc.Mat().create(comm = self._comm)
        # Local and global matrix's sizes.
        n_oct = self._n_oct
        N_oct = self._N_oct
        sizes = (n_oct, 
                 N_oct)
        self._mat.setSizes((sizes, 
                            sizes))
        # Setting type of matrix directly. Using method \"setFromOptions()\"
        # the user can choose what kind of matrix build at runtime.
        #self.__mat.setFromOptions()
        self._mat.setType(PETSc.Mat.Type.AIJ)
        # !!!TODO!!!
        # For better performances, instead of \"setUp()\" use 
        # \"setPreallocationCSR()\".
        # The AIJ format is also called the Yale sparse matrix format or
        # compressed row storage (CSR).
        # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html
        #self._mat.setPreallocationCSR((5, 4))
        self._mat.setUp()
        # Getting ranges of the matrix owned by the current process.
        o_ranges = self._mat.getOwnershipRange()
        h = self._h
        h2 = h * h
        nfaces = glob.nfaces
        is_background = False
        overlap = o_n_oct * h
        p_bound = []
        if not grid:
            is_background = True
            p_bound = self.apply_overlap(overlap)

        for octant in xrange(0, n_oct):
            indices, values = ([] for i in range(0, 2))# Indices/values
            neighs, ghosts = ([] for i in range(0, 2))
            g_octant = o_ranges[0] + octant
            py_oct = self._octree.get_octant(octant)
            center  = self._octree.get_center(octant)[:2]
            # Check to know if a quad(oc)tree on the background is penalized.
            is_penalized = False
            # Background grid.
            if is_background:
                is_penalized = check_into_squares(center     ,
                                                  p_bound    ,
                                                  self.logger,
                                                  log_file)
                if is_penalized:
                    key = (grid, g_octant)
                    self.__temp_data_local.update({key : center})
                # Residual evaluation...
                if check_into_squares(center     ,
                                      f_bound    ,
                                      self.logger,
                                      log_file): #and not is_penalized:
                    sol_value = self._solution.getValue(g_octant)
                    self.__residual_local.update({tuple(center) : sol_value})
            # Here we are, upper grids.
            else:
                circle_center = (0.5, 0.5)
                circle_radius = 0.125
                is_penalized = check_into_circle(center       ,
                                                 circle_center,
                                                 circle_radius)

            indices.append(g_octant)
            values.append(((-4.0 / h2) - penalization) if is_penalized 
                           else (-4.0 / h2))

            for face in xrange(0, nfaces):
                if not self._octree.get_bound(py_oct, 
                                              face):
                    (neighs, ghosts) = self._octree.find_neighbours(octant, 
                                                                    face  , 
                                                                    1     , 
                                                                    neighs, 
                                                                    ghosts)
                    if not ghosts[0]:
                        index = neighs[0] + o_ranges[0]
                    else:
                        index = self._octree.get_ghost_global_idx(neighs[0])
                    indices.append(index)
                    values.append(1.0 / h2)
                    
            self._mat.setValues(g_octant, # Rows
                                indices,  # Columns
                                values)   # Values to be inserted

        # ATTENTION!! Non using these functions will give you an unassembled
        # matrix PETSc.
        self._mat.assemblyBegin()
        self._mat.assemblyEnd()
        msg = "Initialized matrix"
        extra_msg = "with sizes \"" + str(self.__mat.getSizes()) + \
                    "\" and type \"" + str(self.__mat.getType()) + "\""
        self.log_msg(msg   ,
                     "info",
                     extra_msg)

    # Initialize extra arrays.
    def init_e_arrays(self,
                      array = None):
        self.e_array = self.init_array("extra array",
                                       array)
        self._e_array_gb = self.init_array("extra array ghost boundary",
                                           array)

    def init_array(self       ,
                   # Array name.
                   a_name = "",
                   array = None):
        pen = self._pen
        grid = self._proc_g
        n_oct = self._n_oct
        N_oct = self._N_oct
        sizes = (n_oct, 
                 N_oct)
        # Global ghosts.
        g_ghosts = self._global_ghosts
        # Temporary array.
        t_array = PETSc.Vec().createGhost(g_ghosts    ,
                                          size = sizes,
                                          comm = self._comm)
        t_array.setUp()

        if array is None:
            t_array.set(0)
        else:
            try:
                assert isinstance(array, numpy.ndarray)
                # Temporary PETSc vector.
                t_petsc = PETSc.Vec().createGhostWithArray(g_ghosts    ,
                                                           array       ,
                                                           size = sizes,
                                                           comm = self._comm)
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
    
    def init_rhs(self, 
                 numpy_array):
        grid = self._proc_g
        is_background = False
        if not grid:
            is_background = True

        numpy_rhs = numpy.subtract(numpy_array, 
                                   numpy.multiply(penalization,
                                                  self._inter_extra_array.getArray())) if \
                    is_background else \
                    numpy_array
        self._rhs = self.init_array("right hand side",
                                    numpy_rhs)
    
    def init_sol(self):
        self._sol = self.init_array("solution")
    
    def init_residual(self):
        self._res = self.init_array("residual")
    
    def solve(self):
        # Creating a "KSP" object.
        ksp = PETSc.KSP()
        pc = PETSc.PC()
        ksp.create(self._comm)
        ksp.setOperators(self._mat,
                         self._mat)

        pc = ksp.getPC()
        # Setting tolerances.
        tol = 1.e-50
        ksp.setTolerances(rtol = tol            , 
                          atol = tol            , 
                          divtol = PETSc.DEFAULT, # Let's PETSc use DEAFULT
                          max_it = PETSc.DEFAULT) # Let's PETSc use DEAFULT
        ksp.setFromOptions()
        pc.setFromOptions()
        # Solve the system.
        ksp.solve(self._rhs, 
                  self._solution)
        # How many iterations are done.
        it_number = ksp.getIterationNumber()

        msg = "Evaluated solution"
        extra_msg = "Using \"" + str(it_number) + "\" iterations."
        self.log_msg(msg   ,
                     "info",
                     extra_msg)
        self.init_e_arrays()
    
    def set_intercomm_structures(self):
        # Setting "self.__temp_data_local" to a dictionary because after 
        # the "allgather" operations it will became a list of dictionaries.
        # The "self.__temp_data_local" will contains local data to be exchanged
        # between grids of different levels.
        self.__temp_data_local = {}
        # The "self.__temp_data_global" will contains exchanged data between 
        # grids of different levels.
        self.__temp_data_global = []
        # The "self.__intra_extra_indices_local" will contains indices of the 
        # local data to be exchanged between grids of different levels.
        self.__intra_extra_indices_local = []
        # The "self.__intra_extra_indices_global" will contains indices of the 
        # excahnged data between grids of different levels.
        self.__intra_extra_indices_global = []
        # The "self.__intra_extra_values_local" will contains values of the 
        # local data to be exchanged between grids of different levels.
        self.__intra_extra_values_local = []
        # The "self.__intra_extra_indices_local" will contains values of the 
        # exchanged data between grids of different levels.
        self.__intra_extra_values_global = []
    
    def update_values(self, 
                      intercomm_dictionary = {}):
        n_oct = self._n_oct
        o_ranges = self._mat.getOwnershipRange()
        b_bound = self._b_bound
        grid = self._proc_g
        max_id_octree_contained = o_ranges[0] + local_nocts
        ids_octree_contained = range(o_ranges[0], max_id_octree_contained)
        # Calling "allgather" to obtain data from the corresponding grid,
        # onto the intercommunicators created, not the intracommunicators.
        # http://www.mcs.anl.gov/research/projects/mpi/mpi-standard/mpi-report-1.1/node114.htm#Node117
        # http://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/
        # http://www.linux-mag.com/id/1412/
        for key, intercomm in intercomm_dictionary.items():
            # Extending a list with the lists obtained by the other processes
            # of the corresponding intercommunicator.
            self.__temp_data_global.extend(intercomm.allgather(self.__temp_data_local))
            self.__residual_global.extend(intercomm.allgather(self.__residual_local))
        # Residual evaluation... 
        for index, dictionary in enumerate(self.__residual_global):
            for center, solution_value in dictionary.items():
                local_idx = self.__octree.get_point_owner_idx(center)
                global_idx = local_idx + o_ranges[0]

                if global_idx in ids_octree_contained:
                    center_cell_container = self.__octree.get_center(local_idx)[:2]
                    location = check_point_position_from_another(center,
                                                                 center_cell_container)
                    neigh_centers, neigh_values = ([] for i in range(0, 2))
                    (neigh_centers, neigh_values) = self.find_right_neighbours(location,
                                                                               local_idx,
                                                                               o_ranges[0])
                    bilinear_value = bilinear_interpolation(center,
                                                            neigh_centers,
                                                            neigh_values)
                    insert_mode = PETSc.InsertMode.INSERT_VALUES
                    value = bilinear_value - solution_value
                    self.__residual.setValue(global_idx, 
                                             value     ,
                                             insert_mode)
        self.__residual.assemblyBegin()
        self.__residual.assemblyEnd()

        # "self.__temp_data_global" will be a list of same structures of data,
        # after the "allgather" call; these structures are dictionaries.
        for index, dictionary in enumerate(self.__temp_data_global):
            for key, center in dictionary.items():
                (x_center, y_center) = center
                into_background = True
                ghost_boundary = False
                if len(key) == 3:
                    ghost_boundary = True
                # We are onto grids of the first level.
                if grid:
                    local_idx = self.__octree.get_point_owner_idx((x_center,
                                                                   y_center))
                # We are onto the background grid.
                else:
                    if key[2] == 0:
                        x_center = x_center - key[3]
                    if key[2] == 1:
                        x_center = x_center + key[3]
                    if key[2] == 2:
                        y_center = y_center - key[3]
                    if key[2] == 3:
                        y_center = y_center + key[3]

                    into_background = check_point_into_square_2D((x_center, 
                                                                  y_center)  ,
                                                                 b_boundaries,
                                                                 self.logger ,
                                                                 log_file)
                    if into_background:
                        # The function "get_point_owner_idx" wants only one argument
                        # so we are passing it a tuple.
                        local_idx = self.__octree.get_point_owner_idx((x_center,
                                                                       y_center))
                    # Is this "else" useful? For me no.
                    else:
                        local_idx = self.__octree.get_point_owner_idx(center)

                global_idx = local_idx + o_ranges[0]

                if global_idx in ids_octree_contained:
                    # Appending a tuple containing the grid number and
                    # the corresponding octant index.
                    if ghost_boundary:
                        self.__intra_extra_indices_local.append((key[0], key[1], key[2]))
                    else:
                        self.__intra_extra_indices_local.append((key[0], key[1]))
                    if into_background:
                        center_cell_container = self.__octree.get_center(local_idx)[:2]
                        location = check_point_position_from_another((x_center,
                                                                      y_center),
                                                                     center_cell_container)
                        neigh_centers, neigh_values = ([] for i in range(0, 2))
                        (neigh_centers, neigh_values) = self.find_right_neighbours(location ,
                                                                                   local_idx,
                                                                                   o_ranges[0])
                        solution_value = bilinear_interpolation((x_center, 
                                                                 y_center)   ,
                                                                neigh_centers,
                                                                neigh_values)
                    else:
                        solution_value = ExactSolution2D.solution(x_center, 
                                                                  y_center)

                    self.__intra_extra_values_local.append(solution_value)
        # Updating data for each process into "self.__intra_extra_indices_global"
        # and "self.__intra_extra_values_global", calling "allgather" to obtain 
        # data from the corresponding grid onto the intercommunicators created, 
        # not the intracommunicators.
        for key, intercomm in intercomm_dictionary.items():
            self.__intra_extra_indices_global.extend(intercomm.allgather(self.__intra_extra_indices_local))
            self.__intra_extra_values_global.extend(intercomm.allgather(self.__intra_extra_values_local))

        for index, values in enumerate(self.__intra_extra_indices_global):
            for position, value in enumerate(values):
                # Check if the global index belong to the process.
                if value[1] in ids_octree_contained:
                    # Check if we are onto the right grid.
                    if value[0] == self.__proc_grid:
                        intra_extra_value = self.__intra_extra_values_global[index][position]
                        # Background grid.
                        if grid == 0:
                            insert_mode = PETSc.InsertMode.INSERT_VALUES
                            # Here "insert_mode" does not affect nothing.
                            if len(value) == 3:
                                self.__inter_extra_array_ghost_boundary.setValue(value[1],
                                                                                 intra_extra_value,
                                                                                 insert_mode)
                            else:
                                self.__inter_extra_array.setValue(value[1]         , 
                                                                  intra_extra_value,
                                                                  insert_mode)
                        else:
                            insert_mode = PETSc.InsertMode.ADD_VALUES
                            self.__inter_extra_array.setValue(value[1]         ,
                                                              intra_extra_value,
                                                              insert_mode)

        self.__inter_extra_array.assemblyBegin()
        self.__inter_extra_array.assemblyEnd()
        self.__inter_extra_array_ghost_boundary.assemblyBegin()
        self.__inter_extra_array_ghost_boundary.assemblyEnd()
        # Resetting structures used for the "allgather" functions.
        self.set_intercomm_structures()
        self.logger.info("Updated  inter_extra_array for comm \"" +
                         str(self.__comm.Get_name())              + 
                         "\" and rank \""                         +
                         str(self.__comm.Get_rank())              +
                         "\" of grid \""                          +
                         str(self.__proc_grid)                    +
                         "\":\n"                                  +
                         str(self.__inter_extra_array.getArray()))
