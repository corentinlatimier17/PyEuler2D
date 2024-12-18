import numpy as np

class Mesh():
    def __init__(self, filepath, n_ghosts=2):

        self.filepath = filepath # Path of the mesh file
        self.n_ghosts = n_ghosts # Number of ghost layers we want to use -> we assume two layers

        self.ni = None # Number of cells in the i-direction
        self.nj = None # Number of cells in the j-direction

        self.x_coordinates = None # Matrix of size (ni+1, nj+1) that contains x coordinate of each vertice in the mesh
        self.y_coordinates = None # Matrix of size (ni+1, nj+1) that contains y coordinate of each vertice in the mesh

        self.x_ext = None # Matrix of size (ni+1 + 2*n_ghosts, nj+1 + 2*n_ghosts) that contains x coordinate of each vertice in the extended mesh (physical mesh + ghost layers)
        self.y_ext = None # Matrix of size (ni+1 + 2*n_ghosts, nj+1 + 2*n_ghosts) that contains y coordinate of each vertice in the extended mesh (physical mesh + ghost layers)

        self.areas = None # Matrix of size (ni + 2*n_ghosts, nj + 2*n_ghosts) that contains area of each cell in the extended mesh (physical mesh + ghost layers)

        self.Delta_ix = None  # Spatial variation along x-axis in i direction of each cell (matrix of size (ni + 2*n_ghosts, nj + 1 + 2*n_ghosts))
        self.Delta_iy = None  # Spatial variation along x-axis in i direction of each cell (matrix of size (ni + 2*n_ghosts, nj + 1 + 2*n_ghosts))

        self.Delta_jx = None  # Spatial variation along x-axis in j direction of each cell (matrix of size (ni + 2*n_ghosts + 1, nj + 2*n_ghosts))
        self.Delta_jy = None  # Spatial variation along y-axis in j direction of each cell (matrix of size (ni + 2*n_ghosts + 1, nj + 2*n_ghosts))

        self.read_mesh()
        self.create_extended_mesh()
        self.compute_areas()
        self.compute_faces()


    def read_mesh(self):
        """This function reads a mesh in plot3D format
        -> (x,y) coordinates of each vertices are stored in x_coordinates and y_coordinates attributes in a matrix form
        -> ni, nj attributes of the mesh object are initialized
        """
        with open(self.filepath, 'r') as f:
            # Skip the first line that contains the number of blocks
            next(f)

            # Read the n_vertices_i and n_vertices_j on the second line
            line = f.readline().strip()
            values = line.split()  # Split line into values

            mi = int(values[0]) # Number of vertices in the i-direction
            mj = int(values[1]) # Number of vertices in the j-direction

            self.ni = mi-1 # If we have mi vertices in the i-direction, we have mi-1 cells in the i-direction
            self.nj = mj-1 # If we have mj vertices in the j-direction, we have mj-1 cells in the j-direction
                
            # Initialize arrays for x and y
            self.x_coordinates = np.zeros((mi, mj))
            self.y_coordinates = np.zeros((mi, mj))
                
            # Process the x-coordinates
            for i in range(mi):
                for j in range(mj):
                    line = f.readline().strip()
                    values = line.split() 
                    self.x_coordinates[i, j] = float(values[0]) 

            # Process the y-coordinates
            for i in range(mi):
                for j in range(mj):
                    line = f.readline().strip()
                    values = line.split() 
                    self.y_coordinates[i, j] = float(values[0])  
        
    def create_extended_mesh(self):
        """This function creates a mesh extended with ghost layers so as to using the same stencil in the discretization for each "physical cell"
        -> fill the x_ext and y_ext attributes
        -> for the physical mesh, (x,y) coordinates of x_coordinates and y_coordinates are used 
        -> for the ghost layers, it depends on the boundary conditions of the problem : 

                j ->    farfield
                    -----------------
                    |               |
        connect     |               | connect
                ^   |               |
                | i  ----------------
                          wall
        """

        self.x_ext = np.zeros((self.n_ghosts*2 + self.ni + 1, self.n_ghosts*2 + self.nj + 1))
        self.y_ext = np.zeros((self.n_ghosts*2 + self.ni + 1, self.n_ghosts*2 + self.nj + 1))

        self.x_ext[self.n_ghosts : self.n_ghosts + self.ni + 1, self.n_ghosts : self.n_ghosts + self.nj + 1] = self.x_coordinates
        self.y_ext[self.n_ghosts : self.n_ghosts + self.ni + 1, self.n_ghosts : self.n_ghosts + self.nj + 1] = self.y_coordinates


        # Boundary conditions n°1 : connection between first layer (physical cell) (in the j-direction) and last layer (in the j-direction)
        # -> ghost cells will have the same geometrical properties to account of the connection
        self.x_ext[:, 0]  = self.x_ext[:, -(self.n_ghosts + 2)]
        self.x_ext[:, 1]  = self.x_ext[:, -(self.n_ghosts + 1)]
        self.x_ext[:, -1] = self.x_ext[:,   self.n_ghosts + 1 ]
        self.x_ext[:, -2] = self.x_ext[:,   self.n_ghosts     ]

        self.y_ext[:, 0]  = self.y_ext[:, -(self.n_ghosts + 2)]
        self.y_ext[:, 1]  = self.y_ext[:, -(self.n_ghosts + 1)]
        self.y_ext[:, -1] = self.y_ext[:,   self.n_ghosts + 1 ]
        self.y_ext[:, -2] = self.y_ext[:,   self.n_ghosts     ]

        # Boundary condition n°2 : wall -> we extrapolate the geometrical properties of the ghost cells from the physical cells at the wall
        self.x_ext[0, :]  = self.x_ext[self.n_ghosts, :]
        self.x_ext[1, :]  = self.x_ext[self.n_ghosts, :]

        self.y_ext[0, :]  = self.y_ext[self.n_ghosts, :]
        self.y_ext[1, :]  = self.y_ext[self.n_ghosts, :]

        # Boundary condition n°3 : farfield -> we extrapolate the geometrical properties of the ghost cells from the physical cells at the farfield
        self.x_ext[-1, :] = self.x_ext[-(self.n_ghosts + 1), :]
        self.x_ext[-2, :] = self.x_ext[-(self.n_ghosts + 1), :]

        self.y_ext[-1, :] = self.y_ext[-(self.n_ghosts + 1), :]
        self.y_ext[-2, :] = self.y_ext[-(self.n_ghosts + 1), :]
    
    def compute_areas(self):
        """This function computes the area of each cell in the extended mesh 

        0 ---------- 3
         |           |
         |           |
         |           |
        1 -----------2
        This function has been vectorized for efficiency
        """
        self.areas = np.zeros((self.ni + 2*self.n_ghosts, self.nj + 2*self.n_ghosts))

        # Extraire les coordonnées des sommets pour chaque cellule
        x0, y0 = self.x_ext[:-1 , :-1], self.y_ext[:-1 , :-1]  # Bas-gauche
        x1, y1 = self.x_ext[1:  , :-1], self.y_ext[1:  , :-1]    # Bas-droite
        x2, y2 = self.x_ext[1:  ,  1:], self.y_ext[1:  ,  1:]      # Haut-droite
        x3, y3 = self.x_ext[:-1 ,  1:], self.y_ext[:-1 ,  1:]    # Haut-gauche

        # Calculer l'aire en utilisant l'algorithme vectorisé
        self.areas = 0.5 * np.abs((x2 - x0) * (y1 - y3) - (y2 - y0) * (x1 - x3))

    def compute_faces(self):
        """Compute the spatial variations along the i and j directions (vectorized)."""
        # Compute Delta_i (differences in the i-direction)
        self.Delta_ix = self.x_ext[1:, :] - self.x_ext[:-1, :]
        self.Delta_iy = self.y_ext[1:, :] - self.y_ext[:-1, :]

        # Compute Delta_j (differences in the j-direction)
        self.Delta_jx = self.x_ext[:, 1:] - self.x_ext[:, :-1]
        self.Delta_jy = self.y_ext[:, 1:] - self.y_ext[:, :-1]
        return self.Delta_iy
