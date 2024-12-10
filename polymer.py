import numpy as np
import matplotlib.pyplot as plt
from numba import njit



def getData(function,STEPS = 10_000,iterations = 1):
    endToEndData = np.zeros((iterations,STEPS))
    RGsData = np.zeros((iterations,STEPS))
    BLsData = np.array([])

    for i in range(iterations):
        print(i)
        positions = function(STEPS)

        endToEnds = np.linalg.norm(positions - positions[0],axis=1)
        RGs = np.zeros(STEPS)
        for j in range(1,STEPS):
            RGs[j] = cal_radius_of_gyration(positions[:j])
        BLs = cal_bond_lengths(positions)

        endToEndData[i] = endToEnds
        RGsData[i] = RGs
        BLsData = np.concatenate([BLsData,BLs])

    return np.mean(endToEndData,axis=0),np.mean(RGsData,axis=0),BLsData





@njit
def cal_end_to_end_distance(pos):
    return np.linalg.norm(pos[-1] - pos[0])





@njit
def cal_end_to_end_distances(pos):
    return np.linalg.norm(pos - pos[0],axis=0)



def cal_radius_of_gyration(pos):
    center_of_mass = np.mean(pos, axis=0)
    return np.sqrt(np.mean(np.sum((pos - center_of_mass)**2, axis=1)))


def cal_bond_lengths(pos):
    return np.linalg.norm(np.diff(pos, axis=0), axis=1)


#1


@njit
def FJC(num_segments = 1000 ,bond_length = 1.0 ,dimension = 3):
    
    # Initialize arrays to store positions of monomers
    positions = np.zeros((num_segments, dimension))

    # Generate FJC positions
    for i in range(1, num_segments):
        step = np.random.normal(0, 1, dimension)  # Random step in each dimension
        step /= np.linalg.norm(step)  # Normalize step to unit length
        positions[i] = positions[i-1] + bond_length * step

    return positions


#2



def is_valid_move(position, lattice):
    """Check if the move to the new position is valid (i.e., not occupied)."""
    return tuple(position) not in lattice

@njit
def get_neighbors(position):
    """Generate all possible moves from the current position."""
    neighbors = [
        position + np.array([1, 0, 0]),
        position + np.array([-1, 0, 0]),
        position + np.array([0, 1, 0]),
        position + np.array([0, -1, 0]),
        position + np.array([0, 0, 1]),
        position + np.array([0, 0, -1])
    ]
    return neighbors

def SAW(num_segments = 1000 ):

    start_position = np.array([0, 0, 0])  # Starting position

    # Initialize arrays to store the positions of the polymer segments
    positions = np.zeros((num_segments, 3), dtype=int)
    positions[0] = start_position
    lattice = {tuple(start_position)}  # Set to keep track of occupied positions

    # Generate the polymer chain
    for i in range(1, num_segments):
        neighbors = get_neighbors(positions[i-1])
        valid_moves = [pos for pos in neighbors if is_valid_move(pos, lattice)]
        if not valid_moves:
            print("Trapped! The chain cannot grow further.")
            break
        new_position = valid_moves[np.random.randint(len(valid_moves))]
        positions[i] = new_position
        lattice.add(tuple(new_position))


    return positions


#3



def MDC(num_segments = 1000  ,bond_length = 1.0  ,temperature = 300  ,k_b = 1.38e-23  ,mass = 1.0 ,time_step = 1e-3):
    num_steps = 10*num_segments
    # Initialize arrays to store the positions and velocities of the polymer segments
    positions = np.zeros((num_segments, 3))
    velocities = np.zeros((num_segments, 3))

    # Initial random positions
    positions[1:] = np.cumsum(np.random.normal(0, bond_length, size=(num_segments-1, 3)), axis=0)


        # Simulation loop
    for step in range(num_steps):
        # Calculate forces
        forces = np.zeros_like(positions)
        
        # Harmonic bond forces
        bond_vectors = np.diff(positions, axis=0)
        bond_lengths = np.linalg.norm(bond_vectors, axis=1)
        bond_unit_vectors = bond_vectors / bond_lengths[:, np.newaxis]
        bond_forces = - (bond_lengths - bond_length)[:, np.newaxis] * bond_unit_vectors
        forces[1:] += bond_forces
        forces[:-1] -= bond_forces
        
        # Langevin thermostat
        friction_coefficient = np.sqrt(2 * mass * k_b * temperature / time_step)
        random_forces = friction_coefficient * np.random.normal(size=velocities.shape)
        forces += random_forces
        
        # Update velocities and positions
        velocities += forces / mass * time_step
        positions += velocities * time_step


    return positions



#4



def BROWNIAN(num_segments = 1000 ,bond_length = 1.0  ,temperature = 300 ,k_b = 1.38e-23,gamma = 1.0,time_step = 1e-3):
    num_steps = 10*num_segments
    # Initialize arrays to store the positions of the polymer segments
    positions = np.zeros((num_segments, 3))
    positions[1:] = np.cumsum(np.random.normal(0, bond_length, size=(num_segments-1, 3)), axis=0)


    # Simulation loop
    for step in range(num_steps):
        # Calculate bond vectors and forces
        bond_vectors = np.diff(positions, axis=0)
        bond_lengths = np.linalg.norm(bond_vectors, axis=1)
        forces = - (bond_lengths - bond_length)[:, np.newaxis] * bond_vectors / bond_lengths[:, np.newaxis]
        
        # Add thermal noise
        noise = np.sqrt(2 * k_b * temperature * gamma / time_step) * np.random.normal(size=positions.shape)
        
        # Update positions with Brownian dynamics equation
        velocities = forces / gamma
        positions[1:] += velocities * time_step + noise[1:] * np.sqrt(time_step)


    return positions



#5
@njit
def GAUSSIAN(num_segments = 1000,bond_length_mean = 1.0):

    # Generate the polymer chain
    bond_lengths = np.random.normal(bond_length_mean, 0.1, num_segments)
    angles = np.random.uniform(0, 2 * np.pi, (num_segments, 2))

    # Initialize the positions array
    positions = np.zeros((num_segments, 3))


    # Generate the positions
    for i in range(1, num_segments):
        theta, phi = angles[i]
        direction = np.array([
            bond_lengths[i] * np.sin(theta) * np.cos(phi),
            bond_lengths[i] * np.sin(theta) * np.sin(phi),
            bond_lengths[i] * np.cos(theta)
        ])
        positions[i] = positions[i-1] + direction


    return positions



#6


def RIS(num_segments = 1000 ,bond_length = 1.0 ,dihedral_angles = [0, 120, -120],dihedral_probabilities = [0.5, 0.25, 0.25]):
    # Initialize arrays to store the positions of the polymer segments
    positions = np.zeros((num_segments, 3))

    # Initialize the initial bond and direction
    bond_vector = np.array([bond_length, 0, 0])
    direction = np.array([0, 0, 1])



    # Generate the polymer chain
    for i in range(1, num_segments):
        # Select a dihedral angle based on the given probabilities
        dihedral_angle = np.random.choice(dihedral_angles, p=dihedral_probabilities)
        dihedral_angle_rad = np.deg2rad(dihedral_angle)

        # Generate random theta and phi angles for the new bond direction
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        # Calculate the rotation matrix around the z-axis by the dihedral angle
        rotation_matrix_z = np.array([
            [np.cos(dihedral_angle_rad), -np.sin(dihedral_angle_rad), 0],
            [np.sin(dihedral_angle_rad), np.cos(dihedral_angle_rad), 0],
            [0, 0, 1]
        ])

        # Rotate the bond vector by the dihedral angle
        bond_vector = np.dot(rotation_matrix_z, bond_vector)

        # Calculate the new bond direction
        bond_vector = bond_length * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Update the position of the next segment
        positions[i] = positions[i-1] + bond_vector

    return positions



#7


def FRC(num_segments = 1000 ,bond_length = 1.0 ,bond_angle = np.pi / 3 ):

    # Initialize arrays to store the positions of the polymer segments
    positions = np.zeros((num_segments, 3))

    # Initial bond direction
    direction = np.array([1, 0, 0])


    # Generate the polymer chain
    for i in range(1, num_segments):
        # Generate random rotation about the current bond
        theta = bond_angle
        phi = 2 * np.pi * np.random.rand()
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        direction = np.dot(rotation_matrix, direction)
        direction = direction / np.linalg.norm(direction)  # Normalize to unit vector
        
        # Rotate about the z-axis by random angle phi
        rotation_matrix_z = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])
        direction = np.dot(rotation_matrix_z, direction)
        direction = direction / np.linalg.norm(direction)  # Normalize to unit vector
        
        # Update the position of the next segment
        positions[i] = positions[i-1] + bond_length * direction

    return positions



#8


def MCL_init(num_segments = 100 ,lattice_size = 50 ,dimension = 3  ,max_displacement = 1):
    num = 0
    max_num = 10000
    # Initialize lattice and positions
    lattice = np.zeros((lattice_size, lattice_size, lattice_size), dtype=bool)
    positions = np.zeros((num_segments, dimension), dtype=int)

    # Place the initial chain on the lattice
    positions[0] = np.random.randint(0, lattice_size, size=dimension)
    lattice[tuple(positions[0])] = True
    for i in range(1, num_segments):
        placed = False
        while not placed:
            if num == max_num:
                return MCL_init(num_segments,lattice_size,dimension,max_displacement)
                 

            num += 1
            direction = np.random.randint(-1, 2, size=dimension)
            new_position = positions[i-1] + direction
            if np.all(new_position >= 0) and np.all(new_position < lattice_size) and not lattice[tuple(new_position)]:
                positions[i] = new_position
                lattice[tuple(new_position)] = True
                placed = True

    return positions,lattice




def MCL(num_segments = 100 ,lattice_size = 50 ,dimension = 3  ,max_displacement = 1):
    num_steps = 10 * num_segments

    # Initialize lattice and positions
    lattice = np.zeros((lattice_size, lattice_size, lattice_size), dtype=bool)
    positions = np.zeros((num_segments, dimension), dtype=int)

    positions,lattice = MCL_init(num_segments,lattice_size,dimension,max_displacement)

    # Simulation
    for step in range(num_steps):
        segment_index = np.random.randint(1, num_segments - 1)
        direction = np.random.randint(-1, 2, size=dimension)
        new_position = positions[segment_index] + direction
        if np.all(new_position >= 0) and np.all(new_position < lattice_size) and not lattice[tuple(new_position)]:
            # Check bond length constraints
            if np.all(np.linalg.norm(new_position - positions[segment_index - 1]) <= max_displacement) and \
                    np.all(np.linalg.norm(new_position - positions[segment_index + 1]) <= max_displacement):
                lattice[tuple(positions[segment_index])] = False
                positions[segment_index] = new_position
                lattice[tuple(new_position)] = True

    return positions



#9


@njit
def KLM(num_segments = 100,kuhn_length = 1.0,dimension = 3):
    # Initialize arrays to store positions of segments
    positions = np.zeros((num_segments, dimension))

    # Generate initial positions (linear configuration)
    for i in range(1, num_segments):
        direction = np.random.normal(0, 1, dimension)
        direction /= np.linalg.norm(direction)  # Normalize to unit length
        positions[i] = positions[i-1] + kuhn_length * direction

    return positions
    


#10



@njit
def REPTATION(num_segments = 100,bond_length = 1.0 ,dimension = 3,tube_diameter = 5.0):

    num_steps = 10 * num_segments
    # Initialize arrays to store positions of monomers
    positions = np.zeros((num_segments, dimension))

    # Generate initial positions (linear configuration)
    for i in range(1, num_segments):
        positions[i] = positions[i-1] + np.array([bond_length, 0, 0])

    # Reptation model simulation
    for step in range(num_steps):
        # Select a random segment (excluding the ends)
        segment_index = np.random.randint(1, num_segments-1)
        
        # Move the selected segment in a random direction within the tube
        random_displacement = np.random.normal(0, bond_length, dimension)
        random_displacement /= np.linalg.norm(random_displacement)  # Normalize to unit length
        random_displacement *= bond_length  # Scale by bond length
        
        # Ensure the new position is within the tube
        new_position = positions[segment_index] + random_displacement
        displacement_from_axis = new_position - positions[segment_index-1]
        if np.linalg.norm(displacement_from_axis) <= tube_diameter:
            positions[segment_index] = new_position

    return positions



#11




def WLC(num_segments=100, segment_length=1.0, persistence_length=10.0):
    chain = np.zeros((num_segments, 3))
    directions = np.zeros((num_segments, 3))
    directions[0] = np.array([0, 0, 1])  # Start with the first direction along z-axis

    for i in range(1, num_segments):
        theta = np.random.normal(0, np.sqrt(segment_length / persistence_length))
        phi = np.random.uniform(0, 2 * np.pi)

        # Spherical to Cartesian conversion
        direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Ensure the new direction is correlated with the previous direction
        directions[i] = (directions[i-1] + direction)
        directions[i] /= np.linalg.norm(directions[i])

        # Update chain position
        chain[i] = chain[i-1] + segment_length * directions[i]

    return chain



#12




def EVC(num_segments = 100,lattice_size = 100,dimension = 3,max_displacement = 1):

    num_steps = 10 * num_segments
    # Initialize lattice and positions
    lattice = np.zeros((lattice_size, lattice_size, lattice_size), dtype=bool)
    positions = np.zeros((num_segments, dimension), dtype=int)

    # Place the initial chain on the lattice
    positions[0] = np.random.randint(0, lattice_size, size=dimension)
    lattice[tuple(positions[0])] = True
    for i in range(1, num_segments):
        placed = False
        while not placed:
            direction = np.random.randint(-1, 2, size=dimension)
            new_position = positions[i-1] + direction
            if np.all(new_position >= 0) and np.all(new_position < lattice_size) and not lattice[tuple(new_position)]:
                positions[i] = new_position
                lattice[tuple(new_position)] = True
                placed = True

    # Simulation
    for step in range(num_steps):
        segment_index = np.random.randint(1, num_segments - 1)
        direction = np.random.randint(-1, 2, size=dimension)
        new_position = positions[segment_index] + direction
        if np.all(new_position >= 0) and np.all(new_position < lattice_size) and not lattice[tuple(new_position)]:
            # Check bond length constraints
            if np.all(np.linalg.norm(new_position - positions[segment_index - 1]) <= max_displacement) and \
                    np.all(np.linalg.norm(new_position - positions[segment_index + 1]) <= max_displacement):
                lattice[tuple(positions[segment_index])] = False
                positions[segment_index] = new_position
                lattice[tuple(new_position)] = True

    return positions



#13



def initialize_chain(N, L):
    allowed_bonds = np.array([(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2),
                          (2, 2, 0), (2, -2, 0), (-2, 2, 0), (-2, -2, 0),
                          (2, 0, 2), (2, 0, -2), (-2, 0, 2), (-2, 0, -2),
                          (0, 2, 2), (0, 2, -2), (0, -2, 2), (0, -2, -2),
                          (2, 2, 2), (2, 2, -2), (2, -2, 2), (2, -2, -2),
                          (-2, 2, 2), (-2, 2, -2), (-2, -2, 2), (-2, -2, -2)])
    chain = np.zeros((N, 3), dtype=int)
    for i in range(1, N):
        while True:
            direction = allowed_bonds[np.random.randint(len(allowed_bonds))]
            new_position = chain[i-1] + direction
            if np.all(new_position >= 0) and np.all(new_position < L) and not np.any(np.all(chain[:i] == new_position, axis=1)):
                chain[i] = new_position
                break
    return chain

# Perform one Monte Carlo step
def monte_carlo_step(chain, L):
    num = 0 
    num_max = 10000
    allowed_bonds = np.array([(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2),
                          (2, 2, 0), (2, -2, 0), (-2, 2, 0), (-2, -2, 0),
                          (2, 0, 2), (2, 0, -2), (-2, 0, 2), (-2, 0, -2),
                          (0, 2, 2), (0, 2, -2), (0, -2, 2), (0, -2, -2),
                          (2, 2, 2), (2, 2, -2), (2, -2, 2), (2, -2, -2),
                          (-2, 2, 2), (-2, 2, -2), (-2, -2, 2), (-2, -2, -2)])
    N = len(chain)
    i = np.random.randint(1, N-1)
    old_pos = chain[i]
    while True:
        if num == num_max :
            return chain 
        num += 1


        direction = allowed_bonds[np.random.randint(len(allowed_bonds))]
        new_pos = old_pos + direction
        if np.all(new_pos >= 0) and np.all(new_pos < L) and not np.any(np.all(chain == new_pos, axis=1)):
            chain[i] = new_pos
            break
    return chain

# Compute end-to-end distance
def end_to_end_distance(chain):
    return np.linalg.norm(chain[-1] - chain[0])

# Compute radius of gyration
def radius_of_gyration(chain):
    com = np.mean(chain, axis=0)
    return np.sqrt(np.mean(np.sum((chain - com)**2, axis=1)))




def BFC(N = 500,L = 64):
    steps = 20 * N
    # Run the simulation
    chain = initialize_chain(N, L)
    end_to_end_distances = []
    radii_of_gyration = []

    for step in range(steps):
        chain = monte_carlo_step(chain, L)
        # if new_chain == 

        if step % 100 == 0:
            end_to_end_distances.append(end_to_end_distance(chain))
            radii_of_gyration.append(radius_of_gyration(chain))

    return chain

 