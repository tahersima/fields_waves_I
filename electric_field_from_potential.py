#Mohammad H. Tahersima / All Rights Reserved 
#Compute and plot electric potential distributions and corresponding fields 

import numpy as np
import matplotlib.pyplot as plt

def create_grid(grid_size, bounds=(-5, 5)):
    """
    Receives grid_size, an integer indicating spatial resolution 
    Creates a 3D grid and returns the meshgrid coordinates and step sizes.
    """
    start, stop = bounds
    coords = np.linspace(start, stop, grid_size)
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    return x, y, z, coords
  
def calculate_field_and_gradients(potential_func, X, Y, Z, x_coords, y_coords, z_coords):
    """
    Calculates the scalar field and its numerical gradients.
    """
    field = potential_func(X, Y, Z)
    dx, dy, dz = np.gradient(field, x_coords, y_coords, z_coords, edge_order=2)
    return field, dx, dy, dz
  
def plot_gradient_vectors(ax, X, Y, F, dx, dy, points, z_slice_index, grid_size, bounds=(-5, 5)):
    """
    Plots a 2D cross-section of the field with gradient vectors at specified points.
    """
    start, stop = bounds
    # Plot 2D cross-section
    contour = ax.contourf(X[:, :, z_slice_index], Y[:, :, z_slice_index], F[:, :, z_slice_index], 20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Field Strength')
    # Plot gradient vectors at selected points
    for point in points:
        # Convert point to grid indices
        xi = int((point[0] - start) * (grid_size - 1) / (stop - start))
        yi = int((point[1] - start) * (grid_size - 1) / (stop - start))
        # Get gradients (only dx and dy components for 2D plot)
        gx = dx[xi, yi, z_slice_index]
        gy = dy[xi, yi, z_slice_index]
        # Plot vector
        ax.quiver(point[0], point[1], gx, gy, color='red', scale=15, width=0.005, headwidth=5, headlength=7)
    # Mark the points
    ax.scatter(points[:, 0], points[:, 1], c='white', s=50, edgecolors='black')
  
def plot_electric_field(ax, dx, dy, z_slice_index, grid_size, bounds=(-5, 5), num_points=20):
    """
    Calculates and plots the electric field (E = -∇φ) as a quiver plot.
    """
    start, stop = bounds
    # Create a new, smaller grid for the electric field
    coords_e = np.linspace(start + 0.5, stop - 0.5, num_points)
    X_e, Y_e = np.meshgrid(coords_e, coords_e)
    Ex = np.zeros_like(X_e)
    Ey = np.zeros_like(Y_e)
    # Calculate E-field at each grid point
    for i in range(num_points):
        for j in range(num_points):
            # Convert position to original grid indices
            xi = int((X_e[i,j] - start) * (grid_size - 1) / (stop - start))
            yi = int((Y_e[i,j] - start) * (grid_size - 1) / (stop - start))
            # Electric field is negative gradient of potential
            Ex[i,j] = -dx[xi, yi, z_slice_index]
            Ey[i,j] = -dy[xi, yi, z_slice_index]
    # Plot electric field vectors
    ax.quiver(X_e, Y_e, Ex, Ey, color='blue', scale=25, width=0.004, headwidth=3, headlength=5, label='Electric Field (E = -∇φ)')
    ax.legend(loc='upper right')
  
# --- Main Execution ---
if __name__ == "__main__":
    # define the scalar field function
    def potential(x, y, z):
        return x**3*y*z + 0.2 * z**2
    # set initial parameters
    GRID_SIZE = 100
    Z_SLICE_INDEX = GRID_SIZE // 2
    POINTS = np.array([[-4, 3, 0], [2, -1, 0]]) # at z=0; 2D plot
    # generate grid and compute field/gradients
    X, Y, Z, coords = create_grid(GRID_SIZE)
    F, dx, dy, dz = calculate_field_and_gradients(potential, X, Y, Z, coords, coords, coords)
    # show potential values
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    plot_gradient_vectors(ax1, X, Y, F, dx, dy, POINTS, Z_SLICE_INDEX, GRID_SIZE)
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_title('2D Cross-Section with Gradient Vectors')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    plt.show()
    # show field vectors
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    plot_electric_field(ax2, dx, dy, Z_SLICE_INDEX, GRID_SIZE)
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    ax2.set_title('Electric Field (E = -∇φ)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    plt.show()
