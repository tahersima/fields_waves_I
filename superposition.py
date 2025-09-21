# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 2025
@author: Mohammad H. Tahersima
a brute force solution to electrostatic problems
computes electric fields and potential due to arbitrary number charges   
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

k = 8.99e9  # Coulomb's constant (N·m²/C²)
epsilon_0 = 8.85e-12  # Assuming free space: permittivity of free space (C²/N·m²)
MICRONS_TO_METERS = 1e-6  # Conversion factor from μm to m

def generate_mesh(x_range=(-10, 10), y_range=(-10, 10), spacing=0.1):
    """Generate a 2D mesh grid in μm"""
    x = np.arange(x_range[0], x_range[1] + spacing, spacing)
    y = np.arange(y_range[0], y_range[1] + spacing, spacing)
    X, Y = np.meshgrid(x, y)
    return X, Y

def place_random_charges(k_charges, x_range=(-10, 10), y_range=(-10, 10)):
    """Place k random charges with random positions and charges"""
    positions = np.random.uniform(low=[x_range[0], y_range[0]], 
                                 high=[x_range[1], y_range[1]], 
                                 size=(k_charges, 2))
    charges = np.random.uniform(-5, 5, k_charges)  # Charges in μC
    return positions, charges

def calculate_electric_fields(X, Y, charge_positions, charges):
    """Calculate electric potential and field at each point in the mesh"""
    # Convert charges from μC to C
    charges_c = charges * 1e-6
    
    # Convert positions from μm to m for calculation
    X_m = X * MICRONS_TO_METERS
    Y_m = Y * MICRONS_TO_METERS
    charge_positions_m = charge_positions * MICRONS_TO_METERS
    
    # Initialize arrays
    V = np.zeros_like(X)  # Electric potential (V)
    Ex = np.zeros_like(X)  # Electric field x-component (V/μm)
    Ey = np.zeros_like(X)  # Electric field y-component (V/μm)
    
    # Calculate contributions from each charge
    for i in range(len(charges)):
        dx_m = X_m - charge_positions_m[i, 0]
        dy_m = Y_m - charge_positions_m[i, 1]
        r_m = np.sqrt(dx_m**2 + dy_m**2)
        
        # Avoid division by zero (add small epsilon)
        r_m = np.where(r_m < 1e-15, 1e-15, r_m)
        
        # Electric potential (V)
        V += k * charges_c[i] / r_m
        
        # Electric field components (V/m)
        Ex_i = k * charges_c[i] * dx_m / r_m**3
        Ey_i = k * charges_c[i] * dy_m / r_m**3
        
        # Convert from V/m to V/μm
        Ex += Ex_i * MICRONS_TO_METERS
        Ey += Ey_i * MICRONS_TO_METERS
    
    return V, Ex, Ey

def plot_charges(charge_positions, charges):
    """Plot the location and magnitude of each charge"""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with color representing charge magnitude
    scatter = plt.scatter(charge_positions[:, 0], charge_positions[:, 1], 
                         c=charges, cmap='coolwarm', s=100, 
                         edgecolors='black', linewidth=1, vmin=-5, vmax=5)
    
    plt.colorbar(scatter, label='Charge (μC)')
    plt.xlabel('X position (μm)')
    plt.ylabel('Y position (μm)')
    plt.title('Location and Magnitude of Point Charges')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    
    # Add charge values as text annotations
    for i, (pos, charge) in enumerate(zip(charge_positions, charges)):
        plt.annotate(f'{charge:.1f}μC', (pos[0], pos[1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold')

def plot_potential_heatmap(X, Y, V, charge_positions):
    """Plot heatmap of electric potential"""
    plt.figure(figsize=(12, 9))
    
    # Create the heatmap
    im = plt.imshow(V, extent=[-10, 10, -10, 10], 
                   origin='lower', cmap='viridis',
                   aspect='equal', interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Electric Potential (V)', fontsize=12)
    
    # Overlay charge positions
    plt.scatter(charge_positions[:, 0], charge_positions[:, 1], 
               c='red', s=50, edgecolors='white', linewidth=1,
               label='Charges')
    
    plt.xlabel('X position (μm)', fontsize=12)
    plt.ylabel('Y position (μm)', fontsize=12)
    plt.title('Electric Potential Heatmap', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_electric_field(X, Y, Ex, Ey, V, charge_positions):
    """Plot electric field as streamlines over potential background"""
    plt.figure(figsize=(12, 9))
    
    # Create potential background
    im = plt.imshow(V, extent=[-10, 10, -10, 10], 
                   origin='lower', cmap='viridis', alpha=0.7,
                   aspect='equal')
    
    # Add colorbar for potential
    cbar = plt.colorbar(im)
    cbar.set_label('Electric Potential (V)', fontsize=12)
    
    # Plot electric field streamlines
    # Subsample for cleaner visualization
    stride = 5
    X_sub = X[::stride, ::stride]
    Y_sub = Y[::stride, ::stride]
    Ex_sub = Ex[::stride, ::stride]
    Ey_sub = Ey[::stride, ::stride]
    
    # Calculate field magnitude for coloring streamlines
    E_mag = np.sqrt(Ex_sub**2 + Ey_sub**2)
    
    # Plot streamlines colored by field strength
    streamplot = plt.streamplot(X_sub, Y_sub, Ex_sub, Ey_sub, 
                               color=E_mag, cmap='plasma', 
                               density=2, linewidth=1.5, arrowsize=1.5)
    
    # Add colorbar for electric field magnitude
    cbar2 = plt.colorbar(streamplot.lines, label='Electric Field Magnitude (V/μm)')
    
    # Overlay charge positions
    plt.scatter(charge_positions[:, 0], charge_positions[:, 1], 
               c='red', s=80, edgecolors='white', linewidth=2,
               label='Charges')
    
    plt.xlabel('X position (μm)', fontsize=12)
    plt.ylabel('Y position (μm)', fontsize=12)
    plt.title('Electric Field Streamlines with Potential Background', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_field_magnitude(X, Y, Ex, Ey, charge_positions):
    """Plot heatmap of electric field magnitude"""
    plt.figure(figsize=(12, 9))
    
    # Calculate electric field magnitude
    E_mag = np.sqrt(Ex**2 + Ey**2)
    
    # Create the heatmap (use log scale for better visualization)
    im = plt.imshow(np.log10(E_mag + 1e-10), extent=[-10, 10, -10, 10], 
                   origin='lower', cmap='hot',
                   aspect='equal', interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('log₁₀(Electric Field Magnitude) (V/μm)', fontsize=12)
    
    # Overlay charge positions
    plt.scatter(charge_positions[:, 0], charge_positions[:, 1], 
               c='red', s=50, edgecolors='white', linewidth=1,
               label='Charges')
    
    plt.xlabel('X position (μm)', fontsize=12)
    plt.ylabel('Y position (μm)', fontsize=12)
    plt.title('Electric Field Magnitude Heatmap (log scale)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

# Main execution
if __name__ == "__main__":
    # Parameters
    k_charges = 15  # Number of point charges
    spacing = 0.1   # Mesh spacing in μm
    
    # Generate mesh
    X, Y = generate_mesh(spacing=spacing)
    
    # Place random charges
    charge_positions, charges = place_random_charges(k_charges)
    
    # Calculate electric potential and field
    V, Ex, Ey = calculate_electric_fields(X, Y, charge_positions, charges)
    
    # Create plots
    plot_charges(charge_positions, charges)
    plot_potential_heatmap(X, Y, V, charge_positions)
    plot_electric_field(X, Y, Ex, Ey, V, charge_positions)
    plot_field_magnitude(X, Y, Ex, Ey, charge_positions)
    
    plt.tight_layout()
    plt.show()
    
    # Print some information
    print(f"Generated {k_charges} random charges")
    print(f"Mesh size: {X.shape}")
    print(f"Potential range: {V.min():.2e} V to {V.max():.2e} V")
    print(f"Electric field range: {np.sqrt(Ex**2 + Ey**2).min():.2e} to {np.sqrt(Ex**2 + Ey**2).max():.2e} V/μm")
    
    # Show charge details
    print("\nCharge details:")
    for i, (pos, charge) in enumerate(zip(charge_positions, charges)):
        print(f"Charge {i+1}: Position ({pos[0]:.1f}, {pos[1]:.1f}) μm, Charge: {charge:.2f} μC")
