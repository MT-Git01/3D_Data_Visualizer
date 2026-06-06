import numpy as np
import pandas as pd

def generate_lorenz_attractor(num_points=1500, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
    """Generates a 3D Lorenz attractor trajectory."""
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    z = np.zeros(num_points)
    
    # Initial state
    x[0], y[0], z[0] = 0.1, 0.0, 0.0
    
    for i in range(1, num_points):
        dx = sigma * (y[i-1] - x[i-1]) * dt
        dy = (x[i-1] * (rho - z[i-1]) - y[i-1]) * dt
        dz = (x[i-1] * y[i-1] - beta * z[i-1]) * dt
        
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy
        z[i] = z[i-1] + dz
        
    df = pd.DataFrame({
        'X_Lorenz': x,
        'Y_Lorenz': y,
        'Z_Lorenz': z
    })
    return df

def generate_ripple_wave(grid_size=40):
    """Generates a 3D ripple wave surface (Sinc function) with an angular 4th column scalar."""
    x_range = np.linspace(-5.0, 5.0, grid_size)
    y_range = np.linspace(-5.0, 5.0, grid_size)
    
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    
    r = np.sqrt(x_flat**2 + y_flat**2)
    # Avoid division by zero at the origin
    z_flat = np.where(r == 0, 2.0, 2.0 * np.sin(r) / r)
    
    # 4th column: scalar value C based on the angle theta around the z-axis (creates a spiral rainbow)
    # Combined with distance to add depth complexity
    theta = np.arctan2(y_flat, x_flat)
    val_c = theta * np.exp(-0.05 * r**2)
    
    df = pd.DataFrame({
        'X_Wave': x_flat,
        'Y_Wave': y_flat,
        'Z_Wave': z_flat,
        'Scalar_Value': val_c
    })
    return df

if __name__ == '__main__':
    print("Generating 3-column Lorenz Attractor dataset...")
    lorenz_df = generate_lorenz_attractor(1500)
    lorenz_df.to_csv('lorenz_attractor_3d.csv', index=False)
    print("Saved 'lorenz_attractor_3d.csv'")
    
    print("Generating 4-column Ripple Wave dataset...")
    ripple_df = generate_ripple_wave(45)
    ripple_df.to_csv('ripple_wave_4d.csv', index=False)
    print("Saved 'ripple_wave_4d.csv'")
    
    print("Sample datasets created successfully!")
