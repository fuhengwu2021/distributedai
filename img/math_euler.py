import matplotlib.pyplot as plt
import numpy as np

def draw_complex_spiral():
    # 1. Setup the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set background style to match the image (gradient-like look)
    ax.set_facecolor('#f0f0f0') # Light grey top
    # Draw a darker grey rectangle for the bottom half to create the "horizon"
    ground = plt.Rectangle((-2, -2), 4, 2, color='#d9d9d9', zorder=0)
    ax.add_patch(ground)
    
    # 2. Define the Mathematical Sequence
    # The sequence represents powers of (1 + i*pi/9)
    # z_0 = 1
    # z_1 = 1 + i*pi/9
    # ...
    # z_9 = (1 + i*pi/9)^9
    
    base = 1 + 1j * (np.pi / 9)
    steps = 10  # From power 0 to 9
    points = [base**i for i in range(steps)]
    points = np.array(points)

    # 3. Draw the Unit Circle (for reference)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, color='black', linewidth=0.8, alpha=0.5, zorder=1)

    # 4. Draw the Spiral (Red Triangles)
    # Each segment is a triangle connected to the origin (0,0)
    origin = np.array([0, 0])
    
    for i in range(len(points) - 1):
        # Define vertices for the current triangle
        # (Origin -> Point i -> Point i+1)
        triangle_cnt = np.array([
            [0, 0],
            [points[i].real, points[i].imag],
            [points[i+1].real, points[i+1].imag]
        ])
        
        # Fill the triangle
        poly = plt.Polygon(triangle_cnt, facecolor='#ff3322', edgecolor='black', 
                           linewidth=0.8, alpha=0.9, zorder=2)
        ax.add_patch(poly)

    # 5. Plot the Specific Dots
    # Point at 1 (Blue)
    ax.scatter(points[0].real, points[0].imag, color='blue', s=80, zorder=3, edgecolors='white', linewidth=0.5)
    
    # Final Point at power 9 (Green)
    final_p = points[-1]
    ax.scatter(final_p.real, final_p.imag, color='#00ff80', s=80, zorder=3, edgecolors='white', linewidth=0.5)
    
    # Reference point at -1 (Orange)
    ax.scatter(-1, 0, color='#ffaa44', s=80, zorder=3, edgecolors='white', linewidth=0.5)
    
    # Origin dot (Small black)
    ax.scatter(0, 0, color='black', s=20, zorder=3)

    # 6. Add Text Labels (using LaTeX)
    
    # Label: -1
    ax.text(-1.15, -0.15, r'$-1$', fontsize=16, color='black')
    
    # Label: 0
    ax.text(0, -0.15, r'$0$', fontsize=16, color='black', ha='center')
    
    # Label: 1
    ax.text(1.05, -0.15, r'$1$', fontsize=16, color='black')

    # Label: Right side equation
    # Positioned near the first step
    ax.text(1.2, 0.4, r'$1 + \frac{i \pi}{9}$', fontsize=22, color='black')

    # Label: Left side equation (Final result)
    # Positioned near the green dot
    ax.text(-1.8, 0.6, r'$\left(1 + \frac{i \pi}{9}\right)^9$', fontsize=22, color='black')

    # 7. Final Formatting
    ax.axhline(0, color='black', linewidth=1, alpha=0.3) # Horizon line
    
    # Set limits to frame the spiral nicely
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-0.8, 1.5)
    
    # Remove standard axes spines/ticks for a cleaner "geometry" look
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Draw the main horizon line explicitly (stronger black line)
    ax.plot([-3, 3], [0, 0], color='black', linewidth=1, zorder=1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_complex_spiral()