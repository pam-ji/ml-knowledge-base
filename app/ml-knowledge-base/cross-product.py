import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

def main():
    """
    Demonstrate the cross product of two vectors in 3D space with interactive controls.
    """
    st.subheader("Vector Cross Product Visualization")
    
    # Interactive widget section
    st.sidebar.header("Control Panel")
    st.sidebar.markdown("Adjust the vectors to see how the cross product changes")
    
    # Vector 1 controls
    st.sidebar.subheader("Vector 1")
    v1_x = st.sidebar.slider("X component", -5.0, 5.0, 1.0, 0.1, key="v1x")
    v1_y = st.sidebar.slider("Y component", -5.0, 5.0, 2.0, 0.1, key="v1y")
    v1_z = st.sidebar.slider("Z component", -5.0, 5.0, 0.0, 0.1, key="v1z")
    
    # Vector 2 controls
    st.sidebar.subheader("Vector 2")
    v2_x = st.sidebar.slider("X component", -5.0, 5.0, 2.0, 0.1, key="v2x")
    v2_y = st.sidebar.slider("Y component", -5.0, 5.0, 1.0, 0.1, key="v2y")
    v2_z = st.sidebar.slider("Z component", -5.0, 5.0, 0.0, 0.1, key="v2z")
    
    # Create vectors from user input
    v1 = np.array([v1_x, v1_y, v1_z])
    v2 = np.array([v2_x, v2_y, v2_z])
    
    # Calculate magnitude of vectors
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # Calculate cross product
    cross_product = np.cross(v1, v2)
    cross_mag = np.linalg.norm(cross_product)
    
    # Calculate the area of the parallelogram
    area = cross_mag
    
    # Calculate the angle between vectors (in radians)
    if v1_mag > 0 and v2_mag > 0:
        dot_product = np.dot(v1, v2)
        cos_angle = dot_product / (v1_mag * v2_mag)
        # Clip to handle floating point errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
    else:
        angle_deg = 0
    
    # Display vector information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Vector Values")
        st.write(f"**Vector 1:** [{v1_x}, {v1_y}, {v1_z}]")
        st.write(f"**Vector 2:** [{v2_x}, {v2_y}, {v2_z}]")
        st.write(f"**Cross Product:** [{cross_product[0]:.2f}, {cross_product[1]:.2f}, {cross_product[2]:.2f}]")
    
    with col2:
        st.write("### Calculation Results")
        st.write(f"**Cross Product Magnitude:** {cross_mag:.2f}")
        st.write(f"**Angle between vectors:** {angle_deg:.2f}°")
        st.write(f"**Area of parallelogram:** {area:.2f} square units")
        
        # Check if vectors are close to being parallel
        if cross_mag < 0.1 and v1_mag > 0 and v2_mag > 0:
            st.warning("Vectors are nearly parallel - cross product is close to zero vector")
    
    # View angle controls
    st.sidebar.subheader("3D View Controls")
    elev = st.sidebar.slider("Elevation", -90, 90, 30, 5)
    azim = st.sidebar.slider("Azimuth", -180, 180, 30, 5)
    
    # Create a 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set the viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Draw vectors as arrows
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='Vector 1')
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='Vector 2')
    ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2], color='g', label='Cross Product')
    
    # Draw the parallelogram (if vectors aren't too close to parallel)
    if cross_mag > 0.01:
        # Create the parallelogram vertices
        verts = np.array([
            [0, 0, 0],  # Origin
            [v1[0], v1[1], v1[2]],  # End of vector 1
            [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]],  # Corner point (v1 + v2)
            [v2[0], v2[1], v2[2]]   # End of vector 2
        ])
        
        # Plot the parallelogram with a semi-transparent face
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.2, color='purple')
    
    # Add origin point
    ax.scatter(0, 0, 0, color='black', s=50)
    
    # Set plot limits and labels
    max_val = max(np.max(np.abs(v1)), np.max(np.abs(v2)), np.max(np.abs(cross_product)), 0.1) + 0.5
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vector Cross Product Visualization')
    ax.legend()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Mathematical explanation
    st.subheader("Cross Product Formula")
    st.latex(r"\vec{A} \times \vec{B} = |A||B|\sin(\theta)\hat{n}")
    
    # Component form
    st.latex(r"\vec{A} \times \vec{B} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ A_x & A_y & A_z \\ B_x & B_y & B_z \end{vmatrix}")
    
    st.latex(r"\vec{A} \times \vec{B} = \begin{pmatrix} A_y B_z - A_z B_y \\ A_z B_x - A_x B_z \\ A_x B_y - A_y B_x \end{pmatrix}")
    
    # Conceptual explanation
    st.write("""
    The cross product has several important properties:
    - It produces a vector that is perpendicular to both input vectors
    - Its magnitude equals the area of the parallelogram formed by the two vectors
    - The direction follows the right-hand rule: if you curl the fingers of your right hand from the first vector toward the second, your thumb points in the direction of the cross product
    - If the vectors are parallel, the cross product is the zero vector
    
    Applications include:
    - Finding perpendicular vectors in 3D space
    - Calculating torque in physics
    - Computing surface normals in computer graphics
    - Determining area of a parallelogram
    """)

# Direct execution (for testing)
if __name__ == "__main__":
    main()