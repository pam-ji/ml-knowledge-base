import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

def main():
    """
    Simple visualization of the vector cross product in 3D
    """
    st.subheader("3D Cross Product Visualization")
    
    # Interactive controls in the sidebar
    st.sidebar.header("Adjust Vectors")
    
    # Vector 1
    st.sidebar.subheader("Vector 1")
    v1_x = st.sidebar.slider("X component", -5.0, 5.0, 1.0, 0.1, key="v1x")
    v1_y = st.sidebar.slider("Y component", -5.0, 5.0, 2.0, 0.1, key="v1y")
    v1_z = st.sidebar.slider("Z component", -5.0, 5.0, 0.0, 0.1, key="v1z")
    
    # Vector 2
    st.sidebar.subheader("Vector 2")
    v2_x = st.sidebar.slider("X component", -5.0, 5.0, 2.0, 0.1, key="v2x")
    v2_y = st.sidebar.slider("Y component", -5.0, 5.0, 1.0, 0.1, key="v2y")
    v2_z = st.sidebar.slider("Z component", -5.0, 5.0, 0.0, 0.1, key="v2z")
    
    # Create vectors
    v1 = np.array([v1_x, v1_y, v1_z])
    v2 = np.array([v2_x, v2_y, v2_z])
    
    # Calculate cross product
    cross_product = np.cross(v1, v2)
    
    # Display vector information
    st.write("### Vector Values")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Vector 1:** [{v1_x}, {v1_y}, {v1_z}]")
        st.write(f"**Vector 2:** [{v2_x}, {v2_y}, {v2_z}]")
    
    with col2:
        st.write(f"**Cross Product:** [{cross_product[0]:.2f}, {cross_product[1]:.2f}, {cross_product[2]:.2f}]")
    
    # 3D view controls
    st.sidebar.subheader("3D View")
    elev = st.sidebar.slider("Elevation", -90, 90, 30, 5)
    azim = st.sidebar.slider("Azimuth", -180, 180, 30, 5)
    
    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Draw vectors as arrows
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='Vector 1')
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='Vector 2')
    ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2], 
             color='g', label='Cross Product')
    
    # Mark origin
    ax.scatter(0, 0, 0, color='black', s=50)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Cross Product Visualization')
    ax.legend()
    
    # Display plot in Streamlit
    st.pyplot(fig)
    
    # Mathematical explanation
    st.subheader("Cross Product Formula")
    st.latex(r"\vec{A} \times \vec{B} = |A||B|\sin(\theta)\hat{n}")
    
    # Component form
    st.latex(r"\vec{A} \times \vec{B} = \begin{pmatrix} A_y B_z - A_z B_y \\ A_z B_x - A_x B_z \\ A_x B_y - A_y B_x \end{pmatrix}")
    
    # Brief explanation
    st.write("""
    The cross product of two vectors:
    - Produces a vector that is perpendicular to both input vectors
    - The magnitude equals the area of the parallelogram formed by the two vectors
    - The direction follows the right-hand rule
    """)
    
    # Code example for the second tab
    st.subheader("Python Code Example")
    st.code("""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define vectors
v1 = np.array([1, 2, 0])
v2 = np.array([2, 1, 0])

# Calculate cross product
cross = np.cross(v1, v2)
print(f"Cross product: {cross}")

# Create 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Draw vectors
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='v2')
ax.quiver(0, 0, 0, cross[0], cross[1], cross[2], color='g', label='v1 × v2')

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
""")

# Direct execution (for testing)
if __name__ == "__main__":
    main()