import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
import torch

def main():
    """
    Demonstrate the cross product in NumPy and PyTorch for Machine Learning applications
    """
    st.subheader("3D Cross Product in Machine Learning")
    
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
    
    # Create vectors using NumPy and PyTorch
    np_v1 = np.array([v1_x, v1_y, v1_z])
    np_v2 = np.array([v2_x, v2_y, v2_z])
    
    torch_v1 = torch.tensor([v1_x, v1_y, v1_z], dtype=torch.float)
    torch_v2 = torch.tensor([v2_x, v2_y, v2_z], dtype=torch.float)
    
    # Calculate cross products
    np_cross = np.cross(np_v1, np_v2)
    torch_cross = torch.cross(torch_v1, torch_v2).numpy()
    
    st.write("## NumPy vs PyTorch Implementations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### NumPy")
        st.code("""
# Create NumPy vectors
v1 = np.array([{}, {}, {}])
v2 = np.array([{}, {}, {}])

# Calculate cross product
cross_product = np.cross(v1, v2)
print(f"Cross product: {cross_product}")
        """.format(v1_x, v1_y, v1_z, v2_x, v2_y, v2_z))
        
        st.write(f"Result: [{np_cross[0]:.2f}, {np_cross[1]:.2f}, {np_cross[2]:.2f}]")
    
    with col2:
        st.write("### PyTorch")
        st.code("""
# Create PyTorch tensors
v1 = torch.tensor([{}, {}, {}], dtype=torch.float)
v2 = torch.tensor([{}, {}, {}], dtype=torch.float)

# Calculate cross product
cross_product = torch.cross(v1, v2)
print(f"Cross product: {cross_product}")
        """.format(v1_x, v1_y, v1_z, v2_x, v2_y, v2_z))
        
        st.write(f"Result: [{torch_cross[0]:.2f}, {torch_cross[1]:.2f}, {torch_cross[2]:.2f}]")
    
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
    ax.quiver(0, 0, 0, np_v1[0], np_v1[1], np_v1[2], color='r', label='Vector 1')
    ax.quiver(0, 0, 0, np_v2[0], np_v2[1], np_v2[2], color='b', label='Vector 2')
    ax.quiver(0, 0, 0, np_cross[0], np_cross[1], np_cross[2], color='g', label='Cross Product')
    
    # Add origin point
    ax.scatter(0, 0, 0, color='black', s=50)
    
    # Set plot limits and labels
    max_val = max(np.max(np.abs(np_v1)), np.max(np.abs(np_v2)), np.max(np.abs(np_cross)), 0.1) + 0.5
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
    
    # ML Applications
    st.write("## Applications in Machine Learning and Computer Vision")
    
    st.write("""
    ### 3D Computer Vision
    In computer vision, cross products are used to compute surface normals and camera orientations:
    
    ```python
    # Calculate normal vector to a 3D surface using 3 points
    def compute_normal(p1, p2, p3):
        # Create vectors along the surface
        v1 = p2 - p1
        v2 = p3 - p1
        # Normal vector is perpendicular to both v1 and v2
        normal = np.cross(v1, v2)
        # Normalize to unit length
        return normal / np.linalg.norm(normal)
    ```
    
    ### Feature Engineering
    Cross products can be used to create new features that capture relationships between vectors:
    
    ```python
    # Create new features from pairs of 3D vectors
    def create_cross_features(vectors):
        features = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                # Add cross product as a new feature
                cross = np.cross(vectors[i], vectors[j])
                features.append(cross)
        return np.array(features)
    ```
    
    ### Robotics and Reinforcement Learning
    Cross products are essential in robotics applications like calculating torque and angular momentum:
    
    ```python
    # Calculate torque in a robotic joint
    def calculate_torque(force_vector, position_vector):
        return np.cross(position_vector, force_vector)
    ```
    """)
    
    # Add 3D rotations with PyTorch code
    if st.checkbox("Show 3D Rotations with PyTorch"):
        st.write("## 3D Rotations in PyTorch")
        
        st.code("""
import torch
from pytorch3d.transforms import axis_angle_to_matrix

# Create rotation vectors (axis * angle)
def create_rotation_matrix(axis, angle_degrees):
    # Normalize axis
    axis = axis / torch.norm(axis)
    # Convert degrees to radians
    angle_rad = torch.tensor(angle_degrees * 3.14159 / 180)
    # Create axis-angle representation (axis * angle)
    axis_angle = axis * angle_rad
    # Convert to rotation matrix
    rotation_matrix = axis_angle_to_matrix(axis_angle)
    return rotation_matrix

# Example usage
axis = torch.tensor([0., 0., 1.])  # Rotation around Z axis
angle = 45.0  # degrees
R = create_rotation_matrix(axis, angle)
print(f"Rotation matrix:\\n{R}")

# Apply rotation to a point
point = torch.tensor([1., 0., 0.])
rotated_point = torch.matmul(R, point)
print(f"Original point: {point}")
print(f"Rotated point: {rotated_point}")
""")
        
        st.info("Cross products are used internally in the calculation of 3D rotations!")

# Direct execution (for testing)
if __name__ == "__main__":
    main()