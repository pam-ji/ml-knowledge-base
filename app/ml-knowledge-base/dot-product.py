import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def main():
    """
    Demonstrate the dot product of two vectors in 2D space with interactive sliders.
    """
    st.subheader("Vector Dot Product Visualization")
    
    # Interactive widget section
    st.sidebar.header("Control Panel")
    st.sidebar.markdown("Adjust the vectors to see how the dot product changes")
    
    # Vector 1 controls
    st.sidebar.subheader("Vector 1")
    v1_x = st.sidebar.slider("X component", -5.0, 5.0, 3.0, 0.1)
    v1_y = st.sidebar.slider("Y component", -5.0, 5.0, 2.0, 0.1)
    
    # Vector 2 controls
    st.sidebar.subheader("Vector 2")
    v2_x = st.sidebar.slider("X component", -5.0, 5.0, 2.0, 0.1)
    v2_y = st.sidebar.slider("Y component", -5.0, 5.0, 1.0, 0.1)
    
    # Create vectors from user input
    v1 = np.array([v1_x, v1_y])
    v2 = np.array([v2_x, v2_y])
    
    # Calculate magnitude of vectors
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # Calculate dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate the angle between vectors (in radians)
    if v1_mag > 0 and v2_mag > 0:
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
        st.write(f"**Vector 1:** [{v1_x}, {v1_y}]")
        st.write(f"**Vector 2:** [{v2_x}, {v2_y}]")
    
    with col2:
        st.write("### Calculation Results")
        st.write(f"**Dot Product:** {dot_product:.2f}")
        st.write(f"**Angle between vectors:** {angle_deg:.2f}°")
        
        # Indicate if vectors are orthogonal or parallel
        if abs(dot_product) < 0.01:
            st.success("Vectors are approximately orthogonal (perpendicular)")
        elif abs(abs(cos_angle) - 1) < 0.01:
            st.success("Vectors are approximately parallel")
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot vectors
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector 1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector 2')
    
    # Plot dot product projection
    if v1_mag > 0:
        # Calculate projection of v2 onto v1
        proj_scalar = dot_product / (v1_mag * v1_mag)
        proj_vector = proj_scalar * v1
        ax.quiver(0, 0, proj_vector[0], proj_vector[1], angles='xy', scale_units='xy', 
                 scale=1, color='g', linestyle='--', label='Projection')
    
    # Add origin point
    ax.scatter(0, 0, color='black', s=50)
    
    # Set plot limits and labels
    max_val = max(abs(v1_x), abs(v1_y), abs(v2_x), abs(v2_y), 1) + 0.5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.legend()
    ax.set_title('Vector Dot Product Visualization')
    ax.set_aspect('equal')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Mathematical explanation
    st.subheader("Dot Product Formula")
    st.latex(r"A \cdot B = |A| |B| \cos(\theta)")
    st.latex(r"A \cdot B = A_x \times B_x + A_y \times B_y")
    
    # Conceptual explanation
    st.write("""
    The dot product has several interpretations:
    - It measures how parallel two vectors are to each other
    - It equals the product of the magnitudes of the vectors and the cosine of the angle between them
    - It equals the sum of the products of the corresponding components
    - It represents the projection of one vector onto another multiplied by the magnitude of the second vector
    """)

# Direct execution (for testing)
if __name__ == "__main__":
    main()
