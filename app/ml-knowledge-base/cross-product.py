import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

def main():
    """
    Demonstrate the cross product of two vectors in 3D space.
    """
    st.subheader("Vector Cross Product Visualization")
    
    # Create two vectors
    v1 = np.array([1, 2, 0])
    v2 = np.array([2, 1, 0])
    
    # Calculate cross product
    cross_product = np.cross(v1, v2)
    
    # Display information
    st.write(f"Vector 1: {v1}")
    st.write(f"Vector 2: {v2}")
    st.write(f"Cross Product: {cross_product}")
    
    # Create a 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw vectors as arrows
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='Vector 1')
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='Vector 2')
    ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2], color='g', label='Cross Product')
    
    # Set plot limits and labels
    max_val = max(np.max(np.abs(v1)), np.max(np.abs(v2)), np.max(np.abs(cross_product)))
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
    
    # Show formula and explanation
    st.subheader("Cross Product Formula")
    st.latex(r"\vec{A} \times \vec{B} = |A||B|\sin(\theta)\hat{n}")
    st.write("The cross product results in a vector that is perpendicular to both input vectors.")
    st.write("Its magnitude is the area of the parallelogram formed by the two vectors.")

# Direct execution (for testing)
if __name__ == "__main__":
    main()