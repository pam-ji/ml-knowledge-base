import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def main():
    """
    Demonstrate the dot product of two vectors in 2D space.
    """
    st.subheader("Vector Dot Product Visualization")
    
    # Create two vectors
    v1 = np.array([3, 2])
    v2 = np.array([2, 1])
    
    # Calculate dot product
    dot_product = np.dot(v1, v2)
    
    # Display information
    st.write(f"Vector 1: {v1}")
    st.write(f"Vector 2: {v2}")
    st.write(f"Dot Product: {dot_product}")
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot vectors
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector 1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector 2')
    
    # Set plot limits and labels
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.legend()
    ax.set_title('Vector Dot Product Visualization')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Show formula and explanation
    st.subheader("Dot Product Formula")
    st.latex(r"A \cdot B = |A| |B| \cos(\theta)")
    st.write("The dot product is a scalar value that represents how parallel two vectors are to each other.")

# Direct execution (for testing)
if __name__ == "__main__":
    main()
