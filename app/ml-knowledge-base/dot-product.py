import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

def main():
    """
    Demonstrate the dot product in NumPy and PyTorch for data science applications
    """
    st.subheader("Vector Dot Product in Data Science")
    
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
    
    # Create vectors using NumPy and PyTorch
    np_v1 = np.array([v1_x, v1_y])
    np_v2 = np.array([v2_x, v2_y])
    
    torch_v1 = torch.tensor([v1_x, v1_y], dtype=torch.float)
    torch_v2 = torch.tensor([v2_x, v2_y], dtype=torch.float)
    
    # Calculate dot products
    np_dot = np.dot(np_v1, np_v2)
    torch_dot = torch.dot(torch_v1, torch_v2).item()
    
    st.write("## NumPy vs PyTorch Implementations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### NumPy")
        st.code("""
# Create NumPy vectors
v1 = np.array([{}, {}])
v2 = np.array([{}, {}])

# Calculate dot product
dot_product = np.dot(v1, v2)
print(f"Dot product: {dot_product:.2f}")
        """.format(v1_x, v1_y, v2_x, v2_y))
        
        st.write(f"Result: {np_dot:.2f}")
    
    with col2:
        st.write("### PyTorch")
        st.code("""
# Create PyTorch tensors
v1 = torch.tensor([{}, {}], dtype=torch.float)
v2 = torch.tensor([{}, {}], dtype=torch.float)

# Calculate dot product
dot_product = torch.dot(v1, v2)
print(f"Dot product: {dot_product:.2f}")
        """.format(v1_x, v1_y, v2_x, v2_y))
        
        st.write(f"Result: {torch_dot:.2f}")
    
    # Create a figure for visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot vectors
    ax.quiver(0, 0, np_v1[0], np_v1[1], angles='xy', scale_units='xy', 
             scale=1, color='red', label='Vector 1')
    ax.quiver(0, 0, np_v2[0], np_v2[1], angles='xy', scale_units='xy', 
             scale=1, color='blue', label='Vector 2')
    
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
    ax.set_title('Vector Representation')
    ax.set_aspect('equal')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Applications in ML
    st.write("## Applications in Machine Learning")
    
    st.write("""
    ### Feature Similarity
    In machine learning, dot products are used to measure similarity between feature vectors:
    
    ```python
    # Calculate similarity between two feature vectors
    similarity = np.dot(feature_vector1, feature_vector2)
    ```
    
    ### Neural Networks
    The basic operation in a neural network layer is a dot product between weights and inputs:
    
    ```python
    # Simple neural network layer
    def layer(x, weights, bias):
        return np.dot(weights, x) + bias
    ```
    
    ### Cosine Similarity
    Commonly used in NLP and recommender systems:
    
    ```python
    def cosine_similarity(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)
    ```
    """)
    
    # Performance comparison
    if st.checkbox("Show Performance Comparison"):
        st.write("## Performance Comparison: NumPy vs PyTorch")
        
        # Generate code for larger dot products
        arr_sizes = [10, 100, 1000, 10000]
        
        code = """
import numpy as np
import torch
import time

# Test different array sizes
sizes = [10, 100, 1000, 10000]

for size in sizes:
    # Create random vectors
    np_v1 = np.random.rand(size)
    np_v2 = np.random.rand(size)
    
    torch_v1 = torch.tensor(np_v1, dtype=torch.float)
    torch_v2 = torch.tensor(np_v2, dtype=torch.float)
    
    # NumPy timing
    start = time.time()
    for _ in range(1000):
        _ = np.dot(np_v1, np_v2)
    np_time = time.time() - start
    
    # PyTorch timing
    start = time.time()
    for _ in range(1000):
        _ = torch.dot(torch_v1, torch_v2)
    torch_time = time.time() - start
    
    print(f"Size {size}:")
    print(f"  NumPy: {np_time:.4f}s")
    print(f"  PyTorch: {torch_time:.4f}s")
    print(f"  Speedup: {np_time/torch_time:.2f}x")
"""
        st.code(code, language="python")
        
        st.info("PyTorch dot products can be significantly faster on GPU for large vectors")

# Direct execution (for testing)
if __name__ == "__main__":
    main()
