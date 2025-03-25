import streamlit as st
import os
import sys
import importlib.util
from pathlib import Path
from typing import Optional, List

# Configure page
st.set_page_config(page_title="Vector Operations Visualizer", layout="wide")
st.title("Vector Operations Visualizer")

# Get list of Python files in ml-knowledge-base directory
tutorial_dir = Path("app/ml-knowledge-base")
tutorial_files: List[Path] = list(tutorial_dir.glob("*.py"))

# Create tabs
tab1, tab2 = st.tabs(["Interactive Visualization", "Source Code"])

# Create selectbox for tutorials
with tab1:
    selected_tutorial = st.selectbox(
        "Select a vector operation to visualize:",
        options=[f.stem for f in tutorial_files],
        format_func=lambda x: x.replace("-", " ").title()
    )
    
    # Find the correct file path for the selected tutorial
    selected_file: Optional[Path] = None
    for file in tutorial_files:
        if file.stem == selected_tutorial:
            selected_file = file
            break
    
    if selected_file:
        try:
            # Import the selected module
            spec = importlib.util.spec_from_file_location(
                selected_file.stem, 
                selected_file
            )
            if spec:  # Explicit check to handle None case
                module = importlib.util.module_from_spec(spec)
                sys.modules[selected_file.stem] = module
                if spec.loader:  # Explicit check to handle None case
                    spec.loader.exec_module(module)
                    
                    # If the module has a main function, run it
                    if hasattr(module, 'main'):
                        module.main()
                else:
                    st.error("Module loader is None, cannot execute module")
            else:
                st.error("Could not create module specification")
        except Exception as e:
            st.error(f"Error running the tutorial: {str(e)}")

# Display source code in second tab
with tab2:
    if selected_file:
        st.subheader(f"Source Code: {selected_file.name}")
        with open(selected_file, "r") as f:
            st.code(f.read(), language="python")
