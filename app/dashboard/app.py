
import streamlit as st
import os
import sys
import importlib.util
from pathlib import Path

# Configure page
st.set_page_config(page_title="ML Knowledge Base", layout="wide")
st.title("ML Knowledge Base Tutorials")

# Get list of Python files in ml-knowledge-base directory
tutorial_dir = Path("ml-knowledge-base")
tutorial_files = list(tutorial_dir.glob("*.py"))

# Create selectbox for tutorials
selected_file = st.selectbox(
    "Select a tutorial:",
    options=tutorial_files,
    format_func=lambda x: x.stem
)

if selected_file:
    st.subheader(f"Content of {selected_file.name}")
    
    # Display file content
    with open(selected_file, "r") as f:
        st.code(f.read(), language="python")
    
    # Add a button to run the selected file
    if st.button(f"Run {selected_file.name}"):
        st.write("Output:")
        
        try:
            # Import and run the selected module
            spec = importlib.util.spec_from_file_location(
                selected_file.stem, 
                selected_file
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[selected_file.stem] = module
            spec.loader.exec_module(module)
            
            # If the module has a main function, run it
            if hasattr(module, 'main'):
                module.main()
        except Exception as e:
            st.error(f"Error running the tutorial: {str(e)}")
