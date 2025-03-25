#!/bin/bash
# Script to build and publish the vector-ops-tutorial package to PyPI

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we have the necessary tools
echo -e "${YELLOW}Checking prerequisites...${NC}"
python3 -m pip --version || { echo "pip is not installed!"; exit 1; }
python3 -m twine --version || { echo "Installing twine..."; pip install twine; }
python3 -m build --version || { echo "Installing build..."; pip install build; }

# Clean up previous builds
echo -e "${YELLOW}Cleaning up previous builds...${NC}"
rm -rf vector_ops_tutorial/dist/ vector_ops_tutorial/build/ vector_ops_tutorial/*.egg-info/

# Navigate to the package directory
cd vector_ops_tutorial

# Build the package
echo -e "${YELLOW}Building the package...${NC}"
python -m build

# Check the package
echo -e "${YELLOW}Checking the package with twine...${NC}"
python -m twine check dist/*

# Ask for PyPI credentials if not already set as environment variables
if [ -z "$TWINE_USERNAME" ]; then
    echo -e "${YELLOW}Enter your PyPI username:${NC}"
    read TWINE_USERNAME
    export TWINE_USERNAME
fi

if [ -z "$TWINE_PASSWORD" ]; then
    echo -e "${YELLOW}Enter your PyPI password:${NC}"
    read -s TWINE_PASSWORD
    export TWINE_PASSWORD
    echo
fi

# Confirm upload
echo -e "${YELLOW}Ready to upload to PyPI. Continue? (y/n)${NC}"
read confirm
if [ "$confirm" != "y" ]; then
    echo "Upload cancelled."
    exit 0
fi

# Upload to PyPI
echo -e "${YELLOW}Uploading to PyPI...${NC}"
python -m twine upload dist/*

# Success message
echo -e "${GREEN}Package successfully published to PyPI!${NC}"
echo -e "${GREEN}Install with: pip install vector-ops-tutorial${NC}"

# Return to original directory
cd ..

# Create an example notebook for Google Colab
echo -e "${YELLOW}Creating example Colab notebook...${NC}"

cat > vector_ops_tutorial_colab.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Vector Operations Tutorial in Google Colab\n",
    "\n",
    "This notebook demonstrates how to use the vector-ops-tutorial package in Google Colab."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup\n",
    "\n",
    "First, let's install the necessary packages:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install the vector-ops-tutorial package\n",
    "!pip install vector-ops-tutorial"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Running the App in Colab\n",
    "\n",
    "To run Streamlit apps in Google Colab, we need to use a tunnel. Let's use [localtunnel](https://github.com/localtunnel/localtunnel)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install localtunnel\n",
    "!npm install -g localtunnel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Write a script to run the Streamlit app\n",
    "%%writefile run_streamlit_app.py\n",
    "from vector_ops_tutorial.app import run_app\n",
    "\n",
    "# Run the app on port 8501 (default Streamlit port)\n",
    "run_app()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run the Streamlit app and create a tunnel\n",
    "# This will print a URL you can click to view the app\n",
    "import subprocess\n",
    "import threading\n",
    "import time\n",
    "import IPython.display as display\n",
    "\n",
    "def run_streamlit():\n",
    "    subprocess.run([\"streamlit\", \"run\", \"run_streamlit_app.py\", \"--server.port=8501\", \"--server.address=127.0.0.1\"])\n",
    "\n",
    "# Start Streamlit in a separate thread\n",
    "streamlit_thread = threading.Thread(target=run_streamlit)\n",
    "streamlit_thread.daemon = True\n",
    "streamlit_thread.start()\n",
    "\n",
    "# Wait for Streamlit to initialize\n",
    "time.sleep(5)\n",
    "\n",
    "# Start localtunnel to expose the app\n",
    "lt_process = subprocess.Popen([\"lt\", \"--port\", \"8501\"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n",
    "\n",
    "# Extract the URL\n",
    "for line in lt_process.stdout:\n",
    "    if \"your url is:\" in line.lower():\n",
    "        url = line.strip().split(\":\")[-1].strip()\n",
    "        display.display(display.HTML(f'<p>Click the link to open the Vector Operations Tutorial: <a href=\"{url}\" target=\"_blank\">{url}</a></p>'))\n",
    "        break\n",
    "\n",
    "# Keep the cell running\n",
    "try:\n",
    "    while True:\n",
    "        time.sleep(1)\n",
    "except KeyboardInterrupt:\n",
    "    lt_process.terminate()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Manual Vector Operations\n",
    "\n",
    "If you prefer not to use the Streamlit interface, you can also use the underlying functions directly:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "\n",
    "# Define two vectors\n",
    "v1 = np.array([3, 2, 1])\n",
    "v2 = np.array([1, 0, 2])\n",
    "\n",
    "# Calculate dot product\n",
    "dot_product = np.dot(v1, v2)\n",
    "print(f\"Dot product: {dot_product}\")\n",
    "\n",
    "# Calculate cross product\n",
    "cross_product = np.cross(v1, v2)\n",
    "print(f\"Cross product: {cross_product}\")\n",
    "\n",
    "# Calculate angle between vectors\n",
    "magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)\n",
    "angle = np.arccos(np.clip(dot_product / magnitude_product, -1.0, 1.0)) * 180 / np.pi\n",
    "print(f\"Angle between vectors: {angle:.2f} degrees\")\n",
    "\n",
    "# Visualize in 3D\n",
    "fig = plt.figure(figsize=(10, 8))\n",
    "ax = fig.add_subplot(111, projection='3d')\n",
    "\n",
    "# Draw vectors\n",
    "ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='Vector 1')\n",
    "ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='Vector 2')\n",
    "ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2], color='g', label='Cross Product')\n",
    "\n",
    "# Set axis limits\n",
    "all_coords = np.vstack((v1, v2, cross_product))\n",
    "max_range = np.max(np.abs(all_coords)) * 1.2\n",
    "ax.set_xlim([-max_range, max_range])\n",
    "ax.set_ylim([-max_range, max_range])\n",
    "ax.set_zlim([-max_range, max_range])\n",
    "\n",
    "# Labels\n",
    "ax.set_xlabel('X')\n",
    "ax.set_ylabel('Y')\n",
    "ax.set_zlabel('Z')\n",
    "ax.legend()\n",
    "\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo -e "${GREEN}Example Colab notebook created: vector_ops_tutorial_colab.ipynb${NC}"
echo -e "${GREEN}All done!${NC}"