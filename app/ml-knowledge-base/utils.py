import numpy as np
import matplotlib.pyplot as plt
import torch

def generate_spiral_data(n_samples=100, noise=0.2, n_classes=3):
    """
    Generate synthetic spiral dataset.
    
    Args:
        n_samples (int): Number of samples per class
        noise (float): Amount of noise to add to the data
        n_classes (int): Number of spiral classes/arms
        
    Returns:
        X (np.ndarray): Features of shape (n_samples * n_classes, 2)
        y (np.ndarray): Labels of shape (n_samples * n_classes,)
    """
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype=int)
    
    for j in range(n_classes):
        ix = range(n_samples * j, n_samples * (j + 1))
        r = np.linspace(0.0, 1, n_samples)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, n_samples) + np.random.randn(n_samples) * noise  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
        
    return X, y

def plot_decision_boundary(model, X, y):
    """
    Plot decision boundary for a 2D dataset.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model
        X (np.ndarray): Features of shape (n_samples, 2)
        y (np.ndarray): Labels
    """
    # Set min and max values for grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Generate a mesh grid
    h = 0.02  # mesh step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Create tensor of mesh points
    mesh_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    # Make predictions for each point in the mesh
    with torch.no_grad():
        Z = model(mesh_points)
        _, Z = torch.max(Z, 1)
        Z = Z.numpy()
    
    # Reshape to match the mesh grid
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.7)
    
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, edgecolors='k')

def plot_loss_curve(loss_history):
    """
    Plot the training loss curve.
    
    Args:
        loss_history (list): List of loss values from training
    """
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
