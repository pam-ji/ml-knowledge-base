import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    """
    A simple neural network with one hidden layer.
    
    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Number of neurons in the hidden layer
        output_dim (int): Number of output classes
    """
    def __init__(self, input_dim=2, hidden_dim=20, output_dim=3):
        super(SimpleNN, self).__init__()
        
        # Build a simple network with one hidden layer
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)
