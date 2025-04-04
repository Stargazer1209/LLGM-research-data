import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GraphPeriodicityProjector(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, with_gate=True):
        super(GraphPeriodicityProjector, self).__init__()
        
        # Replace linear layers with learnable parameters
        self.periodicity_weight = nn.Parameter(torch.randn(input_dim, output_dim//4))
        self.graph_feature_weight = nn.Parameter(torch.randn(input_dim, (output_dim-output_dim//2)))
        
        if bias:
            self.periodicity_bias = nn.Parameter(torch.zeros(output_dim//4))
        else:
            self.register_parameter('periodicity_bias', None)
        
        self.graph_feature_bias = nn.Parameter(torch.zeros((output_dim-output_dim//2)))
        self.activation = nn.GELU()
        
        if with_gate:
            self.periodicity_gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
    
    def forward(self, input_features):
        # Manual linear transformations
        periodic_features = F.linear(input_features, self.periodicity_weight, self.periodicity_bias)
        graph_features = self.activation(F.linear(input_features, self.graph_feature_weight, self.graph_feature_bias))
        
        has_gate = hasattr(self, 'periodicity_gate')
        gate_value = torch.sigmoid(self.periodicity_gate) if has_gate else 1.0
        output = torch.cat([gate_value * torch.cos(periodic_features), gate_value * torch.sin(periodic_features), 
                           ((1-gate_value) if has_gate else 1.0) * graph_features], dim=-1)
        return output