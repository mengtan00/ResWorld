import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"from https://github.com/Kashu7100/TokenLearner/blob/main/model.py"

class MlpBlock(nn.Module):
    """Simple MLP block with GELU activation and dropout."""
    def __init__(self, input_dim, mlp_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x
    

class TokenLearner(nn.Module):
    """TokenLearner module Version 1.1 for PyTorch."""
    def __init__(self, num_tokens, in_channels, bottleneck_dim=64, dropout_rate=0.):
        super(TokenLearner, self).__init__()
        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.GroupNorm(1, in_channels, eps=1e-6)
        self.mlp = MlpBlock(input_dim=in_channels, mlp_dim=self.bottleneck_dim, output_dim=self.num_tokens, dropout_rate=self.dropout_rate)

    def forward(self, inputs, deterministic=True):
        """
        Args: 
            inputs: Inputs of shape `[B, HW, C]` or `[B, C, H, W]`.
            
        Returns:
            [B, num_token, C]
        """
        if inputs.dim() == 4:
            n, c, h, w = inputs.shape
            inputs = inputs.view(n, c, h * w).permute(0,2,1)

        selected = self.mlp(self.layer_norm(inputs.permute(0,2,1)).permute(0,2,1))

        # Softmax normalization
        # selected [B, num_token, HW]
        # correct here, should be permute 
        selected = selected.permute(0, 2, 1).softmax(dim=-1)

        # Weighted sum based on the selected tokens
        # feat [B, HW, C]
        feat = inputs.view(inputs.shape[0], -1, inputs.shape[-1])
        outputs = torch.einsum('bsi,bic->bsc', selected, feat)
        # outputs [B, num_token, C]
        return outputs, selected
    
class TokenFuser(nn.Module):
    """Token fusion module in PyTorch."""
    def __init__(self, num_tokens, in_channels, use_normalization=True, bottleneck_dim=64, dropout_rate=0.):
        super().__init__()
        self.use_normalization = use_normalization
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.norm = nn.GroupNorm(1, in_channels, eps=1e-6)
        self.norm1 = nn.GroupNorm(1, in_channels, eps=1e-6) if use_normalization else None
        self.norm2 = nn.GroupNorm(1, in_channels, eps=1e-6) if use_normalization else None
        self.dense = nn.Linear(in_features=num_tokens, out_features=bottleneck_dim)
        self.mlp = MlpBlock(input_dim=in_channels, mlp_dim=bottleneck_dim, output_dim=bottleneck_dim, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, original):
        """Applies token fusion to the generate 2D ouputs.

        Args:
            inputs: Inputs of shape `[B, n_token, C]`.
            original: Inputs of shape `[B, HW, C]` or `[B, C, H, W]`.

        Returns:
            Output tensor with the shape identical to `original'.
        """
        is_2d = False
        if original.dim() == 4:
            is_2d = True
            n, c, h, w = original.shape
            original = original.view(n, c, h * w).permute(0,2,1)

        if self.use_normalization:
            inputs = self.norm1(inputs.permute(0,2,1)).permute(0,2,1)

        # inputs [B, C, D]
        inputs = self.dense(inputs.permute(0,2,1))
        
        if self.use_normalization:
            inputs = self.norm2(inputs)

        original = self.norm(original.permute(0,2,1)).permute(0,2,1)
        # mix [B, HW, D]
        mix = self.mlp(original).sigmoid()
    
        # Using matrix multiplication for fusing tokens back
        fused = torch.einsum('bcs,bhs->bhc', inputs, mix)
        fused = self.dropout(fused)

        if is_2d:
            fused = fused.view(n, h, w, c).permute(0,3,1,2)

        return fused
