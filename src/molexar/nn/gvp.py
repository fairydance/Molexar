"""
GVP (Geometric Vector Perceptron) module for protein pocket geometry encoding.

Adapted from drorlab/gvp-pytorch for modern PyTorch compatibility.
Original: https://github.com/drorlab/gvp-pytorch

Key modifications for Molexar integration:
- Removed torch_geometric dependency for broader compatibility
- Added type hints and modern Python features
- Optimized for batch processing
- Compatible with PyTorch 2.x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import Tuple, Optional, Union

# Type aliases for GVP representations
ScalarFeatures = torch.Tensor  # [N, n_scalar]
VectorFeatures = torch.Tensor  # [N, n_vector, 3]
GVPFeatures = Tuple[ScalarFeatures, VectorFeatures]


def tuple_sum(*args: GVPFeatures) -> GVPFeatures:
    """Sums any number of tuples (s, V) elementwise."""
    s_list, v_list = zip(*args)
    # Sum all scalar features and all vector features separately
    return (sum(s_list), sum(v_list))


def tuple_cat(*args: GVPFeatures, dim: int = -1) -> GVPFeatures:
    """
    Concatenates any number of tuples (s, V) elementwise.

    Args:
        dim: dimension along which to concatenate when viewed
             as the `dim` index for the scalar-channel tensors.
             This means that `dim=-1` will be applied as
             `dim=-2` for the vector-channel tensors.
    """
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x: GVPFeatures, idx: torch.Tensor) -> GVPFeatures:
    """Indexes into a tuple (s, V) along the first dimension."""
    return x[0][idx], x[1][idx]


def _norm_no_nan(x: torch.Tensor, axis: int = -1, keepdims: bool = False, 
                  eps: float = 1e-8, sqrt: bool = True) -> torch.Tensor:
    """
    L2 norm of tensor clamped above a minimum value `eps`.
    
    Args:
        sqrt: if False, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x: torch.Tensor, nv: int) -> GVPFeatures:
    """
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)`.
    """
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v


def _merge(s: ScalarFeatures, v: VectorFeatures) -> torch.Tensor:
    """
    Merges a tuple (s, V) into a single torch.Tensor.
    Vector channels are flattened and appended to scalar channels.
    """
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)


class GVP(nn.Module):
    """
    Geometric Vector Perceptron.
    
    Processes scalar and vector features separately but jointly:
    - Scalar features: processed through standard MLP
    - Vector features: processed with geometric awareness (norms, rotations)
    
    Args:
        in_dims: tuple (n_scalar_in, n_vector_in)
        out_dims: tuple (n_scalar_out, n_vector_out)
        h_dim: intermediate number of vector channels (optional)
        activations: tuple (scalar_act, vector_act)
        vector_gate: use vector gating mechanism
    """
    
    def __init__(self, 
                 in_dims: Tuple[int, int], 
                 out_dims: Tuple[int, int],
                 h_dim: Optional[int] = None,
                 activations: Tuple = (F.relu, torch.sigmoid),
                 vector_gate: bool = False):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
    
    def forward(self, x: Union[GVPFeatures, torch.Tensor]) -> Union[GVPFeatures, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: tuple (s, V) or single tensor (if no vector input)
            
        Returns:
            tuple (s, V) or single tensor (if no vector output)
        """
        if self.vi:
            s, v = x
            # Transpose for matrix operations: [N, V, 3] -> [N, 3, V]
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device, dtype=s.dtype)
        
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    """Vector channel dropout - drops entire vector channels together."""
    
    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.dummy_param.device
        if not self.training:
            return x
        
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device, dtype=torch.float32)
        ).unsqueeze(-1).to(dtype=x.dtype)
        return mask * x / (1 - self.drop_rate)


class Dropout(nn.Module):
    """Combined dropout for tuples (s, V)."""
    
    def __init__(self, drop_rate: float):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)
    
    def forward(self, x: Union[GVPFeatures, torch.Tensor]) -> Union[GVPFeatures, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    """Combined LayerNorm for tuples (s, V)."""
    
    def __init__(self, dims: Tuple[int, int]):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
    
    def forward(self, x: Union[GVPFeatures, torch.Tensor]) -> Union[GVPFeatures, torch.Tensor]:
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class GVPConvLayer(nn.Module):
    """
    Graph convolution layer with GVPs for message passing.
    
    Args:
        node_dims: node embedding dimensions (n_scalar, n_vector)
        edge_dims: edge embedding dimensions (n_scalar, n_vector)
        n_message: number of GVPs in message function
        n_feedforward: number of GVPs in feedforward function
        drop_rate: dropout probability
        activations: tuple of activation functions
        vector_gate: use vector gating
    """
    
    def __init__(self,
                 node_dims: Tuple[int, int],
                 edge_dims: Tuple[int, int],
                 n_message: int = 3,
                 n_feedforward: int = 2,
                 drop_rate: float = 0.1,
                 activations: Tuple = (F.relu, torch.sigmoid),
                 vector_gate: bool = False):
        super().__init__()
        
        self.si, self.vi = node_dims
        self.so, self.vo = node_dims
        self.se, self.ve = edge_dims
        
        # Message function
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)
        
        if n_message == 1:
            self.message_func = GVP_(
                (2*self.si + self.se, 2*self.vi + self.ve),
                (self.so, self.vo),
                activations=(None, None)
            )
        else:
            modules = []
            modules.append(
                GVP_((2*self.si + self.se, 2*self.vi + self.ve), node_dims)
            )
            for _ in range(n_message - 2):
                modules.append(GVP_(node_dims, node_dims))
            modules.append(
                GVP_(node_dims, node_dims, activations=(None, None))
            )
            self.message_func = nn.Sequential(*modules)
        
        # Normalization and dropout
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])
        
        # Feedforward function
        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = (4*node_dims[0], 2*node_dims[1])
            ff_func.append(GVP_(node_dims, hid_dims))
            for _ in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
    
    def forward(self, 
                x: GVPFeatures,
                edge_index: torch.Tensor,
                edge_attr: GVPFeatures,
                node_mask: Optional[torch.Tensor] = None) -> GVPFeatures:
        """
        Forward pass with message passing.
        
        Args:
            x: node features (s, V)
            edge_index: [2, n_edges] tensor of source-target pairs
            edge_attr: edge features (s, V)
            node_mask: boolean mask for nodes to update
            
        Returns:
            Updated node features (s, V)
        """
        x_s, x_v = x
        src, dst = edge_index
        n_nodes = x_s.shape[0]

        # Flatten vector features for gathering
        x_v_flat = x_v.reshape(x_v.shape[0], 3 * x_v.shape[1])

        # Gather source (j) and destination (i) features per edge
        s_j = x_s[src]
        v_j = x_v_flat[src].view(-1, x_v.shape[1], 3)
        s_i = x_s[dst]
        v_i = x_v_flat[dst].view(-1, x_v.shape[1], 3)

        # Edge attributes are already per-edge — no indexing needed
        # Form message: (source, edge_attr, destination)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))

        # Apply message function
        message = self.message_func(message)

        # Merge for scatter aggregation
        message_merged = _merge(*message)

        # Mean aggregation at destination nodes
        agg = torch.zeros(n_nodes, message_merged.shape[-1], device=x_s.device, dtype=message_merged.dtype)
        agg.index_add_(0, dst, message_merged)
        count = torch.zeros(n_nodes, device=x_s.device, dtype=message_merged.dtype)
        count.index_add_(0, dst, torch.ones(dst.shape[0], device=x_s.device, dtype=message_merged.dtype))
        count = count.clamp(min=1).unsqueeze(-1)
        agg = agg / count

        # Split back to (s, V)
        dh = _split(agg, self.vo)
        
        # Apply mask if provided
        if node_mask is not None:
            x_orig = x
            x = tuple_index(x, node_mask)
            dh = tuple_index(dh, node_mask)
        
        # Residual connection + normalization
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        
        # Feedforward
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        
        # Restore mask
        if node_mask is not None:
            x_orig[0][node_mask], x_orig[1][node_mask] = x[0], x[1]
            x = x_orig
        
        return x


class GVPEncoder(nn.Module):
    """
    GVP encoder for protein pocket geometry.
    
    Takes atom coordinates and features, outputs pocket embedding.
    
    Args:
        node_in_dim: input node dimensions (scalar, vector)
        edge_in_dim: input edge dimensions (scalar, vector)
        node_h_dim: hidden node dimensions
        edge_h_dim: hidden edge dimensions
        n_layers: number of GVP layers
        drop_rate: dropout rate
    """
    
    def __init__(self,
                 node_in_dim: Tuple[int, int] = (6, 3),
                 edge_in_dim: Tuple[int, int] = (32, 1),
                 node_h_dim: Tuple[int, int] = (256, 16),
                 edge_h_dim: Tuple[int, int] = (32, 1),
                 n_layers: int = 3,
                 drop_rate: float = 0.1):
        super().__init__()
        
        # Initial projections
        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )
        
        # GVP layers
        self.layers = nn.ModuleList([
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(n_layers)
        ])
        
        # Output projection: GVP outputs scalar features
        ns, nv = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0))  # Output scalar features of node_h_dim[0]
        )
        
        self.output_dim = ns
    
    def forward(self,
                node_features: GVPFeatures,
                edge_index: torch.Tensor,
                edge_features: GVPFeatures,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            node_features: (scalar_features, vector_features)
            edge_index: [2, n_edges] tensor
            edge_features: (scalar_features, vector_features)
            batch: batch indices for graph-level pooling
            
        Returns:
            Pocket embedding [batch_size, output_dim]
        """
        h_V = self.W_v(node_features)
        h_E = self.W_e(edge_features)
        
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        
        # Global pooling
        out = self.W_out(h_V)  # [n_nodes, n_scalar]
        
        if batch is not None:
            # Batch-wise mean pooling
            batch_size = batch.max().item() + 1
            pooled = torch.zeros(batch_size, out.shape[1], device=out.device, dtype=out.dtype)
            pooled.index_add_(0, batch, out)
            count = torch.bincount(batch, minlength=batch_size).unsqueeze(-1).clamp(min=1).to(dtype=out.dtype)
            out = pooled / count
        
        return out





