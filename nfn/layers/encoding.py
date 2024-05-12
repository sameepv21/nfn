import math
import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange
from nfn.common import WeightSpaceFeatures, NetworkSpec
import math

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module for encoding weights and biases.
    """

    def __init__(self, network_spec, hidden_size, channels, num_layers=6, num_heads=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.network_spec = network_spec
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.channels = channels
        self.dropout = dropout

        # Calculate input size based on network_spec
        self.input_size, _ = network_spec.get_io()

        # Define layers
        self.pos_encoder = PositionalEncoding(d_model=channels, dropout=dropout, max_len=1024)
        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(channels, hidden_size)

    def _calculate_input_size(self, network_spec):
        # Calculate input size based on the number of input channels in the network specification
        input_size = sum(network_spec)
        return input_size

    def forward(self, wsfeat):
        """
        Forward pass of the encoder.
        
        Args:
        - wsfeat (WeightSpaceFeatures): Input WeightSpaceFeatures object containing weights and biases.

        Returns:
        - encoded_wsfeat (WeightSpaceFeatures): Encoded WeightSpaceFeatures object.
        """
        out_weights, out_bias = [], []
        for i in range(len(self.network_spec)):
            # Fetch weights and biases
            weight, bias = wsfeat[i]

            # Preprocess and get the corrected shape
            weight = self._correct_dim(weight, index = i, is_weight = True)
            bias = self._correct_dim(bias, index = i, is_weight = False)

            # Encode weights
            encoded_weights = self._encode_tensor(weight)
            
            # Encode biases
            encoded_bias = self._encode_tensor(bias)

            # Correct and rearrange the encoded weights and bias
            final_weight_rearrange = Rearrange("h w bs embed_dim -> bs embed_dim h w")
            final_bias_rearrange = Rearrange("h bs embed_dim -> bs embed_dim h")

            encoded_weights = torch.reshape(encoded_weights, (self.network_spec.weight_spec[i].shape[0], self.network_spec.weight_spec[i].shape[1], 4, 256))
            encoded_weights = final_weight_rearrange(encoded_weights)

            encoded_bias = torch.reshape(encoded_bias, (self.network_spec.bias_spec[i].shape[0], 4, 256))
            encoded_bias = final_bias_rearrange(encoded_bias)

            # encoded_weights = final_rearrange(encoded_weights)
            # encoded_bias = final_rearrange(encoded_bias)

            # Append in the list
            out_weights.append(encoded_weights)
            out_bias.append(encoded_bias)
        return WeightSpaceFeatures(out_weights, out_bias)

    def _correct_dim(self, x, index, is_weight):
        # conv weight filter dims.
        filter_dims = (None,) * (x.ndim - 4)

        # Defing embedding look ups for weights and biases
        weight_emb = nn.Embedding(len(self.network_spec), self.channels)
        bias_emb = nn.Embedding(len(self.network_spec), self.channels)
        
        # Define post-rearrange term for weights and bias
        weight_emb_rearrange = Rearrange("bs embed_dim out_neuron curr_neuron -> (out_neuron curr_neuron) bs embed_dim")
        bias_emb_rearrange = Rearrange("bs embed_dim curr_neuron -> curr_neuron bs embed_dim")
        
        # If weight is being managed
        if is_weight:
            x = x + weight_emb.weight[index][(None, Ellipsis, None, None, *filter_dims)]
            # returned_weight = weight_emb_rearrange(x)
            return weight_emb_rearrange(x)
        
        x = x + bias_emb.weight[index][None, :, None]
        returned_bias = bias_emb_rearrange(x)
        return bias_emb_rearrange(x)

    def _encode_tensor(self, tensor):
        # Perform positional encoding
        tensor = self.pos_encoder(tensor)
        # Apply transformer encoder
        encoded_tensor = self.transformer_encoder(tensor)
        # Linear projection
        encoded_tensor = self.linear(encoded_tensor)  # Pooling over sequence dimension
        return encoded_tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class GaussianFourierFeatureTransform(nn.Module):
    """
    Given an input of size [batches, num_input_channels, ...],
     returns a tensor of size [batches, mapping_size*2, ...].
    """

    def __init__(self, network_spec, in_channels, mapping_size=256, scale=10):
        super().__init__()
        self.network_spec = network_spec
        self.in_channels = in_channels
        self._mapping_size = mapping_size
        self.out_channels = mapping_size * 2
        self.scale = scale
        self.register_buffer("_B", torch.randn((in_channels, mapping_size)) * scale)

    def encode_tensor(self, x):
        # Put channels dimension last.
        x = (x.transpose(1, -1) @ self._B).transpose(1, -1)
        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

    def forward(self, wsfeat):
        out_weights, out_biases = [], []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            out_weights.append(self.encode_tensor(weight))
            out_biases.append(self.encode_tensor(bias))
        return WeightSpaceFeatures(out_weights, out_biases)

    def __repr__(self):
        return f"GaussianFourierFeatureTransform(in_channels={self.in_channels}, mapping_size={self._mapping_size}, scale={self.scale})"


def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class IOSinusoidalEncoding(nn.Module):
    def __init__(self, network_spec: NetworkSpec, max_freq=10, num_bands=6, enc_layers=True):
        super().__init__()
        self.network_spec = network_spec
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.enc_layers = enc_layers
        self.n_in, self.n_out = network_spec.get_io()

    def forward(self, wsfeat: WeightSpaceFeatures):
        device, dtype = wsfeat.weights[0].device, wsfeat.weights[0].dtype
        L = len(self.network_spec)
        layernum = torch.linspace(-1., 1., steps=L, device=device, dtype=dtype)
        if self.enc_layers:
            layer_enc = fourier_encode(layernum, self.max_freq, self.num_bands)  # (L, 2 * num_bands + 1)
        else:
            layer_enc = torch.zeros((L, 2 * self.num_bands + 1), device=device, dtype=dtype)
        inpnum = torch.linspace(-1., 1., steps=self.n_in, device=device, dtype=dtype)
        inp_enc = fourier_encode(inpnum, self.max_freq, self.num_bands)  # (n_in, 2 * num_bands + 1)
        outnum = torch.linspace(-1., 1., steps=self.n_out, device=device, dtype=dtype)
        out_enc = fourier_encode(outnum, self.max_freq, self.num_bands)  # (n_out, 2 * num_bands + 1)

        d = 2 * self.num_bands + 1

        out_weights, out_biases = [], []
        for i in range(L):
            weight, bias = wsfeat[i]
            b, _, *axes = weight.shape
            enc_i = layer_enc[i].unsqueeze(0)[..., None, None]
            for _ in axes[2:]:
                enc_i = enc_i.unsqueeze(-1)
            enc_i = enc_i.expand(b, d, *axes) # (B, d, n_row, n_col, ...)
            bias_enc_i = layer_enc[i][None, :, None].expand(b, d, bias.shape[-1])  # (B, d, n_row)
            if i == 0:
                # weight has shape (B, c_in, n_out, n_in)
                inp_enc_i = inp_enc.transpose(0, 1).unsqueeze(0).unsqueeze(-2)  # (1, d, 1, n_col)
                for _ in axes[2:]:
                    inp_enc_i = inp_enc_i.unsqueeze(-1)
                enc_i = enc_i  + inp_enc_i
            if i == len(wsfeat) - 1:
                out_enc_i = out_enc.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # (1, d, n_row, 1)
                for _ in axes[2:]:
                    out_enc_i = inp_enc_i.unsqueeze(-1)
                enc_i = enc_i  + out_enc_i
                bias_enc_i = bias_enc_i + out_enc.transpose(0, 1).unsqueeze(0)
            out_weights.append(torch.cat([weight, enc_i], dim=1))
            out_biases.append(torch.cat([bias, bias_enc_i], dim=1))
        return WeightSpaceFeatures(out_weights, out_biases)

    def num_out_chan(self, in_chan):
        return in_chan + (2 * self.num_bands + 1)


class LearnedPosEmbedding(nn.Module):
    def __init__(self, network_spec: NetworkSpec, channels):
        super().__init__()
        self.channels = channels
        self.network_spec = network_spec
        length = len(network_spec)
        self.weight_emb = nn.Embedding(len(network_spec), channels)
        self.bias_emb = nn.Embedding(len(network_spec), channels)
        num_inp, num_out = network_spec.get_io()
        self.inp_emb = nn.Embedding(num_inp, channels)
        self.out_emb = nn.Embedding(num_out, channels)
        self.inp_weight_arrange = Rearrange("n_in c -> 1 c 1 n_in")
        self.out_weight_arrange = Rearrange("n_out c -> 1 c n_out 1")
        self.out_bias_arrange = Rearrange("n_out c -> 1 c n_out")

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        out_weights, out_biases = [], []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            filter_dims = (None,) * (weight.ndim - 4)  # conv weight filter dims.
            temp = self.weight_emb.weight[i][(None, Ellipsis, None, None, *filter_dims)]
            weight = weight + self.weight_emb.weight[i][(None, Ellipsis, None, None, *filter_dims)]
            bias = bias + self.bias_emb.weight[i][None, :, None]
            if i == 0:
                weight = weight + self.inp_weight_arrange(self.inp_emb.weight)[(Ellipsis, *filter_dims)]
            if i == len(wsfeat.weights) - 1:
                weight = weight + self.out_weight_arrange(self.out_emb.weight)[(Ellipsis, *filter_dims)]
                bias = bias + self.out_bias_arrange(self.out_emb.weight)
            out_weights.append(weight)
            out_biases.append(bias)
        return WeightSpaceFeatures(tuple(out_weights), tuple(out_biases))
    def __repr__(self):
        return f"LearnedPosEmbedding(channels={self.channels})"