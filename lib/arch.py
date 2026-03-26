import torch
import torch.nn as nn
import torch.nn.functional as F
from .odst import ODST


class DenseBlock(nn.Sequential):
    def __init__(self, input_dim, layer_dim, num_layers, tree_dim=1, max_features=None,
                 input_dropout=0.0, flatten_output=True, Module=ODST, **kwargs):
        #input_dim -> number of input features
        #layer_dim -> number of output features per tree in each layer
        #num_layers -> number of layers in the block
        #tree_dim -> number of output features per tree (default 1, i.e. scalar output)
        #max_features -> maximum number of features to use in each layer (default None, i.e. use all features)
        #input_dropout -> dropout rate for input features (default 0.0, i.e. no dropout)
        #flatten_output -> whether to flatten the output of the block (default True, i.e. output shape [..., num_layers * layer_dim * tree_dim])
        #Module -> the module to use for each layer (default ODST)
        layers = []
        for i in range(num_layers):
            oddt = Module(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True, **kwargs) #creation of one ODST layer
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float('inf')) #add the new features to the input dimension for the next layer, but do not exceed max_features
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = num_layers, layer_dim, tree_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x): #forward pass through the dense block
        initial_features = x.shape[-1] #number of features in the input tensor
        for layer in self: #each layer = ODST layer
            layer_inp = x
            if self.max_features is not None: 
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features #number of features to keep from the input, ensuring we do not exceed max_features
                if tail_features != 0:
                    layer_inp = torch.cat([layer_inp[..., :initial_features], layer_inp[..., -tail_features:]], dim=-1) #keep the initial features and the tail features, drop the middle features if max_features is exceeded
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout) #apply dropout to the input features during training
            h = layer(layer_inp) #pass the input through the ODST layer, which will output new features based on the input features
            x = torch.cat([x, h], dim=-1) #concatenate the input features with the new features from the ODST layer, so that the next layer can use all features

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
        return outputs
