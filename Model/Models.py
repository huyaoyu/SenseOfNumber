
from __future__ import print_function

import copy
import numpy as np
import torch
import torch.nn as nn

from . import LinearModelVisualization as lmv

class NaiveFullyConnected(nn.Module):
    def __init__(self):
        super(NaiveFullyConnected, self).__init__()

        self.model = nn.Sequential( \
            nn.Linear( 30, 8 ), # 0
            nn.Tanh(),          # A
            # nn.Linear( 32, 32 ),
            nn.Linear( 8, 1 )   # 1
        )

    def forward(self, x):
        return self.model(x)

class NaiveFullyConnected_V(NaiveFullyConnected):
    def __init__(self):
        super(NaiveFullyConnected_V, self).__init__()

    def forward(self, x):
        # Make a copy of x.
        c_x = x.clone()

        # Make a copy of the model.
        c_m = copy.deepcopy(self.model)

        # Create a list of lmv.Layer objects.
        layers = []

        # Input layer.
        layer = lmv.InputLayer("Input")
        layer.BA = c_x.cpu()[0,:].numpy().reshape((-1,))
        layer.output = copy.deepcopy( layer.BA )
        layers.append( layer )

        with torch.no_grad():
            # 0.
            layer = lmv.NeuronLayer("0", True)

            layer.weight = c_m[0].weight.detach().cpu().numpy()
            layer.make_weight_x( c_x.cpu()[0,:].numpy() )
            layer.bias   = c_m[0].bias.detach().cpu().numpy().reshape((-1,))

            output = c_m[0](c_x)
            layer.BA     = output.cpu().numpy().reshape((-1,))

            output = c_m[1](output)
            layer.output = output.cpu().numpy().reshape((-1,))

            layers.append(layer)

            # 1.
            layer = lmv.OutputLayer("1")

            layer.weight = c_m[2].weight.detach().cpu().numpy()
            layer.make_weight_x( output.numpy() )
            layer.bias = c_m[2].bias.detach().cpu().numpy().reshape((-1,))

            output = c_m[2](output)
            layer.BA = output.cpu().numpy().reshape((-1,))
            layer.output = copy.deepcopy( layer.BA )

            layers.append(layer)

        # Generate the graph.
        linearModelGraph = lmv.LinearModelGraph("TestGraph", layers)
        
        circles, lines = linearModelGraph.generate_graph()

        # Save the visualization as NumPy arrays.
        circles = np.stack(circles, axis=0)
        lines = np.stack(lines, axis=0)
        
        np.save("circles.npy", circles)
        np.save("lines.npy", lines)

        return self.model(x)