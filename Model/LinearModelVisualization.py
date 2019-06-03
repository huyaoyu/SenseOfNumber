
from __future__ import print_function

import copy
import numpy as np
import math

class ColorMap(object):
    def __init__(self, name):
        self.name = name

    def get_color(self, x):
        pass

    def convert_to_intensity(self, x):
        x = np.fabs(x)

        mask = x > 1.0
        x[mask] = 1.0

        x = x * 255
        
        return x.astype(np.uint8)

class BlueRedMap(ColorMap):
    def __init__(self, name):
        super(BlueRedMap, self).__init__( name )

    def get_color(self, x, limits):
        """
        x is a vector.
        limits is a two-column array. limits[:, 0] should be smaller than 0 and limits[:, 1] should be
        bigger than 0. 

        For x values bigger or equal 0, the most solid red color is achieved when x reaches limits[:,1].
        Similarly, when x < 0, the most solid blue color is achieved when it reaches limits[:,0].

        The returned list is in RGBA order.
        """

        c = np.zeros( (x.shape[0], 4), dtype=np.uint8 )

        mask = x >= 0
        if ( np.any(mask) ):
            c[mask, 0] = 255
            c[mask, 3] = self.convert_to_intensity( x[mask] / limits[mask, 1] )

        mask = x < 0
        if ( np.any(mask) ):
            c[mask, 2] = 255
            c[mask, 3] = self.convert_to_intensity( x[mask] / limits[mask, 0] )

        return c

class Layer(object):
    def __init__(self, id, flagActivation=True, flagWeight=True):
        self.id = id
        self.flagActivation = flagActivation
        self.flagWeight = flagWeight

        self.weight  = None
        self.bias    = None # Must be a 1D NumPy array.
        self.weightX = None
        self.BA      = None # Before activation
        self.output  = None
    
    def copy_weight(self, w):
        self.weight = copy.deepcopy( w )

    def copy_bias(self, b):
        self.bias = copy.deepcopy( b )
    
    def copy_BA(self, ba):
        self.BA = copy.deepcopy( BA )

    def copy_output(self, output):
        self.output = copy.deepcopy( output )

    def have_activation(self):
        return self.flagActivation

    def have_weight(self):
        return self.flagWeight

    def get_n(self):
        return bias.shape[0]
    
    def make_weight_x(self, x):
        if ( self.weight is None ):
            raise Exception("self.weight is None")

        self.weightX = np.zeros_like( self.weight, dtype=np.float32 )

        for i in range(self.weightX.shape[0]):
            temp = np.multiply( self.weight[i, :], x )
            self.weightX[i, :] = temp

class InputLayer(Layer):
    def __init__(self, id):
        super(InputLayer, self).__init__( id, False, False )
    
class NeuronLayer(Layer):
    def __init__(self, id, flagActivation):
        super(NeuronLayer, self).__init__( id, flagActivation=flagActivation )

class OutputLayer(Layer):
    def __init__(self, id):
        super(OutputLayer, self).__init__( id, flagActivation=False )

class LinearModelGraph(object):
    def __init__(self, name, layers):
        self.name = name

        self.layers = layers # A list of Layer objects. The order should be the same with the model.

        self.neronRadius  = 1
        self.neronSpace   = 3
        self.strideWieght = 5 # The horizontal distance of the weight lines.

        self.colorMap = BlueRedMap("BlueRedMap")

    def set_layers(self, layers):
        self.layers = layers

    def generate_graph(self):
        """
        wd: The working directory.
        prefix: The prefix of filename.
        """
        nLayers = len( self.layers )
        if ( 0 == nLayers ):
            raise Exception("Zero length layers.")

        circles = [] # x, y, radius, R, G, B, A.
        lines   = [] # x0, y0, x1, y1, R, G, B, A.

        startingXCoor = 0.0
        x0 = startingXCoor
        yOrigin = 0.0

        for i in range(nLayers):
            layer = self.layers[i]

            # Check if it has weights.
            if ( layer.have_weight() ):
                # Get the minimum and maximum values of the weight.
                weightMin = layer.weightX.min()
                weightMax = layer.weightX.max()

                if ( weightMin >=0 ):
                    weightMin = -1

                # Get the previous layer.
                lp = self.layers[i-1]
                
                x1 = x0 + self.strideWieght

                for iP in range( lp.output.shape[0] ):
                    y0 = yOrigin + iP * self.neronSpace

                    w = layer.weightX[:, iP]
                    limits = np.zeros((w.shape[0] ,2), dtype=np.float32)
                    limits[:, 0] = weightMin
                    limits[:, 1] = weightMax

                    # Get the color map.
                    cm = self.colorMap.get_color( w, limits )

                    for iC in range( layer.output.shape[0] ):
                        line = np.zeros( (8), dtype=np.float32 )

                        line[0] = x0
                        line[1] = y0
                        line[2] = x1
                        line[3] = yOrigin + iC * self.neronSpace
                        line[4] = cm[iC, 0]
                        line[5] = cm[iC, 1]
                        line[6] = cm[iC, 2]
                        line[7] = cm[iC, 3]

                        lines.append( line )

                # Shift x0.
                x0 += self.strideWieght

            # Get the color of BA.
            minBA = layer.BA.min()
            maxBA = layer.BA.max()
            if ( minBA >= 0 ):
                minBA = -1.0

            limits = np.zeros( (layer.BA.shape[0], 2), dtype=np.float32 )
            limits[:, 0] = minBA
            limits[:, 1] = maxBA
            
            cm = self.colorMap.get_color( layer.BA, limits )

            # Nerons.
            for iC in range( layer.output.shape[0] ):
                circle = np.zeros( (7), dtype=np.float32 )

                circle[0] = x0
                circle[1] = yOrigin + iC * self.neronSpace
                circle[2] = self.neronRadius
                circle[3] = cm[iC, 0]
                circle[4] = cm[iC, 1]
                circle[5] = cm[iC, 2]
                circle[6] = cm[iC, 3]

                circles.append( circle )

            if ( layer.have_activation() ):
                x0 += self.neronRadius * 2

                # Get the color of output.
                minOutput = layer.output.min()
                maxOutput = layer.output.max()
                if ( minOutput >= 0 ):
                    minOutput = -1.0

                limits = np.zeros( (layer.output.shape[0], 2), dtype=np.float32 )
                limits[:, 0] = minOutput
                limits[:, 1] = maxOutput

                cm = self.colorMap.get_color( layer.output, limits )

                # Nerons.
                for iC in range( layer.output.shape[0] ):
                    circle = np.zeros( (7), dtype=np.float32 )

                    circle[0] = x0
                    circle[1] = yOrigin + iC * self.neronSpace
                    circle[2] = self.neronRadius
                    circle[3] = cm[iC, 0]
                    circle[4] = cm[iC, 1]
                    circle[5] = cm[iC, 2]
                    circle[6] = cm[iC, 3]

                    circles.append( circle )

        return circles, lines
