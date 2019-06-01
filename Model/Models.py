
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class NaiveFullyConnected(nn.Module):
    def __init__(self):
        super(NaiveFullyConnected, self).__init__()

        self.model = nn.Sequential( \
            nn.Linear( 20, 32 ),
            nn.Tanh(),
            # nn.Linear( 32, 32 ),
            nn.Linear( 32, 1 )
        )

    def forward(self, x):
        return self.model(x)