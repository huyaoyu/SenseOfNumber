
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

import Arguments
from Model.Models import NaiveFullyConnected

class Job(object):
    def __init__(self):
        self.model = None

        self.epoches = 0

    def initialize(self):
        pass
    
    def finalize(self):
        pass

    def load_model(self, fn):
        pass

    def set_epoches(self, e):
        self.epoches = e

class TrainJob(Job):
    def __init__(self):
        super(TrainJob, self).__init__()

        self.criterion = None
        self.optimizer = None

    def initialize(self):
        pass

    def finalize(self):
        pass

    def load_model(self, fn):
        pass

    def execute(self):
        pass

class TestJob(Job):
    def __init__(self):
        super(TestJob, self).__init__()

    def initialize(self):
        pass
    
    def finalize(self):
        pass

    def load_model(self, fn):
        pass

    def execute(self):
        pass

class TrainNaiveFullyConnected(TrainJob):
    def __init__(self):
        super(TrainNaiveFullyConnected, self).__init__()

        self.model = NaiveFullyConnected()

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0)

        # Training process specific variables.
        self.vectors   = None
        self.distances = None

    def initialize(self):
        # Read the input dataset.
        self.vectors   = np.loadtxt("vectors.dat", dtype=np.int32)
        self.distances = np.loadtxt("distances.dat", dtype=np.int32)
        self.distances = self.distances.reshape((-1, 1))

        print("vectors.shape = {}".format( self.vectors.shape ))

    def finalize(self):
        # Save the model.
        torch.save( self.model.state_dict(), "saved.pt" )

    def load_model(self, fn):
        self.model.load_state_dict( torch.load(fn) )

    def execute(self):
        nVectors  = self.vectors.shape[0]
        lenVector = self.vectors.shape[1]

        # Training.
        idx = np.linspace(0, nVectors-1, nVectors, dtype=np.int32)
        N = 5 # Size of the miniBatch

        nTrainingLoop = int( nVectors / N )

        self.model.train()

        for i in range(self.epoches):
            # Randomize an index array.
            np.random.shuffle(idx)
            idxPos = 0

            lossSum = 0.0

            for j in range(nTrainingLoop):
                # Fill in the miniBatch.
                mbIdx = idx[idxPos:idxPos+N]
                mb    = self.vectors[mbIdx, :]

                mbDist = self.distances[mbIdx, :]
                
                idxPos += N

                # Convert the mini batch into torch tensor.
                mb     = torch.from_numpy(mb).float()
                mbDist = torch.from_numpy(mbDist).float()

                # Forward.
                dPred = self.model( mb )

                # Loss.
                loss = self.criterion( dPred, mbDist )
                lossSum += loss.item()

                # Optimize.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print("i = %d, j = %02d, loss = %4.2f." % ( i, nTrainingLoop, lossSum/nTrainingLoop ) )

if __name__ == "__main__":

    args = Arguments.args

    if ( "train" == args.job_mode ):
        job = TrainNaiveFullyConnected()
    elif ( "test" == args.job_mode ):
        job = None
    elif ( "infer" == args.job_mode ):
        job = None
    else:
        raise Exception("Unexpected job mode (%s)." % ( args.job_mode ))

    if ( True == args.load_model ):
        job.load_model("./saved.pt")
    
    job.set_epoches( args.epoches )
    job.initialize()
    job.execute()
    job.finalize()
