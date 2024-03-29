
from __future__ import print_function

import json
import numpy as np
import torch
import torch.nn as nn

import Arguments
from Model.Models import NaiveFullyConnected, NaiveFullyConnected_V
from Utilities.Filesystem import test_dir

class Job(object):
    def __init__(self, workingDir="./"):
        self.workingDir = workingDir

        test_dir(self.workingDir)
        
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
    def __init__(self, workingDir="./"):
        super(TrainJob, self).__init__(workingDir=workingDir)

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
    def __init__(self, workingDir="./"):
        super(TestJob, self).__init__(workingDir=workingDir)

        self.criterion = None

    def initialize(self):
        pass
    
    def finalize(self):
        pass

    def load_model(self, fn):
        pass

    def execute(self):
        pass

class TrainNaiveFullyConnected(TrainJob):
    def __init__(self, fnVec, fnDist, workingDir="./"):
        super(TrainNaiveFullyConnected, self).__init__(workingDir=workingDir)

        self.model = NaiveFullyConnected()

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0)

        # Training process specific variables.
        self.fnVec     = fnVec
        self.fnDist    = fnDist
        self.vectors   = None
        self.distances = None

    def initialize(self):
        # Read the input dataset.
        self.vectors   = np.loadtxt(self.workingDir + "/" + self.fnVec, dtype=np.int32)
        self.distances = np.loadtxt(self.workingDir + "/" + self.fnDist, dtype=np.int32)
        self.distances = self.distances.reshape((-1, 1))

        print("vectors.shape = {}".format( self.vectors.shape ))

    def finalize(self):
        # Save the model.
        torch.save( self.model.state_dict(), self.workingDir + "/saved.pt" )

    def load_model(self, fn):
        self.model.load_state_dict( torch.load( self.workingDir + "/" + fn ) )

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

class TestNaiveFullyConnected(TestJob):
    def __init__(self, fnVec, fnDist, workingDir="./"):
        super(TestNaiveFullyConnected, self).__init__(workingDir=workingDir)

        self.model = NaiveFullyConnected()

        self.criterion = nn.MSELoss(reduction='none')

        # Testing process specific variables.
        self.fnVec     = fnVec
        self.fnDist    = fnDist
        self.vectors   = None
        self.distances = None

    def initialize(self):
        # Read the input dataset.
        self.vectors   = np.loadtxt(self.workingDir + "/" + self.fnVec, dtype=np.int32)
        self.distances = np.loadtxt(self.workingDir + "/" + self.fnDist, dtype=np.int32)
        self.distances = self.distances.reshape((-1, 1))

        print("vectors.shape = {}".format( self.vectors.shape ))
    
    def finalize(self):
        pass
    
    def load_model(self, fn):
        self.model.load_state_dict( torch.load( self.workingDir + "/" + fn ) )
    
    def execute(self):
        nVectors  = self.vectors.shape[0]
        lenVector = self.vectors.shape[1]

        # Training.
        idx = np.linspace(0, nVectors-1, nVectors, dtype=np.int32)
        N = 1 # Size of the miniBatch

        nLoops = int( nVectors / N )

        self.model.eval()

        idxPos = 0

        lossSum = 0.0

        for j in range(nLoops):
            # Fill in the miniBatch.
            mbIdx = idx[idxPos:idxPos+N]
            mb    = self.vectors[mbIdx, :]

            mbDist = self.distances[mbIdx, :]
            
            idxPos += N

            # Convert the mini batch into torch tensor.
            mb     = torch.from_numpy(mb).float()
            mbDist = torch.from_numpy(mbDist).float()

            # Forward.
            with torch.no_grad():
                dPred = self.model( mb )

            # Show test result for single entry.
            trueDist = mbDist.numpy()[0, 0]
            predDist = dPred.numpy()[0, 0]

            # Loss.
            loss = self.criterion( dPred, mbDist )
            lossSum += loss.item()

            print( "j = %3d, true = %2d, pred = %5.2f, loss = %f." % ( j, trueDist, predDist, loss.item() ) )

        print( "%d Total tests. Average loss = %f." % ( nLoops, lossSum/nLoops ) )
        print( "All test doen." )

class TestNaiveFullyConnected_V(TestNaiveFullyConnected):
    def __init__(self, fnVec, fnDist, idx=0, workingDir="./"):
        super(TestNaiveFullyConnected_V, self).__init__(fnVec, fnDist, workingDir=workingDir)

        self.model = NaiveFullyConnected_V()
        self.idx = idx
        self.intermediateDataDir = None

    def set_intermediate_data_dir(self, d):
        self.intermediateDataDir = d

        test_dir( self.workingDir + "/" + self.intermediateDataDir )
        print("Intermediate diretory is %s." % (self.intermediateDataDir))

    def save_circles_lines(self, fn, circles, lines):
        np.savez( fn, circles=circles, lines=lines )

    def evaluate_single(self):
        nVectors  = self.vectors.shape[0]
        lenVector = self.vectors.shape[1]

        # Training.
        idx = np.linspace(0, nVectors-1, nVectors, dtype=np.int32)
        N = 1 # Size of the miniBatch

        self.model.eval()

        idxPos = self.idx * N

        # Fill in the miniBatch.
        mbIdx = idx[idxPos:idxPos+N]
        mb    = self.vectors[mbIdx, :]

        mbDist = self.distances[mbIdx, :]

        # Convert the mini batch into torch tensor.
        mb     = torch.from_numpy(mb).float()
        mbDist = torch.from_numpy(mbDist).float()
        
        # Forward.
        with torch.no_grad():
            dPred, circles, lines = self.model( mb )

        # Show test result for single entry.
        trueDist = mbDist.numpy()[0, 0]
        predDist = dPred.numpy()[0, 0]

        # Loss.
        loss = self.criterion( dPred, mbDist )

        print( "idx = %3d, true = %2d, pred = %5.2f, loss = %f." % ( self.idx, trueDist, predDist, loss.item() ) )

        # Save the result.
        if ( self.intermediateDataDir is None ):
            singleFile = "CirclesLines_%d.npz" % (self.idx)
        else:
            singleFile = "%s/CirclesLines_%d.npz" % (self.intermediateDataDir, self.idx)

        indexing = { \
            "circlesName": "circles",
            "linesName": "lines",
            "results": [ 
                { "inputIndex": self.idx, "file": singleFile, "trueDist": int(trueDist), "predDist": float(predDist), "loss": float(loss.item()) }
             ]
             }

        with open(self.workingDir + "/CirclesLines_Indexing.json", "w") as fp:
            json.dump( indexing, fp, separators=(',', ': '), indent=2 )

        self.save_circles_lines( self.workingDir + "/" + singleFile, circles, lines )

    def evaluate_all(self):
        nVectors  = self.vectors.shape[0]
        lenVector = self.vectors.shape[1]

        # Training.
        idx = np.linspace(0, nVectors-1, nVectors, dtype=np.int32)
        N = 1 # Size of the miniBatch

        nLoops = int( nVectors / N )

        self.model.eval()

        idxPos = 0

        entries = []

        for i in range(nLoops):
            # Fill in the miniBatch.
            mbIdx = idx[idxPos:idxPos+N]
            mb    = self.vectors[mbIdx, :]

            mbDist = self.distances[mbIdx, :]

            # Convert the mini batch into torch tensor.
            mb     = torch.from_numpy(mb).float()
            mbDist = torch.from_numpy(mbDist).float()
            
            # Forward.
            with torch.no_grad():
                dPred, circles, lines = self.model( mb )

            # Show test result for single entry.
            trueDist = mbDist.numpy()[0, 0]
            predDist = dPred.numpy()[0, 0]

            # Loss.
            loss = self.criterion( dPred, mbDist )

            print( "i = %d, idxPos = %d, true = %2d, pred = %5.2f, loss = %f." % ( i, idxPos, trueDist, predDist, loss.item() ) )

            # Save the result.
            if ( self.intermediateDataDir is None ):
                singleFile = "CirclesLines_%d.npz" % (idxPos)
            else:
                singleFile = "%s/CirclesLines_%d.npz" % (self.intermediateDataDir, idxPos)

            self.save_circles_lines( self.workingDir + "/" + singleFile, circles, lines )

            entries.append( { \
                "inputIndex": idxPos, 
                "file": singleFile, 
                "trueDist": int(trueDist), 
                "predDist": float(predDist), 
                "loss": float(loss.item())
                 } )

            idxPos += N

        # Compose the JSON object.
        indexing = { \
            "circlesName": "circles",
            "linesName": "lines",
            "results": entries
             }

        # Save the JSON file.
        with open(self.workingDir + "/CirclesLines_Indexing.json", "w") as fp:
            json.dump( indexing, fp, separators=(',', ': '), indent=2 )

    def execute(self):
        if ( -1 == self.idx ):
            self.evaluate_all()
        elif ( self.idx >= 0 ):
            self.evaluate_single()
        else:
            raise Exception("Unexpected self.idx value: %d." % (self.idx))

        print("Done with execution.")

if __name__ == "__main__":

    args = Arguments.args

    if ( "train" == args.job_mode ):
        job = TrainNaiveFullyConnected( args.fn_vec, args.fn_dist, workingDir=args.working_dir )
    elif ( "test" == args.job_mode ):
        job = TestNaiveFullyConnected( args.fn_vec, args.fn_dist, workingDir=args.working_dir )
    elif ( "test_v" == args.job_mode ):
        job = TestNaiveFullyConnected_V( args.fn_vec, args.fn_dist, args.test_v_idx, workingDir=args.working_dir )
        job.set_intermediate_data_dir( args.test_v_intermediate_dir )
    elif ( "infer" == args.job_mode ):
        job = None
        raise Exception("Not implemented yet.")
    else:
        raise Exception("Unexpected job mode (%s)." % ( args.job_mode ))

    if ( True == args.load_model ):
        job.load_model( "saved.pt" )
    
    job.set_epoches( args.epoches )
    job.initialize()
    job.execute()
    job.finalize()
