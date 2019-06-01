
from __future__ import print_function

import numpy as np

VECTOR_LEN = 20
MAX_DISTANCE = 15

def create_dataset(vecLen, minDist, maxDist, fnVec, fnDist):
    # Create the training data.
    vectors = []
    distances = []

    for i in range(vecLen):
        for j in range(minDist-1, maxDist):
            if ( i + j + 1 >= vecLen ):
                break
            
            v = np.zeros((vecLen), dtype=np.int32)
            v[i] = 1
            v[i+j+1] = 1
            
            vectors.append( v )
            distances.append( j + 1 )

    # Print information.
    print("Length of the dataset is %d." % ( len( vectors ) ))

    # Convert the list into NumPy arrays.
    vectors   = np.stack( vectors, axis=0 ).astype(np.int32)
    distances = np.array( distances, np.int32 )

    # Save the data into filesystem.
    np.savetxt(fnVec, vectors, fmt="%d")
    np.savetxt(fnDist, distances, fmt="%d")

if __name__ == "__main__":
    create_dataset( VECTOR_LEN, 1, MAX_DISTANCE, "vectors.dat", "distances.dat" )
    create_dataset( VECTOR_LEN, 16, VECTOR_LEN - 1, "vectorsInfer.dat", "distancesInfer.dat" )

    print("Done.")