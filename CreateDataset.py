
from __future__ import print_function

import numpy as np

VECTOR_LEN = 20
MAX_DISTANCE = 15

def create_dataset_continuous(vecLen, minDist, maxDist, fnVec, fnDist):
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

def create_dataset_selective(selection, fnVec, fnDist):
    """
    selection: A list contains only 1 and zeros. The length of this list is the vector length.
    Positions where is set to 1 is a selected distance. If position of index 0 is set to 1, this means
    the distance of 1 is selected as the data. NOTE: The last position of selection is of no use.
    """

    # Get the length of the vector.
    vecLen = len( selection )

    # Create the data list.
    vectors   = []
    distances = []

    for i in range(vecLen):
        for j in range(vecLen-1):
            if ( i + j + 1 >= vecLen ):
                break
            
            if ( 1 != selection[j] ):
                continue

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
    create_dataset_continuous( VECTOR_LEN, 1, MAX_DISTANCE, "vectors.dat", "distances.dat" )
    create_dataset_continuous( VECTOR_LEN, 16, VECTOR_LEN - 1, "vectorsInfer.dat", "distancesInfer.dat" )

    selection = np.array( [ \
        # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, \
          1, 1, 1, 0, 0, 0, 1, 1, 1, 1, \
          1, 1, 1, 1, 1, 0, 0, 0, 0, 0 \
         ], dtype=np.int32 )

    create_dataset_selective(selection, "VectorsSelectiveTraining.dat", "DistancesSelectiveTraining.dat")

    selection = np.logical_not( selection )
    create_dataset_selective(selection, "VectorsSelectiveInfer.dat", "DistancesSelectiveInfer.dat")

    print("Done.")