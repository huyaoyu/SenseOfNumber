
from __future__ import print_function

import numpy as np

VECTOR_LEN = 20
MAX_DISTANCE = 15

if __name__ == "__main__":
    # Create the training data.
    vectors = []
    distances = []
    
    for i in range(VECTOR_LEN):
        for j in range(MAX_DISTANCE):
            if ( i + j + 1 >= VECTOR_LEN ):
                break
            
            v = np.zeros((VECTOR_LEN), dtype=np.int32)
            v[i] = 1
            v[i+j+1] = 1
            
            vectors.append( v )
            distances.append( j + 1 )

    # Print information.
    print("Length of the training dataset is %d." % ( len( vectors ) ))

    # Convert the list into NumPy arrays.
    vectors   = np.stack( vectors, axis=0 ).astype(np.int32)
    distances = np.array( distances, np.int32 )

    # Save the data into filesystem.
    np.savetxt("vectors.dat", vectors, fmt="%d")
    np.savetxt("distances.dat", distances, fmt="%d")
