
from __future__ import print_function

import argparse
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("Draw a linear model with matplotlib.")

    parser = argparse.ArgumentParser(description="Draw a linear model with matplotlib.")

    parser.add_argument("--circles", type=str, default="circles.npy", \
        help="The filename of the circles.")

    parser.add_argument("--lines", type=str, default="lines.npy", \
        help="The filename of the lines.")

    args = parser.parse_args()

    # Load the files.
    circles = np.load(args.circles)
    lines   = np.load(args.lines)
    
    # Print information.
    print("%d circles and %d lines. " % ( circles.shape[0], lines.shape[0] ))

    # Debug.
    x = lines[:,[0, 2]].transpose()
    y = lines[:,[1, 3]].transpose()
    c = lines[:, 4:] / 255

    ct = [(t[0], t[1], t[2], t[3]) for t in c ]

    fig, ax = plt.subplots()

    for i in range(x.shape[1]):
        ax.plot( x[:, i], y[:, i], color=ct[i] )

    for cc in circles:
        color = cc[3:7] / 255
        circle = Circle(( cc[0], cc[1] ), cc[2], fill=True)
        circle.set_facecolor( color )
        circle.set_edgecolor( "k" )
        ax.add_patch(circle)

    plt.show()
