
from __future__ import print_function

import argparse
import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

from Utilities.Filesystem import test_dir

if __name__ == "__main__":
    print("Draw a linear model with matplotlib.")

    parser = argparse.ArgumentParser(description="Draw a linear model with matplotlib.")

    parser.add_argument("--working-dir", type=str, default="./", \
        help="The working directory.")
    
    parser.add_argument("--input", type=str, default="CirclesLines_Indexing.json", \
        help="The input json file.")

    parser.add_argument("--output-dir", type=str, default="Visulization", \
        help="The output directory relative to the working directory.")

    args = parser.parse_args()

    # Check the working directory.
    test_dir( args.working_dir )
    print("Working directory is %s." % (args.working_dir))

    # Check the output directory.
    outDir = args.working_dir + "/" + args.output_dir
    test_dir( outDir )
    print("Output directory is %s." % (outDir) )

    # Load the json file.
    with open( args.working_dir + "/" + args.input, "r" ) as fp:
        indexing = json.load( fp )

    # Load the file list.
    entries = indexing["results"]
    circlesName = indexing["circlesName"]
    linesName = indexing["linesName"]

    for entry in entries:
        clNpz = np.load( args.working_dir + "/" + entry["file"] )
        circles = clNpz[ circlesName ]
        lines   = clNpz[ linesName ]
        
        # Print information.
        print( "%s loaded." % (args.working_dir + "/" + entry["file"]) )
        print( "%d circles and %d lines. " % ( circles.shape[0], lines.shape[0] ) )

        # Debug.
        x = lines[:,[0, 2]].transpose()
        y = lines[:,[1, 3]].transpose()
        c = lines[:, 4:] / 255

        ct = [(t[0], t[1], t[2], t[3]) for t in c ]

        fig = plt.figure( figsize=(3.15, 6.3), dpi=300 )
        ax = fig.gca()

        # fig, ax = plt.subplots()

        for i in range(x.shape[1]):
            ax.plot( x[:, i], y[:, i], color=ct[i] )

        for cc in circles:
            color = cc[3:7] / 255
            circle = Circle(( cc[0], cc[1] ), cc[2], fill=True)
            circle.set_facecolor( color )
            circle.set_edgecolor( "k" )
            ax.add_patch(circle)

        ax.axis("off")
        xlim = ax.get_xlim()
        ax.set_xlim( (xlim[0]-5, xlim[1]+5) )

        ax.set_title("%d, T%d, P%.4f" % ( entry["inputIndex"], entry["trueDist"], entry["predDist"] ))

        # Save the figure.
        outFn = "%s/LinearVis_%d.png" % (outDir, entry["inputIndex"])
        fig.savefig(outFn, dpi=300, bbox_inches="tight", pad_inches=0.0)
        print("%s saved." % (outFn))
