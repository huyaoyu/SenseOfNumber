
from __future__ import print_function

import argparse
import numpy as np
import plotly
import plotly.graph_objs as go

if __name__ == "__main__":
    print("Draw a linear model with plotly offline.")

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

    # Plotly graph objects.
    pgo = []

    for i in range(x.shape[1]):
        color = ct[i]
        line = go.Scatter( \
            x = x[:, i],
            y = y[:, i],
            mode = "lines",
            line = { "color": "rgba(%f, %f, %f, %f)" % ( color[0], color[1], color[2], color[3] ) },
            hoverinfo="none" )
        
        pgo.append( line )

    # Circles.
    colorCircles = circles[:, 3:7] / 255
    color = [ "rgba(%f, %f, %f, %f)" % ( c[0], c[1], c[2], c[3] ) for c in colorCircles ]
    text = [ "%f" % (t) for t in circles[:, 7] ]

    pgo.append( \
        go.Scatter( \
            x = circles[:, 0],
            y = circles[:, 1],
            mode = "markers",
            marker={ "color": color, "size":50, "line": { "color":"rgba(0,0,0,1)", "width": 2 } },
            text = text,
            hoverinfo="text"
            ) 
         )

    plotly.offline.plot( { \
        "data": pgo, \
        "layout": go.Layout(title="lines+scatters", showlegend=False)}, \
        auto_open=True )

    # for cc in circles:
    #     color = cc[3:] / 255
    #     circle = Circle(( cc[0], cc[1] ), cc[2], fill=True)
    #     circle.set_facecolor( color )
    #     circle.set_edgecolor( "k" )
    #     ax.add_patch(circle)
