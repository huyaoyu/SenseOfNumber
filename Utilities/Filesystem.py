
from __future__ import print_function

import os

def test_dir(d):
    if ( False == os.path.isdir(d) ):
        os.makedirs(d)
