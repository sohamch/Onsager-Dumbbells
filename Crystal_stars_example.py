import numpy as np
import onsager.crystal as crystal
from jumpnet3 import *
from stars import *
from test_structs import *
from states import *
from gensets import *
famp0 = [np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126]
family = [famp0]
pairs_pure = genpuresets(cube,0,family)

jset = purejumps(cube,0,pairs_pure,0.3,0.01,0.01)

crys_stars = StarSet(cube,0,jset,pairs_pure,2)

for state in crys_stars.starset[0]:
    print (state)
