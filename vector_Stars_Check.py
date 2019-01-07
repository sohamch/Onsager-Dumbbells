#Here we will check portions of the code that are used in vector stars
#Then we'll see if the results match those we's expect.
#Later this will be converted to a test suite.
import numpy as np
import onsager.crystal as crystal
from stars import *
from test_structs import *
from states import *
from representations import *
from functools import reduce
#We will work with a crystal star for a cubic lattice.
#1. Determine orientation families. Here we consider only one for both pure and mixed.
famp0 = [np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126]
family = [famp0]
#2. Create state information containers
pdbcontainer = dbStates(cube,0,family)
mdbcontainer = mStates(cube,0,family)
#Extract omega0 and omega2 jumps
jset0 = pdbcontainer.jumpnetwork(0.3,0.01,0.01)
jset2 = mdbcontainer.jumpnetwork(0.3,0.01,0.01)
#create starset
crys_stars = StarSet(pdbcontainer,mdbcontainer,jset0,jset2,2)

#First, we'll check if the general procedure of finding the vector basis works correctly
db = dumbbell(0,o,np.array([2,1,0]))
pair0 = SdPair(0,np.array([0,0,0]),db)
glist=[]
#Find group operations that leave state unchanged
for g in crys_stars.crys.G:
    pairnew = pair0.gop(crys_stars.crys,crys_stars.chem,g)
    if pairnew == pair0 or pairnew==-pair0:
        glist.append(g)
#Check that glist is correct - should get an identity and a mirror
#Meaning that two vectors remain unchanged. So should get one parallel and one perpendicular vector star.
len(glist)
for g in glist:
    print(crystal.GroupOp.optype(g.rot))
#Find the intersected vector basis for these group operations
vb=reduce(crystal.CombineVectorBasis,[crystal.VectorBasis(*g.eigen()) for g in glist])
#Get orthonormal vectors
vb
#Check that the vector basis is two dimensional and that the normal vector is the z-axis
vlist = crys_stars.crys.vectlist(vb)
vlist
#Check that we have two basis vectors in accordance to the geometry of our state.
