import numpy as np
import onsager.crystal as crystal
from gensets import *
from test_structs import *
from states import *
from representations import *
# from states import *
famp0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
famp12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
family = [famp0,famp12]
pairs_pure = genpuresets(tet2,0,family)
len(pairs_pure)
thrange=1
sdpairs = genPairSets(tet2,0,pairs_pure,thrange)


dblist=[]
db=dumbbell(1, np.array([-1., -1.,  0.]), np.array([-1, -1,  0]))
dblist.append(db)
for g in tet2.G:
    newdb = db.gop(tet2,0,g)
    if not any(db1==newdb or db1==-newdb for db1 in dblist):
        dblist.append(newdb)
        # print (crystal.GroupOp.optype(g.rot))
dblist

famp0 = [np.array([1.,0.,0.])]
famp12 = [np.array([1.,1.,0.])]
family = [famp0,famp12]
pairs_pure = genpuresets(tet2,0,family)
pairset = genPairSets(tet2,0,pairs_pure,1)
pset = Pairstates(tet2,0,pairset)
count=0
for plis in pset.sympairlist:
    for pair in plis:
        if pair.db.i==0 and np.allclose(pair.db.R,np.array([1,1,0]),atol=1e-8):
            count=1
            lis=plis.copy()
            break
    if count == 1:
        break
    
