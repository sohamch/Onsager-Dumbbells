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
