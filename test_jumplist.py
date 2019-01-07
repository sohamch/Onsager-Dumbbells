import numpy as np
import onsager.crystal as crystal
from states import *
from Onsager_calc_db import BareDumbbell
from collision import *

latt = np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])*0.55
DC_Si = crystal.Crystal(latt,[[np.array([0.,0.,0.]),np.array([0.25,0.25,0.25])]],["Si"])

#Now make the [110] dumbbell states
#Silicon atomic diameter - 0.2 nm - that is the length of the dumbbell.
famp0 = [np.array([1.,1.,0.])/np.sqrt(2)*0.2]
family = [famp0]
pdbcontainer = dbStates(DC_Si,0,family)

iorlist=pdbcontainer.iorlist
iorlist
o1_check = np.array([1.,0.,1.])/np.sqrt(2)*0.2
o2_check = np.array([1.,1.,0.])/np.sqrt(2)*0.2

o1 = o1_check if any(np.allclose(o,o1_check,atol=1e-8) for i,o in iorlist) else -o1_check
o2 = o2_check if any(np.allclose(o,o2_check,atol=1e-8) for i,o in iorlist) else -o2_check

o1,o2

db1 = dumbbell(0, o1,np.array([0,0,0]))
db2 = dumbbell(0, o2,np.array([0,0,0]))

jumpset=set([])
crys,chem = pdbcontainer.crys,pdbcontainer.chem
c1=-1
c2=-1
test_jump1 = jump(db1,db2,c1,c2)
print(test_jump1)
jlist=[]
jindlist=[]
col_cond = not (collision_self(crys,chem,test_jump1,0.01,0.01) or collision_others(crys,chem,test_jump1,0.01))
col_cond
for g in crys.G:
    jnew = test_jump1.gop(crys,chem,g)
    mul1=1
    if any(np.allclose(-jnew.state1.o,o1)for i,o1 in pdbcontainer.iorlist):
        db1new = dumbbell(jnew.state1.i,-jnew.state1.o,jnew.state1.R)
        mul1=-1
    else:
        db1new = dumbbell(jnew.state1.i,jnew.state1.o,jnew.state1.R)
    mul2=1
    if any(np.allclose(-jnew.state2.o,o1)for i,o1 in pdbcontainer.iorlist):
        db2new = dumbbell(jnew.state2.i,-jnew.state2.o,jnew.state2.R)
        mul2=-1
    else:
        db2new = dumbbell(jnew.state2.i,jnew.state2.o,jnew.state2.R)
    print(db1new)
    jnew = jump(db1new-db1new.R,db2new-db1new.R,jnew.c1*mul1,jnew.c2*mul2)#Check this part
    print(jnew.state1)
    db1newneg = dumbbell(jnew.state2.i,jnew.state2.o,jnew.state1.R)
    db2newneg = dumbbell(jnew.state1.i,jnew.state1.o,2*jnew.state1.R-jnew.state2.R)
    jnewneg = jump(db1newneg,db2newneg,jnew.c2,jnew.c1)
    # db1new = pdbcontainer.gdumb(g,db1)
    # db2new = pdbcontainer.gdumb(g,db2)
    # #Take back the starting state at the origin unit cell
    # jnew = jump(db1new[0]-db1new[0].R,db2new[0]-db1new[0].R,c1*db1new[1],1*db2new[1])#Check this part
    # db1newneg = dumbbell(jnew.state2.i,jnew.state2.o,jnew.state1.R)
    # db2newneg = dumbbell(jnew.state1.i,jnew.state1.o,-jnew.state2.R)
    # jnewneg = jump(db1newneg,db2newneg,jnew.c2,jnew.c1)
    if not jnew in jumpset:
        if jnewneg in jumpset:
            print(jnew)
            print(jnewneg)
            raise RuntimeError("Negative jump already considered before")
        #add both the jump and it's negative
        jlist.append(jnew)
        jlist.append(jnewneg)
        # jindlist.append(indexed(jnew))
        # jindlist.append(indexed(jnewneg))
        jumpset.add(jnew)
        jumpset.add(jnewneg)

len(jumpset)
test_jump1 in jumpset
#The test jump is not in the list itself. what's wrong?

print(test_jump1)
for j in jumpset:
    # if j.state1.i==test_jump1.state1.i and j.state2.i==test_jump1.state2.i and np.allclose(np.array([0,1,0]),j.state2.R-j.state1.R) and np.allclose(j.state2.o,test_jump1.state2.o) and np.allclose(j.state1.o,test_jump1.state1.o):
        print(j)
