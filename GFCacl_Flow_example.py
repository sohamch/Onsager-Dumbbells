import numpy as np
from test_structs import *
from states import *
import onsager.crystal as crystal
from GFcalc_dumbbells import *

cube = crystal.Crystal(np.array([[0.28,0.,0.],[0.,0.28,0.],[0.,0.,0.28]]),[[np.zeros(3)]])
famp0 = [np.array([0.126,0.,0.])]
family = [famp0]
pdbcontainer = dbStates(cube,0,family)

jset,jind = pdbcontainer.jumpnetwork(0.3,0.01,0.01)

#What are the gops that do the same as identity
pdbcontainer.iorlist
glist=[]
translist=[]
test_state1 = pdbcontainer.iorlist[1]
test_state2 = pdbcontainer.iorlist[2]
print(test_state)
for g in pdbcontainer.crys.G:
    R, (ch,inew) = pdbcontainer.crys.g_pos(g,np.array([0,0,0]),(pdbcontainer.chem,test_state1[0]))
    onew  = np.dot(g.cartrot,test_state1[1])
    if any(np.allclose(onew+t[1],0,atol=1.e-8) for t in pdbcontainer.iorlist):
        onew = -onew
    if inew==test_state2[0] and np.allclose(test_state2[1],onew):
        glist.append(g)
        translist.append(g.trans)

optypes=[crystal.GroupOp.optype(g.rot) for g in glist]
print(optypes)
for trans in translist:
    print (trans)
glist4=[]
for i,op in enumerate(optypes):
    if abs(op)==4:
        glist4.append(glist[i])
g1 = glist4[0]

onew  = np.dot(g1.cartrot,test_state[1])
onew
glist=[]
for jlist in jind:
    tup=jlist[0]
    lis=[]
    for gind,g in enumerate(pdbcontainer.crys.G):
        for j in jlist:
            if pdbcontainer.indexmap[gind][tup[0]]==j[0]:
                if pdbcontainer.indexmap[gind][tup[1]]==j[1]:
                    if np.allclose(tup[2],pdbcontainer.crys.g_direc(g,j[2])):
                            lis.append(g)
                            break
    glist.append(lis)

j1 = jset[0][0]
j2 = jset[0][0]
j1==j2
gtest = glist[0][0]
gtest.cartrot
j2new = j1.gop(pdbcontainer.crys,pdbcontainer.chem,gtest)

print(j1)
print(j2)
print(j2new)




jnet_cube_vacancy = cube.jumpnetwork(0, 0.3)
jtest1 = jnet_cube_vacancy[0][0]
jtest2 = jnet_cube_vacancy[0][1]
gvaclist=[]
for g in self.crys.G:
    # more complex: have to check the tuple (i,j) *and* the rotation of dx
    # AND against the possibility that we are looking at the reverse jump too
    if (g.indexmap[self.chem][i0] == i
        and g.indexmap[self.chem][j0] == j
        and np.allclose(dx, np.dot(g.cartrot, dx0))) or \
            (g.indexmap[self.chem][i0] == j
             and g.indexmap[self.chem][j0] == i
             and np.allclose(dx, -np.dot(g.cartrot, dx0))):
        oplist.append(g)
        break

cube_GF = GF_dumbbells(pdbcontainer, jind)

preT = list(np.ones(len(jind)))
betaeneT = list(np.ones(len(jind)))
cube_GF.SetRates([1], [0], preT, betaeneT)
cube_GF.d
#eigenvalues of D, d - with which we'll scale the q-s
#isotropic for cubic lattice - check

tet = crystal.Crystal(np.array([[0.28,0.,0.],[0.,0.28,0.],[0.,0.,0.32]]),[[np.zeros(3)]])
famp0 = [np.array([0.126,0.,0.]),np.array([0.,0.,0.16])]
family = [famp0]
pdbcontainer = dbStates(tet,0,family)

jset,jind = pdbcontainer.jumpnetwork(0.35,0.01,0.01)

glist=[]
test_state = pdbcontainer.iorlist[2]
print(test_state)
for g in pdbcontainer.crys.G:
    R, (ch,inew) = pdbcontainer.crys.g_pos(g,np.array([0,0,0]),(pdbcontainer.chem,test_state[0]))
    onew  = np.dot(g.cartrot,test_state[1])
    if any(np.allclose(onew+t[1],0,atol=1.e-8) for t in pdbcontainer.iorlist):
        onew = -onew
    if inew==test_state[0] and np.allclose(test_state[1],onew):
        glist.append(g)

optypes=[crystal.GroupOp.optype(g.rot) for g in glist]
print(optypes)

jnet_tet_vacancy = tet.jumpnetwork(0, 0.35)

len(jind)
len(jnet_tet_vacancy)

tet_GF = GF_dumbbells(pdbcontainer, jind)

preT = list(np.ones(len(jind)))
betaeneT = list(np.ones(len(jind)))
tet_GF.SetRates([1,1], [0,0], preT, betaeneT)
tet_GF.d
