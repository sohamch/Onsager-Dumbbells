import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple
from representations import *
from jumpnet2 import *
from test_structs import *
crys = crystal.Crystal(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
fam_p0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
purelist=[fam_p0]
fam_m0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
mixlist=[fam_m0]
plist, mlist = gen_orsets(crys,0,purelist,mixlist)
plist[0]
mlist[0]
fam_p0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
fam_p12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
purelist=[fam_p0,fam_p12]
fam_m0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
fam_m12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
mixlist=[fam_m0,fam_m12]
plist, mlist = gen_orsets(omega_Ti,0,purelist,mixlist)
len(plist)
plist
