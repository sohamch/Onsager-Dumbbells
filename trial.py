import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
import unittest
from representations import *
from collections import namedtuple

class tril():
    def __init__(self):
        self.or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))
        self.db1 = dumbbell(0,or_1,np.array([1.,0.,0.]),1)
