#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 05:58:32 2026

@author: murraycantor
"""

import Distributions as Dis
import matplotlib.pyplot as plt
nsamps = 20000

t1 = Dis.Triangular_PDF(3, 5, 8)

plt.plot(t1.x,t1.y)
plt.show()

n1 = Dis.Normal_PDF(5, 1)
plt.plot(n1.x,n1.y)
plt.show()

st = t1.samples(nsamps)
sn = n1.samples(nsamps)

ms = st*sn

e1 = Dis.Empirical_PDF(ms)
plt.plot(e1.x,e1.y)
plt.show()

plt.plot(e1.x,e1.cy)
plt.show()







