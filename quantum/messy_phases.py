#!/usr/bin/env python
# coding: utf-8


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import cm


# #free particle:
from quantum_solutions import Psi,Psi_box


from quantum_solutions import Packet

# In[5]:


import matplotlib
def complex_array_to_rgb(X, theme='dark', rmax=None):
    '''Takes an array of complex number and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''
    absmax = rmax or np.abs(X).max()
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.angle(X) / (2 * np.pi) % 1
    if theme == 'light':
        Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
        Y[..., 2] = 1
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
    Y = matplotlib.colors.hsv_to_rgb(Y)
    return Y



from quantum_solutions import Packet_box


# In[ ]:


x=np.linspace(-30,30,400)
t=np.linspace(0,40,200)
X,T=np.meshgrid(x,t)
v0=0.4
N=200
w=4
K=2
# fig=plt.figure()
# ax=fig.add_subplot()
# ax.contourf(X,T,Packet(X,T,w,N,v0,K).real,30,cmap=cm.coolwarm)
# ax.plot([0,30],[0,30/v0])
# ax.set_xlim(-30,30)
# ax.set_ylim(0,40)
randphases=[np.tan(np.random.random()*np.pi/2) for i in range(200)]
phases=lambda v: randphases[int(v*100)]

Psi1=Packet(X,T,w,N,v0,K,phases)

fig=plt.figure(figsize=[12,5])
ax=fig.add_subplot(121)
ax.contourf(X,T,Psi1.real,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax.set_xlim(-30,30)
ax.set_ylim(0,40)
ax2=fig.add_subplot(122)
ax2.contourf(X,T,np.arctan(.1*abs(Psi1)**2),30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(-30,30)
ax2.set_ylim(0,40)
ax2.plot([0,30],[0,30/v0])
ax2.plot([0,30],[0,30])


# In[ ]:



fig=plt.figure(figsize=[6,6])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')


# In[ ]:



plt.show()
print('messy_phases done')