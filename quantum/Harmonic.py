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


from quantum_solutions import Hermites, psi_harmonic, Psi_harmonic




t_start=0
t_end=12
x_start=-20
x_end=20

x=np.linspace(x_start,x_end,400)
t=np.linspace(t_start,t_end,200)
X,T=np.meshgrid(x,t)
n=3
m=.5

v0=0.4

# randphases=[np.tan(np.random.random()*np.pi/2) for i in range(200)]
# phases=lambda v: randphases[int(v*100)]

Psi1=Psi_harmonic(X,T,m,n,Hermites(n)[n])#,phases

fig=plt.figure(figsize=[12,5])
ax=fig.add_subplot(121)
ax.contourf(X,T,Psi1.real,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax.set_xlim(x_start,x_end)
ax.set_ylim(t_start,t_end)

ax2=fig.add_subplot(122)
ax2.contourf(X,T,abs(Psi1)**2,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(x_start,x_end)
ax2.set_ylim(t_start,t_end)
ax2.plot([0,30],[0,30/v0])
ax2.plot([0,30],[0,30])


# In[ ]:



fig=plt.figure(figsize=[6,6])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')


plt.show()
# In[ ]:





# In[ ]:
from quantum_solutions import Packet_harmonic



# In[ ]:


t_start=0
t_end=30
x_start=-2
x_end=2

x=np.linspace(x_start,x_end,400)
t=np.linspace(t_start,t_end,600)
X,T=np.meshgrid(x,t)
N1=0
N2=10
m=15


# randphases=[np.tan(np.random.random()*np.pi/2) for i in range(200)]
# phases=lambda v: randphases[int(v*100)]

Psi1=Packet_harmonic(X,T,m,N1,N2)#,phases


# print(Psi1.shape)
# Psi_R=np.zeros(shape=Psi1.shape,dtype=np.float64)
# Psi_R[:]=Psi1
# print(Psi1.real[110])


fig=plt.figure(figsize=[16,7])
ax=fig.add_subplot(121)
ax.contourf(X,T,Psi1.real,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax.set_xlim(x_start,x_end)
ax.set_ylim(t_start,t_end)

ax2=fig.add_subplot(122)
ax2.contourf(X,T,abs(Psi1)**2,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(x_start,x_end)
ax2.set_ylim(t_start,t_end)
ax2.plot([0,30],[0,30/v0])
ax2.plot([0,30],[0,30])


# In[ ]:



fig=plt.figure(figsize=[12,12])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')


# In[ ]:

plt.show()
print('Harmonic done')