#!/usr/bin/env python
# coding: utf-8

# 
# other stuff
# 

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import cm

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# #free particle:
from quantum_solutions import Psi,Psi_box


# g=lambda v:1/np.sqrt(1-v**2)
# add=lambda v1,v2:(v1+v2)/(1+v1*v2)
# def Packet(x,t,w,N,v0,K,phases=lambda v:0):
#     return sum([Psi(x,t,w,add(v0,v),phase=phases(v))*np.exp(-(K*v*g(v))**2)/N for v in np.linspace(-1,1,N+2)[1:-1]])
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



# In[16]:


# def Packet_box(x,t,w,M1,M2):
#     B=sum([np.exp(-4*((m-(M1+M2)/2)/(M2-M1))**2) for m in np.arange(M1,M2)])
#     return sum([Psi_box(x,t,w,m)*np.exp(-(5*(m-(M1+M2)/2)/(M2-M1))**2)/B for m in np.arange(M1,M2)])#*
from quantum_solutions import Packet_box

# In[22]:


t_start=0
t_end=12
x_start=-4
x_end=4
x=np.linspace(x_start,x_end,300)
t=np.linspace(t_start,t_end,500)
X,T=np.meshgrid(x,t)
x1=-1.2
x2=1.2
w=12
v0=0
N=500
K=1
Psi1=Packet(X-x1,T,w,N,v0,K)+Packet(X-x2,T,w,N,v0,K)

fig=plt.figure(figsize=[12,5])
ax=fig.add_subplot(121)
ax.contourf(X,T,Psi1.real,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax.set_xlim(-4,4)
ax.set_ylim(t_start,t_end)
ax2=fig.add_subplot(122)
ax2.contourf(X,T,abs(Psi1),30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(x_start,x_end)
ax2.set_ylim(t_start,t_end)


# In[23]:



fig=plt.figure(figsize=[8,8])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')


plt.show()
# In[24]:


#diffraction grating?
t_start=0
t_end=36
x_start=-12
x_end=12

x=np.linspace(x_start,x_end,300)
t=np.linspace(t_start,t_end,500)

X,T=np.meshgrid(x,t)
dx=1.2
w=12
v0=0
N=500
K=1

Psi1=sum([Packet(X-n*dx,T,w,N,v0,K) for n in range(-6,7)])

fig=plt.figure(figsize=[12,5])
ax=fig.add_subplot(121)
ax.contourf(X,T,Psi1.real,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax.set_xlim(x_start,x_end)
ax.set_ylim(t_start,t_end)

ax2=fig.add_subplot(122)
ax2.contourf(X,T,abs(Psi1),30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(x_start,x_end)
ax2.set_ylim(t_start,t_end)


# In[25]:



fig=plt.figure(figsize=[8,8])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')


plt.show()
# In[26]:


#diffraction grating?
t_start=0
t_end=60
x_start=-8
x_end=30

x=np.linspace(x_start,x_end,300)
t=np.linspace(t_start,t_end,500)

X,T=np.meshgrid(x,t)
dx=1.2
w=12
v0=0
N=500
K=1

Psi1=sum([Packet(X-n*dx,T,w,N,v0,K) for n in range(-6,7)])

fig=plt.figure(figsize=[12,5])
ax=fig.add_subplot(121)
ax.contourf(X,T,Psi1.real,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax.set_xlim(x_start,x_end)
ax.set_ylim(t_start,t_end)

ax2=fig.add_subplot(122)
ax2.contourf(X,T,abs(Psi1),30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(x_start,x_end)
ax2.set_ylim(t_start,t_end)


# In[27]:



fig=plt.figure(figsize=[8,8])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')


plt.show()
# In[ ]:


#diffraction grating?
t_start=0
t_end=160
x_start=-20
x_end=100

x=np.linspace(x_start,x_end,500)
t=np.linspace(t_start,t_end,800)

X,T=np.meshgrid(x,t)
dx=1.2
w=12
v0=0
N=1000
K=1

Psi1=sum([Packet(X-n*dx,T,w,N,v0,K) for n in range(-6,7)])

fig=plt.figure(figsize=[12,5])
ax=fig.add_subplot(121)
ax.contourf(X,T,Psi1.real,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax.set_xlim(x_start,x_end)
ax.set_ylim(t_start,t_end)

ax2=fig.add_subplot(122)
ax2.contourf(X,T,abs(Psi1),30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(x_start,x_end)
ax2.set_ylim(t_start,t_end)


# In[ ]:



fig=plt.figure(figsize=[15,15])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')



plt.show()
print('time_diffraction done')