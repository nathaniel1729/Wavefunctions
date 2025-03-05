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
M1,M2=10,20
m=np.linspace(M1,M2)
plt.plot(np.exp(-(5*(m-(M1+M2)/2)/(M2-M1))**2))


# In[ ]:





# In[17]:


t_end=10
x=np.linspace(0,1,400)
t=np.linspace(0,t_end,8000)
X,T=np.meshgrid(x,t)

w=200
M1=15
M2=30

Psi1=Packet_box(X,T,w,M1,M2)

fig=plt.figure(figsize=[15,15])
ax=fig.add_subplot()
ax.contourf(X,T,Psi1.real,30,cmap=cm.coolwarm)
m=(M1+M2)/2
v0=np.sqrt((m*np.pi)**2/(w**2+(m*np.pi)**2))
plt.plot([0,1],[0,1/v0])
ax.plot([0,1],[0,1])
ax.set_xlim(0,1)
ax.set_ylim(0,t_end)


# In[18]:



fig=plt.figure(figsize=[15,15])
ax=fig.add_subplot()

ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')


plt.show()

# In[19]:


t_end=7
x=np.linspace(0,1,400)
t=np.linspace(0,t_end,4000)
X,T=np.meshgrid(x,t)
w=12
M1=0
M2=3

Psi1=Packet_box(X,T,w,M1,M2)

fig=plt.figure(figsize=[15,15])
ax=fig.add_subplot()
ax.contourf(X,T,Psi1.real,30,cmap=cm.coolwarm)
m=(M1+M2)/2
v0=np.sqrt((m*np.pi)**2/(w**2+(m*np.pi)**2))
plt.plot([0,1],[0,1/v0])
ax.plot([0,1],[0,1])
ax.set_xlim(0,1)
ax.set_ylim(0,t_end)



plt.show()
# In[20]:



fig=plt.figure(figsize=[15,15])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')


# In[21]:


t_start=0
t_end=7
x=np.linspace(0,1,400)
t=np.linspace(t_start,t_end,1000)
X,T=np.meshgrid(x,t)
w=500
M1=70
M2=130
plt.contourf(X,T,np.arctan(12*np.abs(Packet_box(X,T,w,M1,M2))**2),30,cmap=cm.coolwarm)
m=(M1+M2)/2
v0=np.sqrt((m*np.pi)**2/(w**2+(m*np.pi)**2))
print(v0)
plt.plot([0,1],[0,1/v0])
plt.plot([0,1],[0,1])
plt.xlim(0,1)
plt.ylim(t_start,t_end)



plt.show()
print(' box1d done')