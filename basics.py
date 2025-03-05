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
# def Psi(x,t,w,v,phase=0):
#     g=1/np.sqrt(1-v**2)
#     return np.exp(1j*w*g*v*x-1j*w*g*t+1j*phase)
# def Psi_box(x,t,w,m=0,phase=0):
#     v=np.sqrt((m*np.pi)**2/(w**2+(m*np.pi)**2))
#     #print(v)
#     return Psi(x,t,w,v,phase)-Psi(x,t,w,-v,phase)
x=np.linspace(0,1,200)
t=np.linspace(0,10,100)
X,T=np.meshgrid(x,t)
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.plot_surface(X,T,Psi_box(X,T,4,2).real)
#plt.plot(x,Psi(x,1,20,.8).real)
ax.view_init(elev=25,azim=-45)


plt.show()
# In[3]:



x=np.linspace(0,5,100)
t=np.linspace(0,5,100)
X,T=np.meshgrid(x,t)
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.plot_surface(X,T,Psi(X,T,2,.6).real)
#plt.plot(x,Psi(x,1,20,.8).real)
ax.view_init(elev=75,azim=-45)


plt.show()
# In[4]:


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


# In[6]:


from quantum_solutions import Packet

x=np.linspace(-30,30,400)
t=np.linspace(0,40,200)
X,T=np.meshgrid(x,t)
v0=0.2
N=100
w=2
K=2
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.plot_surface(X,T,Packet(X,T,w,N,v0,K).real,cmap=cm.coolwarm)
#plt.plot(x,Psi(x,1,20,.8).real)
ax.view_init(elev=75,azim=-25)


plt.show()
# In[7]:



fig=plt.figure()
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Packet(X,T,w,N,v0,K)),origin='lower')


plt.show()
# In[8]:



x=np.linspace(-30,30,400)
t=np.linspace(0,40,200)
X,T=np.meshgrid(x,t)
v0=0.4
N=100
w=2
K=3
# fig=plt.figure()
# ax=fig.add_subplot()
# ax.contourf(X,T,Packet(X,T,w,N,v0,K).real,30,cmap=cm.coolwarm)
# ax.plot([0,30],[0,30/v0])
# ax.set_xlim(-30,30)
# ax.set_ylim(0,40)


Psi1=Packet(X,T,w,N,v0,K)

fig=plt.figure(figsize=[12,5])
ax=fig.add_subplot(121)
ax.contourf(X,T,Psi1.real,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax.set_xlim(-30,30)
ax.set_ylim(0,40)
ax2=fig.add_subplot(122)
ax2.contourf(X,T,abs(Psi1)**2,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(-30,30)
ax2.set_ylim(0,40)
ax2.plot([0,30],[0,30/v0])
ax2.plot([0,30],[0,30])


plt.show()
# In[9]:


fig=plt.figure(figsize=[12,5])
ax=fig.add_subplot(121)
ax.imshow(complex_array_to_rgb(Psi1),origin='lower')
#ax.plot([0,1],[0,1])
#ax.set_xlim(-30,30)
#ax.set_ylim(0,40)
ax2=fig.add_subplot(122)
ax2.contourf(X,T,abs(Psi1)**2,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(-30,30)
ax2.set_ylim(0,40)
ax2.plot([0,30],[0,30/v0])
ax2.plot([0,30],[0,30])


plt.show()
# In[10]:



# x=np.linspace(-30,30,400)
# t=np.linspace(0,40,200)
# X,T=np.meshgrid(x,t)
# v0=0.4
# N=100
# w=2
# K=3
# fig=plt.figure()
# ax=fig.add_subplot()
# ax.contourf(X,T,abs(Packet(X,T,w,N,v0,K))**2,30,cmap=cm.coolwarm)
# ax.plot([0,30],[0,30/v0])
# ax.plot([0,30],[0,30])
# ax.set_xlim(-30,30)
# ax.set_ylim(0,40)


# In[11]:



x=np.linspace(-30,30,400)
t=np.linspace(0,40,2000)
X,T=np.meshgrid(x,t)
v0=0.4
N=1000
w=15
K=10
fig=plt.figure(figsize=[15,15])
ax=fig.add_subplot()
ax.contourf(X,T,Packet(X,T,w,N,v0,K).real,30,cmap=cm.coolwarm)
ax.plot([0,30],[0,30/v0])
ax.set_xlim(-30,30)
ax.set_ylim(0,40)

plt.show()

# In[12]:




fig=plt.figure(figsize=[15,15])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Packet(X,T,w,N,v0,K)),origin='lower')
#ax.plot([0,30],[0,30/v0])
#ax.set_xlim(-30,30)
#ax.set_ylim(0,40)


plt.show()
# In[13]:



x=np.linspace(-10,30,500)
t=np.linspace(0,40,500)
X,T=np.meshgrid(x,t)
v0=0.4
N=1000
w=15
K=10
fig=plt.figure()
ax=fig.add_subplot()
Psi1=Packet(X,T,w,N,v0,K)
ax.contourf(X,T,abs(Psi1)**2,30,cmap=cm.coolwarm)
#ax.plot([0,30],[0,30/v0])
#ax.plot([0,30],[0,30])
ax.set_xlim(-10,30)
ax.set_ylim(0,40)

plt.show()

# In[14]:



fig=plt.figure(figsize=[15,15])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')


# In[15]:


#from quantum_solutions import Packet

x=np.linspace(-5,18,400)
t=np.linspace(0,30,400)
X,T=np.meshgrid(x,t)
v0=0.4
N=1000
w=150
K=30
fig=plt.figure()
ax=fig.add_subplot()
Psi1=Packet(X,T,w,N,v0,K)
ax.contourf(X,T,abs(Psi1)**2,30,cmap=cm.coolwarm)
#ax.plot([0,30],[0,30/v0])
ax.plot([0,30],[0,30])
ax.set_xlim(-5,18)
ax.set_ylim(0,30)



plt.show()
print('basics done')