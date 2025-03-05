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


v0=0.4
from quantum_solutions import Packet_harmonic



from quantum_solutions import dot, Psi_2d, Psi_box_2d_x,  Psi_box_2d_y,  Psi_box_2d, Gaussian
from quantum_solutions import Packet_box_2d,g_2d,tan_to_sin, add_2d,Packet_2d



x_start=0
x_end=1
y_start=0
y_end=1

x=np.linspace(x_start,x_end,400)
y=np.linspace(y_start,y_end,400)
X,Y=np.meshgrid(x,y)
t=.2
w=10
vx,vy=.1,.706
V=[0*X+vx,0*Y+vy]


# randphases=[np.tan(np.random.random()*np.pi/2) for i in range(200)]
# phases=lambda v: randphases[int(v*100)]

Psi1=Psi_2d([X,Y],t,w,V)


# print(Psi1.shape)
# Psi_R=np.zeros(shape=Psi1.shape,dtype=np.float64)
# Psi_R[:]=Psi1
# print(Psi1.real[110])


fig=plt.figure(figsize=[16,7])
ax=fig.add_subplot(121)
ax.contourf(X,Y,Psi1.real,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax.set_xlim(x_start,x_end)
ax.set_ylim(y_start,y_end)

ax2=fig.add_subplot(122)
ax2.contourf(X,Y,abs(Psi1)**2,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(x_start,x_end)
ax2.set_ylim(y_start,y_end)
ax2.plot([0,30],[0,30/v0])
ax2.plot([0,30],[0,30])


# In[ ]:



fig=plt.figure(figsize=[8,8])
ax=fig.add_subplot()
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')



print('(linear wave done)')
plt.show()

from quantum_solutions import draw_frames



x_start=-8
x_end=20
y_start=-8
y_end=20

x=np.linspace(x_start,x_end,30)
y=np.linspace(y_start,y_end,30)
X,Y=np.meshgrid(x,y)
w=2
V0=[.4,.1]
K=.9
dV=[1/(10*K),1/(10*K)]
N=8

t=[0,6,12,18]

# randphases=[np.tan(np.random.random()*np.pi/2) for i in range(200)]
# phases=lambda v: randphases[int(v*100)]
f=lambda t: Packet_2d([X,Y],t,w,V0,dV,N)
draw_frames(f,X,Y,t)


print('(packet 1 done)')
plt.show()
# In[ ]:


x_start=-8
x_end=20
y_start=-8
y_end=20

x=np.linspace(x_start,x_end,30)
y=np.linspace(y_start,y_end,30)
X,Y=np.meshgrid(x,y)

w=2
V0=[.4,.1]
K=1.5
dV=[1/(10*K),1/(10*K)]
N=12

t=np.linspace(0,1,8)*32

f=lambda t: Packet_2d([X,Y],t,w,V0,dV,N)
draw_frames(f,X,Y,t)


print('(packet 2 done)')
plt.show()
# In[ ]:


x_start=-20
x_end=20
y_start=-20
y_end=20

x=np.linspace(x_start,x_end,60)
y=np.linspace(y_start,y_end,60)
X,Y=np.meshgrid(x,y)
t=0.2
w=4
V0=[0,0]
K=2.8
dV=[1/(10*K),1/(10*K)]
N=20
x1=-4
x2=4


t=np.linspace(0,1,8)*80

f=lambda t: Packet_2d([X-x1,Y],t,w,V0,dV,N)+Packet_2d([X-x2,Y],t,w,V0,dV,N)
draw_frames(f,X,Y,t)


# In[ ]:


x_start=0
x_end=1
y_start=0
y_end=1

x=np.linspace(x_start,x_end,400)
y=np.linspace(y_start,y_end,400)
X,Y=np.meshgrid(x,y)
t=0.45
w=1
m=2
n=3


# randphases=[np.tan(np.random.random()*np.pi/2) for i in range(200)]
# phases=lambda v: randphases[int(v*100)]

Psi1=Psi_box_2d([X,Y],t,w,m,n)


# print(Psi1.shape)
# Psi_R=np.zeros(shape=Psi1.shape,dtype=np.float64)
# Psi_R[:]=Psi1
# print(Psi1.real[110])


fig=plt.figure(figsize=[16,7])
ax=fig.add_subplot(121)
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')
#ax.contourf(X,Y,Psi1.real,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
#ax.set_xlim(x_start,x_end)
#ax.set_ylim(y_start,y_end)

ax2=fig.add_subplot(122)
ax2.contourf(X,Y,abs(Psi1)**2,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(x_start,x_end)
ax2.set_ylim(y_start,y_end)
#ax2.plot([0,30],[0,30/v0])
#ax2.plot([0,30],[0,30])


print('(simple box and 2d diffraction done)')
plt.show()
# In[ ]:


x_start=0
x_end=1
y_start=0
y_end=1

x=np.linspace(x_start,x_end,400)
y=np.linspace(y_start,y_end,400)
X,Y=np.meshgrid(x,y)
t=0.45
w=2
m=4
n=3

# randphases=[np.tan(np.random.random()*np.pi/2) for i in range(200)]
# phases=lambda v: randphases[int(v*100)]
h=1
#pl=h
#p=h/l
#nl/2=1
#l=2/n
#p=hn/2
px=h*m*np.pi
py=h*n*np.pi
p=[px,py]
#p=gmv
#mc**2=hw
#p=vghw/c**2
#vg=pc**2/(hw)
#=p/hw
vg=[px/(h*w),py/(h*w)]
#vg**2=v/(1+v**2)
g=np.sqrt(1+dot(vg,vg))
V=[vg[0]/g,vg[1]/g]




#=add_2d([np.sqrt((m*np.pi)**2/(w**2+(m*np.pi)**2)),0],[0,np.sqrt((n*np.pi)**2/(w**2+(n*np.pi)**2))])
#print([np.sqrt((m*np.pi)**2/(w**2+(m*np.pi)**2)),0])
print(V)
print(V[0]/V[1])
# Psi1=(Psi_2d([X,Y],t,w,V)-Psi_2d([X,Y],t,w,[-V[0],-V[1]]))+(Psi_2d([X,Y],t,w,[-V[0],V[1]])-Psi_2d([X,Y],t,w,[-V[0],V[1]]))
#Psi_box_2d_x([X,Y],t,w,m)+Psi_box_2d_y([X,Y],t,w,n)
Psi_i=[
    (Psi_2d([X,Y],t,w,[V[0],V[1]])+Psi_2d([X,Y],t,w,[-V[0],-V[1]]))-(Psi_2d([X,Y],t,w,[-V[0],V[1]])+Psi_2d([X,Y],t,w,[V[0],-V[1]]))
    for t in np.linspace(.5,.65,8)]
for i in range(8):
    Psi_i[i][0][0]=-4
    Psi_i[i][0][1]=4
#

# print(Psi1.shape)
# Psi_R=np.zeros(shape=Psi1.shape,dtype=np.float64)
# Psi_R[:]=Psi1
# print(Psi1.real[110])




fig=plt.figure(figsize=[16,16])
for i in range(8):
    ax=fig.add_subplot(4,4,1+i)
    #ax.contourf(X,Y,Psi_i[i].real,30,cmap=cm.coolwarm)
    
    ax.imshow(complex_array_to_rgb(Psi_i[i]),origin='lower',aspect= 'auto')
    #ax.plot([0,1],[0,1])
    #ax.set_xlim(x_start,x_end)
    #ax.set_ylim(y_start,y_end)
# for i in range(8):
#     ax2=fig.add_subplot(4,4,9+i)
#     ax2.contourf(X,Y,abs(Psi_i[i])**2,30,cmap=cm.coolwarm)
#     #ax.plot([0,1],[0,1])
#     ax2.set_xlim(x_start,x_end)
#     ax2.set_ylim(y_start,y_end)
#     #ax2.plot([0,30],[0,30/v0])
#     #ax2.plot([0,30],[0,30])

# # print(Psi1.shape)
# # Psi_R=np.zeros(shape=Psi1.shape,dtype=np.float64)
# # Psi_R[:]=Psi1
# # print(Psi1.real[110])


# fig=plt.figure(figsize=[16,7])
# ax=fig.add_subplot(121)
# ax.contourf(X,Y,Psi1.real,30,cmap=cm.coolwarm)
# #ax.plot([0,1],[0,1])
# ax.set_xlim(x_start,x_end)
# ax.set_ylim(y_start,y_end)

# ax2=fig.add_subplot(122)
# ax2.contourf(X,Y,abs(Psi1)**2,30,cmap=cm.coolwarm)
# #ax.plot([0,1],[0,1])
# ax2.set_xlim(x_start,x_end)
# ax2.set_ylim(y_start,y_end)

print('(4x3 box over time done)')
plt.show()

# In[ ]:


x_start=0
x_end=1
y_start=0
y_end=1

x=np.linspace(x_start,x_end,400)
y=np.linspace(y_start,y_end,400)
X,Y=np.meshgrid(x,y)
#t=0.45
w=1.6
m1=3
n1=6
m2=4
n2=7
m3=4
n3=8



n_frames=12
t=np.linspace(0,3,n_frames)

f=lambda t: Psi_box_2d([X,Y],t,w,m1,n1)+Psi_box_2d([X,Y],t,w,m2,n2)+Psi_box_2d([X,Y],t,w,m3,n3)
draw_frames(f,X,Y,t)


print('(weird box mix done)')
plt.show()
# In[ ]:


x_start=0
x_end=1
y_start=0
y_end=1

x=np.linspace(x_start,x_end,80)
y=np.linspace(y_start,y_end,80)
X,Y=np.meshgrid(x,y)
t=0.2
w=18
q_mean=np.array([30,20])
q_spread=np.array([8,8])


# randphases=[np.tan(np.random.random()*np.pi/2) for i in range(200)]
# phases=lambda v: randphases[int(v*100)]


n_frames=8


t=np.linspace(.8,2.8,n_frames)

f=lambda t: Packet_box_2d([X,Y],t,w,q_mean,q_spread)
draw_frames(f,X,Y,t)


print('(localized bouncing particle in box 1 done)')
plt.show()
# In[ ]:


x_start=0
x_end=1
y_start=0
y_end=1

x=np.linspace(x_start,x_end,160)
y=np.linspace(y_start,y_end,160)
X,Y=np.meshgrid(x,y)
t=0.2
w=120
q_mean=np.array([60,40])
q_spread=np.array([12,12])

n_frames=16

t=np.linspace(.8,3.0,n_frames)

f=lambda t: Packet_box_2d([X,Y],t,w,q_mean,q_spread)
draw_frames(f,X,Y,t)


print('(last thing done)')
plt.show()
print('2d_free_and_box done')
