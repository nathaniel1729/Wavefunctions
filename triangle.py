#!/usr/bin/env python
# coding: utf-8


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import cm


# #free particle:

# In[5]:


from merged_math import complex_array_to_rgb



v0=0.4



from merged_math import dot, Psi_2d


h=1


from merged_math import draw_frames



ep_0=1
h_=1
m=1
e=1
E_1=-(m/(2*h_**2))*(e**2/(4*np.pi*ep_0))**2
a=(4*np.pi*ep_0*h_**2)/(m*e**2)
###############








from merged_math import Psi_triangle, Psi_triangle_2,p_to_v


#######################################################################  Plane wave  ###############################################################################################
print('Plane wave')

x_start=0
x_end=1
y_start=0
y_end=1

x=np.linspace(x_start,x_end,400)
y=np.linspace(y_start,y_end,400)
X,Y=np.meshgrid(x,y)
t=.2
w=10
p=[12,4]
vx,vy=p_to_v(p,w)
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
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')
#ax.plot([0,1],[0,1])
#ax.set_xlim(x_start,x_end)
#ax.set_ylim(y_start,y_end)

ax2=fig.add_subplot(122)
ax2.contourf(X,Y,abs(Psi1)**2,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(x_start,x_end)
ax2.set_ylim(y_start,y_end)
ax2.plot([0,30],[0,30/v0])
ax2.plot([0,30],[0,30])


plt.show()

#######################################################################  Zero Error  ###############################################################################################
print('Zero Error')



# In[ ]:



x_start=-1
x_end=1
y_start=-1
y_end=1

x=np.linspace(x_start,x_end,400)
y=np.linspace(y_start,y_end,400)
X,Y=np.meshgrid(x,y)
t=0

q=2
r=1
w=10


# randphases=[np.tan(np.random.random()*np.pi/2) for i in range(200)]
# phases=lambda v: randphases[int(v*100)]

Psi1=Psi_triangle([X,Y],t,w,q,r)




# print(Psi1.shape)
# Psi_R=np.zeros(shape=Psi1.shape,dtype=np.float64)
# Psi_R[:]=Psi1
# print(Psi1.real[110])


fig=plt.figure(figsize=[16,7])
ax=fig.add_subplot(121)
ax.imshow(complex_array_to_rgb(Psi1),origin='lower',aspect= 'auto')
#ax.plot([0,1],[0,1])
#ax.set_xlim(x_start,x_end)
#ax.set_ylim(y_start,y_end)

ax2=fig.add_subplot(122)
ax2.contourf(X,Y,abs(Psi1)**2,30,cmap=cm.coolwarm)
#ax.plot([0,1],[0,1])
ax2.set_xlim(x_start,x_end)
ax2.set_ylim(y_start,y_end)
ax2.plot([0,30],[0,30/v0])
ax2.plot([0,30],[0,30])


plt.show()






#######################################################################  Cycle Bounce  ###############################################################################################
print('Cycle Bounc')
# In[ ]:



x_start=-1.6
x_end=1.6
y_start=-1.2
y_end=2

x=np.linspace(x_start,x_end,200)
y=np.linspace(y_start,y_end,200)
X,Y=np.meshgrid(x,y)
t=0

q1=46
r1=1

q2=47
r2=1

w=10



t=np.linspace(0,5,16)

f=lambda t: Psi_triangle_2([X,Y],t,w,q1,r1)+Psi_triangle_2([X,Y],t,w,q2,r2)
draw_frames(f,X,Y,t)

plt.show()



#######################################################################  Directory  ###############################################################################################
print('Directory')


# In[ ]:



x_start=-1.5
x_end=1.5
y_start=-1.2
y_end=1.8

x=np.linspace(x_start,x_end,200)
y=np.linspace(y_start,y_end,200)
X,Y=np.meshgrid(x,y)
t=0

q=2
r=1
w=10



#t=np.linspace(0,5,16)

nums=np.arange(64)


f=lambda num: Psi_triangle_2([X,Y],t,w,num//8+1,num%8+1)
draw_frames(f,X,Y,nums,8)

plt.show()



#######################################################################  Fundamental Mode test  ###############################################################################################
print('Fundamental Mode test')

# In[ ]:



x_start=-1
x_end=1
y_start=-1
y_end=1

x=np.linspace(x_start,x_end,400)
y=np.linspace(y_start,y_end,400)
X,Y=np.meshgrid(x,y)
t=0

w=8



#t=np.linspace(0,5,16)

nums=np.arange(16)
t=np.linspace(0,1,8)

f=lambda t: (1/2+np.sqrt(3)/2*1j)*(Psi_2d([X,Y],t,w,[.5,0])+Psi_2d([X,Y],t,w,[-.25,np.sqrt(3)/4])+Psi_2d([X,Y],t,w,[-.25,-np.sqrt(3)/4]))+(Psi_2d([X,Y],t,w,[-.5,0])+Psi_2d([X,Y],t,w,[.25,-np.sqrt(3)/4])+Psi_2d([X,Y],t,w,[.25,np.sqrt(3)/4]))
draw_frames(f,X,Y,t)


plt.show()
print('triangle done')