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


from quantum_solutions import complex_array_to_rgb



from quantum_solutions import Packet_box


from quantum_solutions import Hermites, psi_harmonic, Psi_harmonic


v0=0.4
from quantum_solutions import Packet_harmonic



from quantum_solutions import dot, Psi_2d, Psi_box_2d_x,  Psi_box_2d_y,  Psi_box_2d, Gaussian
from quantum_solutions import Packet_box_2d,g_2d,tan_to_sin, add_2d,Packet_2d


h=1


from quantum_solutions import draw_frames


from quantum_solutions import Psi_hydrogen_spherical


ep_0=1
h_=1
m=1
e=1
E_1=-(m/(2*h_**2))*(e**2/(4*np.pi*ep_0))**2
a=(4*np.pi*ep_0*h_**2)/(m*e**2)
###############



r=np.linspace(0,1200)
theta=np.linspace(0,2*np.pi)
r,theta=np.meshgrid(r,theta)
t=0
phi=(0.4)*np.pi
n=4
l=3
m=3

plt.contourf(r*np.cos(theta), r*np.sin(theta),Psi_hydrogen_spherical(r,theta,phi,t,n,l,m).real,cmap=cm.coolwarm)

plt.show()

# In[ ]:

from quantum_solutions import Psi_hydrogen_cartesian

x=np.linspace(-600,600)
y=np.linspace(-600,600)
x,y=np.meshgrid(x,y)
t=0
z=0
n=4
l=2
m=2

plt.contourf(x,y,Psi_hydrogen_cartesian(x,y,z,t,n,l,m).real,cmap=cm.coolwarm)

plt.show()

# In[ ]:



x=np.linspace(-600,600)
y=np.linspace(-600,600)
x,y=np.meshgrid(x,y)
z=0
n=4
l=2
m=l
t=np.linspace(-2400,2400,16)

f=lambda t: Psi_hydrogen_cartesian(x,y,z,t,n,l,m)
draw_frames(f,x,y,t)

plt.show()

# In[ ]:



n1=4
l1=2
m1=l1

n2=4
l2=2
m2=-l2

x=np.linspace(-600,600)
y=np.linspace(-600,600)
x,y=np.meshgrid(x,y)
z=0
t=np.linspace(-12000,12000,16)

f=lambda t: Psi_hydrogen_cartesian(x,y,z,t,n1,l1,m1)+Psi_hydrogen_cartesian(x,y,z,t,n2,l2,m2)
draw_frames(f,x,y,t)

plt.show()

# In[ ]:


n1=4
l1=1
m1=1

n2=5
l2=0
m2=0

x=np.linspace(-600,600)
y=np.linspace(-600,600)
x,y=np.meshgrid(x,y)
z=0
t=np.linspace(-12000,12000,16)

f=lambda t: Psi_hydrogen_cartesian(x,y,z,t,n1,l1,m1)+Psi_hydrogen_cartesian(x,y,z,t,n2,l2,m2)
draw_frames(f,x,y,t)


plt.show()
# In[ ]:


n1=4
l1=1
m1=0

n2=4
l2=2
m2=0

x=np.linspace(-600,600)
y=np.linspace(-600,600)
x,y=np.meshgrid(x,y)
z=0
t=np.linspace(-120000,120000,16)

f=lambda t: Psi_hydrogen_cartesian(x,z,y,t,n1,l1,m1)+Psi_hydrogen_cartesian(x,y,z,t,n2,l2,m2)
draw_frames(f,x,y,t)


# In[ ]:
plt.show()
print('hydrogen_1 done')