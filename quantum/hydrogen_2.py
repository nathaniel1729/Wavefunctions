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

from quantum_solutions import Psi_hydrogen_cartesian


from quantum_solutions import Gaussian_2, non_Gaussian, Packet_hydrogen,Packet_hydrogen2




n_m=16
l_m=10
m_m=5

n_s=3
l_s=3
m_s=3

x=np.linspace(-2400,2400)
y=np.linspace(-2400,2400)
x,y=np.meshgrid(x,y)
z=0
t=np.linspace(-1000000,1000100,16)
#print(Psi_hydrogen_cartesian(x,y,z,0,int(np.int32(7)),2,-1))
#print('and')
#plt.plot(Packet_hydrogen(x,y,z,0,n_m,n_s,l_m,l_s,m_m,m_s).real)
0
f=lambda t: Packet_hydrogen(x,y,z,0,n_m,n_s,l_m,l_s,m_m,m_s)#Psi_hydrogen_cartesian(x,y,z,t,6,3,0)#n_m,n_s,l_m,l_s,m_m,m_s)
draw_frames(f,x,y,t)

plt.show()

# In[ ]:


n_m=21
l_m=12
m_m=0

n_s=8
l_s=6
m_s=4


x0=16
y0=2230
z0=24
t0=0

x=np.linspace(-3600,3600)
y=np.linspace(-3600,3600)
x,y=np.meshgrid(x,y)
z=0
t=np.linspace(-2828282,3838383,16)
#print(Psi_hydrogen_cartesian(x,y,z,0,int(np.int32(7)),2,-1))
#print('and')
#plt.plot(Packet_hydrogen(x,y,z,0,n_m,n_s,l_m,l_s,m_m,m_s).real)
0
f=lambda t: Packet_hydrogen2(x,y,z,0,n_m,n_s,l_m,l_s,m_m,m_s,[x0,y0,z0,t0])
draw_frames(f,x,y,t)

plt.show()

# In[ ]:



qs=[]
qs.append([8,6,5,.1])
qs.append([9,5,4,.2])
qs.append([10,4,3,.4])
qs.append([11,3,2,.2])
qs.append([12,2,1,.1])
qs.append([13,1,0,.2])
qs.append([14,0,-1,.1])
x=np.linspace(-2000,2000)
y=np.linspace(-2000,2000)
x,y=np.meshgrid(x,y)
z=0
t=np.linspace(-24000,400000,16)

f=lambda t: sum([Psi_hydrogen_cartesian(x,y,z,t,q[0],q[1],q[2])*q[3] for q in qs])
draw_frames(f,x,y,t)

plt.show()

# In[ ]:



qs=[]
disti=[.2,-.6,-.2,.2,.6,.2]
distj=[.1,.2,.6,.9,1,.9,.6,.2,.1]
for i in range(6):
    for j in range(9):
        qs.append([23+i+j,10-j,8-j,distj[j]*disti[i]/(sum(distj)*sum(disti))])

x=np.linspace(-12000,12000,200)
y=np.linspace(-12000,12000,200)
x,y=np.meshgrid(x,y)
z=0
t=np.linspace(-4800000,40000000,16)

f=lambda t: sum([Psi_hydrogen_cartesian(x,y,z,t,q[0],q[1],q[2],(q[0]+q[1])*np.pi/3)*q[3] for q in qs])
draw_frames(f,x,y,t)

plt.show()

# In[ ]:


n0=1
l0=0
m0=0


x=np.linspace(-200,200,100)
y=np.linspace(-200,200,100)
x,y=np.meshgrid(x,y)
z=0
t=0
q=np.arange(0,16)
#print(q%4)
c1 =lambda q: ((.5-(q%4)/16)**.5)
c2= lambda q: ((.5+(q%4)/16)**.5)
kill_radius=20
kill_center=lambda x,y,r:(abs((x**2+y**2)*3-3-r)-abs((x**2+y**2)*3-3-r-1)+1)
def superimpose(n1,l1,m1,n2,l2,m2,c1,c2,x,y,r=10*(1+q%4)):
    coeff=kill_center(x,y,r)*np.sqrt(abs(x**2+y**2))
    part1=c1*Psi_hydrogen_cartesian(x,z,y,t,n1,l1,m1)#int(n0+q%4),int(l0+q//4),int(m0+q%4))
    part2=c2*Psi_hydrogen_cartesian(x,z,y,t,n2,l2,m2)
    return coeff*(part1+part2)
#superimpose(int(n0+q%4),int(l0+q//4),int(m0+q%4),n0,l0,m0,c1(q),c2(q),x,y,10*(1+q%4))
Cs=[[1,1,2,2],
   [.5,1,.25,.1],
   [.05,.01,.01,.01],
   [.05,.01,.01,.01]]

rs=[[1,2,320,480],
   [1,2,320,480],
   [1,6,30,240],
   [1,6,160,240]]
f=lambda q: superimpose(int(n0+q%4),int(l0+q//4),int(m0+q%4),n0,l0,m0,1,Cs[q//4][q%4],x*(1/3+2/3*(q%4)),y*(1/3+2/3*(q%4)),20*(rs[q//4][q%4]))
#kill_center(x,y,10*(1+q%4))*abs(x**2+y**2)*(c1(q)*Psi_hydrogen_cartesian(x,z,y,t,int(n0+q%4),int(l0+q//4),int(m0+q%4))+c2(q)*Psi_hydrogen_cartesian(x,z,y,t,n0,l0,m0))#
draw_frames(f,x,y,q)
#plt.plot()
#print((abs((3**2+3**2)*3-3-11)-abs((3**2+3**2)*3-3-12))+1)

plt.show()

# In[ ]:


n0=2
l0=1
m0=1


x=np.linspace(-200,200,100)
y=np.linspace(-200,200,100)
x,y=np.meshgrid(x,y)
z=0
t=0
q=np.arange(0,16)
#print(q%4)
c1 =lambda q: ((.5-(q%4)/16)**.5)
c2= lambda q: ((.5+(q%4)/16)**.5)
kill_radius=20
kill_center=lambda x,y,r:(abs((x**2+y**2)*3-3-r)-abs((x**2+y**2)*3-3-r-1)+1)
def superimpose(n1,l1,m1,n2,l2,m2,c1,c2,x,y,r=10*(1+q%4)):
    coeff=kill_center(x,y,r)*np.sqrt(abs(x**2+y**2))
    part1=c1*Psi_hydrogen_cartesian(x,y,z,t,n1,l1,m1)#int(n0+q%4),int(l0+q//4),int(m0+q%4))
    part2=c2*Psi_hydrogen_cartesian(x,y,z,t,n2,l2,m2)
    return coeff*(part1+part2)
#superimpose(int(n0+q%4),int(l0+q//4),int(m0+q%4),n0,l0,m0,c1(q),c2(q),x,y,10*(1+q%4))
C1s=[[1,0,0,0],
   [80,80,0,0],
   [400,350,350,0],
   [800,600,4800,32000]]
C2s=[[1,0,0,0],
   [1,1,0,0],
   [1,1,1,0],
   [1,1,1,1]]

rs=[[1,2,320,480],
   [1,2,320,480],
   [1,6,30,240],
   [1,6,160,240]]
f=lambda q: superimpose(int(n0+q//4),int(l0+q%4),int(m0+q%4),n0,l0,m0,C1s[q//4][q%4],C2s[q//4][q%4],x*(1+1/2*(q//4)),y*(1+1/2*(q//4)),-1)
#superimpose(int(n0+q%4),int(l0+q//4),int(m0+q%4),n0,l0,m0,1,Cs[q//4][q%4],x*(1/3+2/3*(q%4)),y*(1/3+2/3*(q%4)),20*(rs[q//4][q%4]))
#kill_center(x,y,10*(1+q%4))*abs(x**2+y**2)*(c1(q)*Psi_hydrogen_cartesian(x,z,y,t,int(n0+q%4),int(l0+q//4),int(m0+q%4))+c2(q)*Psi_hydrogen_cartesian(x,z,y,t,n0,l0,m0))#
draw_frames(f,x,y,q)
#plt.plot()
#print((abs((3**2+3**2)*3-3-11)-abs((3**2+3**2)*3-3-12))+1)

plt.show()

# In[ ]:


n0=1
l0=0
m0=0


x=np.linspace(-200,200,100)
y=np.linspace(-200,200,100)
x,y=np.meshgrid(x,y)
z=0
t=0
q=np.arange(0,16)
#print(q%4)
c1 =lambda q: ((.5-(q%4)/16)**.5)
c2= lambda q: ((.5+(q%4)/16)**.5)
kill_radius=20
kill_center=lambda x,y,r:(abs((x**2+y**2)*3-3-r)-abs((x**2+y**2)*3-3-r-1)+1)
def superimpose(n1,l1,m1,n2,l2,m2,c1,c2,x,y,r=10*(1+q%4)):
    coeff=1#kill_center(x,y,r)*np.sqrt(abs(x**2+y**2))
    part1=c1*Psi_hydrogen_cartesian(x,y,z,t,n1,l1,m1)#int(n0+q%4),int(l0+q//4),int(m0+q%4))
    part2=c2*Psi_hydrogen_cartesian(x,y,z,t,n2,l2,m2)
    return coeff*(part1+part2)
#superimpose(int(n0+q%4),int(l0+q//4),int(m0+q%4),n0,l0,m0,c1(q),c2(q),x,y,10*(1+q%4))
C1s=[[15,0,0,0],
   [80,80,0,0],
   [150,80,110,0],
   [160,100,150,180]]
C2s=[[1,0,0,0],
   [1,1,0,0],
   [1,1,1,0],
   [1,1,1,1]]

rs=[[1,2,320,480],
   [1,2,320,480],
   [1,6,30,240],
   [1,6,160,240]]
f=lambda q: superimpose(int(n0+q//4+1),int(l0+q%4+1),int(m0+q%4+1),int(n0+q//4),int(l0+q%4),int(m0+q%4),C1s[q//4][q%4],C2s[q//4][q%4],x*(1+1/2*(q//4)),y*(1+1/2*(q//4)),-1)
#superimpose(int(n0+q%4),int(l0+q//4),int(m0+q%4),n0,l0,m0,1,Cs[q//4][q%4],x*(1/3+2/3*(q%4)),y*(1/3+2/3*(q%4)),20*(rs[q//4][q%4]))
#kill_center(x,y,10*(1+q%4))*abs(x**2+y**2)*(c1(q)*Psi_hydrogen_cartesian(x,z,y,t,int(n0+q%4),int(l0+q//4),int(m0+q%4))+c2(q)*Psi_hydrogen_cartesian(x,z,y,t,n0,l0,m0))#
draw_frames(f,x,y,q)
#plt.plot()
#print((abs((3**2+3**2)*3-3-11)-abs((3**2+3**2)*3-3-12))+1)

plt.show()
print('hydrogen_2 done')