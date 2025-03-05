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



from quantum_solutions import Psi_triangle, Psi_triangle_2,p_to_v




from quantum_solutions import inner_product,normed_overlap,attatch_coeffs,Packet_attatched,vg_to_v_list



x_start=-1
x_end=1
y_start=-1
y_end=1

x=np.linspace(x_start,x_end,400)
y=np.linspace(y_start,y_end,400)
X,Y=np.meshgrid(x,y)
t0=0
vgx0,vgy0=-2,4
w=8



#t=np.linspace(0,5,16)

nums=np.arange(16)
t=np.linspace(0,1,8)
P2_data=np.exp(-((X-.5)**2+(Y+.5)**2)/.02)*Psi_2d([X,Y],t0,w,vg_to_v_list([vgx0,vgy0]))

parameters=[]
density=10.9
for i in range(-int(density),int(density)):
    parameters.extend([vg_to_v_list([i/np.sqrt(density)+vgx0,j/np.sqrt(density)+vgy0]) for j in range(-int(np.sqrt(density**2-i**2)),int(np.sqrt(density**2-i**2)))])
#[print(p[0]**2+p[1]**2) for p in parameters]
Psi_packet_thisone=Packet_attatched(lambda data,param: Psi_2d(data[0],data[1],data[2],param),P2_data,[[X,Y],t0,w],parameters)
#(Psi_2d([X,Y],t,w,[.5,0])
f=lambda t: Psi_packet_thisone([[X,Y],t,w])
draw_frames(f,X,Y,t)


plt.show()

# In[ ]:



x_start=0
x_end=1
y_start=0
y_end=1

x=np.linspace(x_start,x_end,400)
y=np.linspace(y_start,y_end,400)
X,Y=np.meshgrid(x,y)
t0=0
x0,y0=.2,.4
init_spread=.02
w=24
p0=[-2*w,4*w]

vgx0,vgy0=p0[0]/w,p0[1]/w#-2,4



#t=np.linspace(0,5,16)

nums=np.arange(16)
t=np.linspace(0,1.5,12)
P2_data=np.exp(-((X-x0)**2+(Y-y0)**2)/init_spread)*Psi_2d([X,Y],t0,w,vg_to_v_list([vgx0,vgy0]))


parameters=[]
mystery_coeff=3.14
for i in range(max(1,abs(int(p0[0]/mystery_coeff))-5),abs(int(p0[0]/mystery_coeff))+5):
    parameters.extend([[i,j] for j in range(max(1,abs(int(p0[1]/mystery_coeff))-5),abs(int(p0[1]/mystery_coeff))+5)])
# for i in range(10,20):
#     parameters.extend([[i,j] for j in range(30,40)])
#print(parameters)

data_init=[[X,Y],t0,w]
#data format:[[X,Y],t,w]
#param format: [m,n]

help_function=lambda data,param: Psi_box_2d(data[0],data[1],data[2],param[0],param[1])

Psi_packet_thisone=Packet_attatched(help_function,P2_data,data_init,parameters)


f=lambda t: Psi_packet_thisone([[X,Y],t,w])
draw_frames(f,X,Y,t)


plt.show()

# In[ ]:



######################################################
x_start=-3200
x_end=3200
y_start=-3200
y_end=3200

x=np.linspace(x_start,x_end,100)
y=np.linspace(y_start,y_end,100)
x,y=np.meshgrid(x,y)
t0=0
z=0
x0,y0=-2000,0
init_spread=150
w=8
p0=[0*w,.08*w]

vgx0,vgy0=p0[0]/w,p0[1]/w#-2,4

t=np.linspace(0,300000,12)
#t=np.linspace(-120000,120000,16)


#t=np.linspace(0,5,16)

P2_data=np.exp(-((x-x0)**2+(y-y0)**2)/init_spread)*Psi_2d([x,y],t0,w,vg_to_v_list([vgx0,vgy0]))


parameters=[]
mystery_coeff=3.14
for n in range(1,26):
    for l in range(n):
        for m in range(0,l):
            parameters.append([n,l,m])
# for i in range(10,20):
#     parameters.extend([[i,j] for j in range(30,40)])
#print(parameters)

data_init=[x,y,z,t0]
#data format:[x,y,z,t]
#param format: [n,m,l]

help_function=lambda data,param: Psi_hydrogen_cartesian(data[0],data[1],data[2],data[3],param[0],param[1],param[2])#[0]

Psi_packet_thisone=Packet_attatched(help_function,P2_data,data_init,parameters)


f=lambda t: Psi_packet_thisone([x,y,z,t])
draw_frames(f,x,y,t)


plt.show()

# In[ ]:


x_start=-1.6
x_end=1.6
y_start=-1.2
y_end=2.0

x=np.linspace(x_start,x_end,150)
y=np.linspace(y_start,y_end,150)
X,Y=np.meshgrid(x,y)
t0=0
x0,y0=0,.5
init_spread=.24
w=36
p0=[0.5*w,0*w]

vgx0,vgy0=p0[0]/w,p0[1]/w#-2,4



#t=np.linspace(0,5,16)

nums=np.arange(16)
t=np.linspace(0,1.5,12)
P2_data=np.exp(-((X-x0)**2+(Y-y0)**2)/init_spread)*Psi_2d([X,Y],t0,w,vg_to_v_list([vgx0,vgy0]))


parameters=[]
mystery_coeff=3.14
# for i in range(max(1,abs(int(p0[0]/mystery_coeff))-5),abs(int(p0[0]/mystery_coeff))+5):
#     parameters.extend([[i,j] for j in range(max(1,abs(int(p0[1]/mystery_coeff))-5),abs(int(p0[1]/mystery_coeff))+5)])
for i in range(0,20):
    parameters.extend([[i,j] for j in range(0,20)])
#print(parameters)

data_init=[X,Y,t0,w]
#data format:[[X,Y],t,w]
#param format: [m,n]

help_function=lambda data,param: Psi_triangle_2(data[0],data[1],data[2],param[0],param[1],data[3])

Psi_packet_thisone=Packet_attatched(help_function,P2_data,data_init,parameters)


f=lambda t: Psi_packet_thisone([X,Y,t,w])
draw_frames(f,X,Y,t)


plt.show()

# In[ ]:


m=np.linspace(-10,10,800)
K=-8000
V=1/np.sqrt(3)
n1_m=(m*V+np.sqrt(-K/(4*np.pi**2)*(1+V**2)-m**2))/(1+V**2)
n2_m=(m*V-np.sqrt(-K/(4*np.pi**2)*(1+V**2)-m**2))/(1+V**2)
#plt.plot(m,n2-n1)
plt.plot(m,2*np.pi*V*(n1_m-n2_m)-np.sqrt((-K-4*np.pi**2*n1_m**2))-np.sqrt((-K-4*np.pi**2*n2_m**2)))


# In[ ]:


n1=np.linspace(-15,15,400)
n2=np.linspace(-15,15,400)
n1,n2=np.meshgrid(n1,n2)
plt.contourf(n1,n2,2*np.pi*V*(n1-n2)-np.sqrt((-K-4*np.pi**2*n1**2))-np.sqrt((-K-4*np.pi**2*n2**2)),cmap=cm.coolwarm)
plt.plot(n1_m,n2_m)
for i in range(-30,-20):
    seg=np.linspace(-4,4)
    plt.plot(seg-i/(4*V),seg+i/(4*V))



plt.show()
print('attatched_stuff done')