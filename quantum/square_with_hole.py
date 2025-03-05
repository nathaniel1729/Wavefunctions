#!/usr/bin/env python
# coding: utf-8


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import cm



v0=0.4
from quantum_solutions import Psi_2d, Psi_box_2d
h=1
from quantum_solutions import draw_frames
ep_0=1
h_=1
m=1
e=1
E_1=-(m/(2*h_**2))*(e**2/(4*np.pi*ep_0))**2
a=(4*np.pi*ep_0*h_**2)/(m*e**2)
###############
from quantum_solutions import Packet_attatched,vg_to_v_list












x_start=0
x_end=1
y_start=0
y_end=1

x=np.linspace(x_start,x_end,400)
y=np.linspace(y_start,y_end,400)
X,Y=np.meshgrid(x,y)
t0=0
x0,y0=.1,.6
init_spread=.015
w=36
p0=[5*w,0*w]

vgx0,vgy0=p0[0]/w,p0[1]/w#-2,4



#t=np.linspace(0,5,16)

nums=np.arange(16)
t=np.linspace(0,1.5,12)
P2_data=np.exp(-((X-x0)**2+(Y-y0)**2)/init_spread)*Psi_2d([X,Y],t0,w,vg_to_v_list([vgx0,vgy0]))


parameters=[]
mystery_coeff=3.14
test_range=10
for i in range(max(1,abs(int(p0[0]/mystery_coeff))-test_range),abs(int(p0[0]/mystery_coeff))+test_range):
    for j in range(max(1,abs(int(p0[1]/mystery_coeff))-test_range),abs(int(p0[1]/mystery_coeff))+test_range):
        if (i*j)%2==0:
            parameters.extend([[i,j]])
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





parameters=[]
for i in range(max(1,abs(int(p0[0]/mystery_coeff))-test_range),abs(int(p0[0]/mystery_coeff))+test_range):
    for j in range(max(1,abs(int(p0[1]/mystery_coeff))-test_range),abs(int(p0[1]/mystery_coeff))+test_range):
        parameters.extend([[i,j]])
# for i in range(10,20):
#     parameters.extend([[i,j] for j in range(30,40)])
#print(parameters)

data_init=[[X,Y],t0,w]
#data format:[[X,Y],t,w]
#param format: [m,n]
C=Psi_box_2d([.5,.5],t0,w,1,1)#this definitely won't work because it's using a low frequency state to try to cancel a high frequency state. the 
#center point will be 0 sometimes but not always.
help_function=lambda data,param: Psi_box_2d(data[0],data[1],data[2],param[0],param[1])*C - Psi_box_2d(data[0],data[1],data[2],1,1)*Psi_box_2d([.5,.5],t0,w,param[0],param[1])

Psi_packet_thisone=Packet_attatched(help_function,P2_data,data_init,parameters)


f=lambda t: Psi_packet_thisone([[X,Y],t,w])
draw_frames(f,X,Y,t)


plt.show()
print('square_with_hole done')
