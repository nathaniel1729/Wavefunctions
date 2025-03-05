
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import cm
from quantum_solutions import Psi,Psi_box
from quantum_solutions import Packet
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
from quantum_solutions import inner_product,normed_overlap,attatch_coeffs,Packet_attatched,Packet_attatched_gs,vg_to_v_list
from quantum_solutions import Psi_m_circular




# x_start=-10
# x_end=10
# y_start=-10
# y_end=10

# x=np.linspace(x_start,x_end,300)
# y=np.linspace(y_start,y_end,300)
# X,Y=np.meshgrid(x,y)
# w=8
# J=1
# n=0
# N=40

# t=np.linspace(0,2,8)
# f=lambda t: Psi_m_circular([X,Y],t,w,J,n,N)
# draw_frames(f,X,Y,t)
# plt.show()


# x_start=-10
# x_end=10
# y_start=-10
# y_end=10

# x=np.linspace(x_start,x_end,300)
# y=np.linspace(y_start,y_end,300)
# X,Y=np.meshgrid(x,y)
# w=8
# J=1
# n=2
# N=40

# t=np.linspace(0,2,8)
# f=lambda t: Psi_m_circular([X,Y],t,w,J,n,N)
# draw_frames(f,X,Y,t)
# plt.show()



x_start=-15
x_end=15
y_start=-15
y_end=15

x=np.linspace(x_start,x_end,300)
y=np.linspace(y_start,y_end,300)
X,Y=np.meshgrid(x,y)
w=8
J=1
t0=0
N=40

n=np.arange(-6,6)#np.linspace(0,2,8)
f=lambda n: Psi_m_circular([X,Y],t0,w,J,n,N)
draw_frames(f,X,Y,n,show_prob=False)
plt.show()





# x_start=-6
# x_end=6
# y_start=-6
# y_end=6

# x=np.linspace(x_start,x_end,300)
# y=np.linspace(y_start,y_end,300)
# X,Y=np.meshgrid(x,y)
# t0=0
# vgx0,vgy0=-1,1
# w=2
# J=2
# n=0
# N=40



# #t=np.linspace(0,5,16)

# nums=np.arange(16)
# t=np.linspace(1,12,12)
# P2_data=np.exp(-((X-3)**2+(Y+3)**2)/.36)*Psi_2d([X,Y],t0,w,vg_to_v_list([vgx0,vgy0]))

# parameters=[]
# for J in np.linspace(2,4,15):
#     for n in [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]:
#         parameters.append([J,n])

# Psi_packet_thisone=Packet_attatched(lambda data,param: Psi_m_circular(data[0],data[1],data[2],param[0],param[1],data[3]),P2_data,[[X,Y],t0,w,N],parameters)

# f=lambda t: Psi_packet_thisone([[X,Y],t,w,N])
# draw_frames(f,X,Y,t)
# plt.show()
