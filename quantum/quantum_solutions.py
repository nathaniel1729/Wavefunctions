
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm

# import numpy as np
# from matplotlib import pyplot as plt
# from quantum_solutions import Psi,Psi_box
# from quantum_solutions import Packet
# from quantum_solutions import complex_array_to_rgb
# from quantum_solutions import Packet_box
# from quantum_solutions import Hermites, psi_harmonic, Psi_harmonic
# v0=0.4
# from quantum_solutions import Packet_harmonic
# from quantum_solutions import dot, Psi_2d, Psi_box_2d_x,  Psi_box_2d_y,  Psi_box_2d, Gaussian
# from quantum_solutions import Packet_box_2d,g_2d,tan_to_sin, add_2d,Packet_2d
# h=1
# from quantum_solutions import draw_frames
# from quantum_solutions import Psi_hydrogen_spherical
# ep_0=1
# h_=1
# m=1
# e=1
# E_1=-(m/(2*h_**2))*(e**2/(4*np.pi*ep_0))**2
# a=(4*np.pi*ep_0*h_**2)/(m*e**2)
# ###############
# from quantum_solutions import Psi_hydrogen_cartesian
# from quantum_solutions import Gaussian_2, non_Gaussian, Packet_hydrogen,Packet_hydrogen2
# from quantum_solutions import Psi_triangle, Psi_triangle_2,p_to_v
# from quantum_solutions import inner_product,normed_overlap,attatch_coeffs,Packet_attatched,vg_to_v_list


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
#free particle:
def Psi(x,t,w,v,phase=0):
    """
    takes x (real # or np array), t (real # or np array), w,v (real numbers) optionally phase (real number), returns a complex number, 
    or complex array if an array was put in. if x and t are arrays, they must have the same shape.
    """
    g=1/np.sqrt(1-v**2)
    return np.exp(1j*w*g*v*x-1j*w*g*t+1j*phase)
def Psi_box(x,t,w,m=0,phase=0):
    """
    takes x (real # or np array), t (real # or np array), w (real number), m (natural number) optionally phase (real number), returns a complex number, 
    or complex array if an array was put in. if x and t are arrays, they must have the same shape.
    """
    v=np.sqrt((m*np.pi)**2/(w**2+(m*np.pi)**2))
    #print(v)
    return Psi(x,t,w,v,phase)-Psi(x,t,w,-v,phase)


g=lambda v:1/np.sqrt(1-v**2)
add=lambda v1,v2:(v1+v2)/(1+v1*v2)
def Packet(x,t,w,N,v0,K,phases=lambda v:0):
    """
    takes x (real # or np array), t (real # or np array), w,N,v0,K (real numbers) optionally phase (function from natural numbers to complex),
    returns a complex number, or complex array if an array was put in. if x and t are arrays, they must have the same shape.
    N and K determines the density and spread of the gaussian in momentum space, v0 determines it's center.
    """
    return sum([Psi(x,t,w,add(v0,v),phase=phases(v))*np.exp(-(K*v*g(v))**2)/N for v in np.linspace(-1,1,N+2)[1:-1]])

def Packet_box(x,t,w,M1,M2):
    """
    takes x (real # or np array), t (real # or np array), w (real #), M1,M2, (natural # with M2>M1)
    returns a complex number, or complex array if an array was put in. if x and t are arrays, they must have the same shape.
    N and K determines the density and spread of the gaussian in momentum space, v0 determines it's center.
    """
    B=sum([np.exp(-4*((m-(M1+M2)/2)/(M2-M1))**2) for m in np.arange(M1,M2)])
    return sum([Psi_box(x,t,w,m)*np.exp(-(5*(m-(M1+M2)/2)/(M2-M1))**2)/B for m in np.arange(M1,M2)])#*



#simple harmonic 



def Hermites(N):
    """
    takes N (natural number), returns a list of lists. each inner list contains the coefficients of a Hermite polynomial.
    """
    hermites=[[1,0,0],[0,2,0,0]]
    for n in range(2,N+1):
        new=[-hermites[n-1][1]]
        for k in range(1,n+1):
            #print(len(hermites[n-1]),k)
            new.append(2*hermites[n-1][k-1]-(k+1)*hermites[n-1][k+1])
        new.append(0)
        new.append(0)
        hermites.append(new)
    #H=[]
    #for n in range(0,N+1):
    #    H.append(lambda x: sum([hermites[n][k]*x**k for k in range(0,n+1)]))
    return hermites
H=lambda x,a_n:sum([a_n_k*x**k for k,a_n_k in enumerate(a_n)])

def psi_harmonic(x,w,m,n,a_n):
    """
    takes x (real # or np array), w,m (real numbers), n (natural number), a_n (list of coeffs for the nth hermite polynomial).
    returns a complex # or complex array if an array was input.
    """
    h=1
    B=np.sqrt(m/h)
    N_n=(abs(n+1/2)/np.pi)**(-2*abs(n+1/2)/np.pi)
    #print(a_n)
    return np.exp(-B**2*x**2/2)*H(B*x,a_n)*N_n
def Psi_harmonic(x,t,m,n,a_n,phase=0):
    """
    takes x, t (real # or np array), m (real number), n (natural number), a_n (list of coeffs for the nth hermite polynomial).
    returns a complex # or complex array if an array was input.
    if x and t are both arrays, they must have the same shape.
    """
    h=1
    w_h=4
    E_n=(n+1/2)*h/w_h
    w=(m+E_n)/h
    #print(a_n)#/np.sqrt(E_n)
    return psi_harmonic(x,w,m,n,a_n)*np.exp(1j*w*t+phase*1j)
x=np.linspace(-3,3)
a=Hermites(48)
# for i in range(48):
#     plt.plot(x,abs(psi_harmonic(x,1,i,a[i])**2))




def Packet_harmonic(x,t,m,N1,N2):
    """
    takes x, t (real # or np array), m (real number), N1,N2 (natural numbers, N2>N1).
    returns a complex # or complex array if an array was input.
    if x and t are both arrays, they must have the same shape.
    uses a gaussian over quantum numbers of the harmonic oscillator between N1 and N2
    """
    a=Hermites(N2)
    Nn=sum([np.exp(-4*((m-(N1+N2)/2)/(N2-N1))**2) for m in np.arange(N1,N2)])
    return sum([Psi_harmonic(x,t,m,n,a[n])*np.exp(-(5*(n-(N1+N2)/2)/(N2-N1))**2)/Nn for n in np.arange(N1,N2)])#*
# m=np.linspace(M1,M2)
# plt.plot(np.exp(-(5*(m-(M1+M2)/2)/(M2-M1))**2))



#2d stuff########################


def dot(u,v):
    """dot product of u,v, works for lists or 1d np arrays. if you put in lists of np arrays, you will get an
    np array containing the dot products for each index in the inner arrays."""
    return sum([u[i]*v[i] for i in range(len(u))])
def Psi_2d(X,t,w,V,phase=0):
    """
    free particle in 2d space with rest frequency w and velocity V. X,V are lists of values, where the values can be real or np arrays. 
    if both X and V use np arrays, they must be the same shape. V should be constant. t,w are real #s. returns  a complex # or a complex array if
    arrays were used for input
    """
    g=1/np.sqrt(1-dot(V,V))
    #print(V[1])
    return np.exp(1j*w*g*dot(V,X)-1j*w*g*t+1j*phase)
def Psi_box_2d_x(X,t,w,m=0,phase=0):
    """
    particle in 2d with walls at x=0,x=1, rest frequency w and quantum # m. X is a 2-element list of values, where the values can be real or np arrays.
    t can be a real # or np array. if both X and t use np arrays, they must be the same shape. w is a real #, m is a natural #.
    returns  a complex # or a complex array if arrays were used for input
    """
    V=[X[0]*0+np.sqrt((m*np.pi)**2/(w**2+(m*np.pi)**2)),0]
    #print(V)
    #print(v)
    return Psi_2d(X,t,w,V,phase)-Psi_2d(X,t,w,[-V[0],-V[1]],phase)
def Psi_box_2d_y(X,t,w,n=0,phase=0):
    """
    particle in 2d with walls at y=0,y=1, rest frequency w and quantum # m. X is a 2-element list of values, where the values can be real or np arrays.
    t can be a real # or np array. if both X and t use np arrays, they must be the same shape. w is a real #, m is a natural #.
    returns  a complex # or a complex array if arrays were used for input
    """
    V=[0,X[1]*0+np.sqrt((n*np.pi)**2/(w**2+(n*np.pi)**2))]
    #print(V)
    #print(v)
    return Psi_2d(X,t,w,V,phase)-Psi_2d(X,t,w,[-V[0],-V[1]],phase)
def Psi_box_2d(X,t,w,m,n):
    """
    particle in a 2d box with walls at y=0,y=1,x=0,x=1, rest frequency w and quantum #s m,n. X is a 2-element list of values, where the values can be real or np arrays.
    t can be a real # or np array. if both X and t use np arrays, they must be the same shape. w is a real #, m,n are natural #s.
    returns  a complex # or a complex array if arrays were used for input
    """
    px=m*np.pi
    py=n*np.pi
    vg=[px/w,py/w]
    g=np.sqrt(1+dot(vg,vg))
    V=[X[0]*0+vg[0]/g,X[1]*0+vg[1]/g]
    return (Psi_2d(X,t,w,[V[0],V[1]])+Psi_2d(X,t,w,[-V[0],-V[1]]))-(Psi_2d(X,t,w,[-V[0],V[1]])+Psi_2d(X,t,w,[V[0],-V[1]]))
    #Psi_box_2d_x(X,t,w,m)*Psi_box_2d_y(X,t,w,n)

def Gaussian(q_mean,q_spread):
    """
    takes two 1d np arrays. the idea is that q_mean is a point in Rn and the entries in q_spread are the range in each dimension. 
    returns an unorganized 1d list of positions in Rn near the point q_mean (each dimension will be a natural number between
    q_mean[i]-q_spread[i] and  q_mean[i]q_spread[i]), and a parallel list of weights corresponding to positions 
    in the first list. INPUTS AND OUTPUTS SHOULD BE MADE OF NATURAL NUMBERS (excpet the weights)
    """
    q_numbers=[q_mean]
    for i,s_i in enumerate(q_spread):
        new=[]
        for dq in range(-s_i,s_i+1):
            v_i=q_mean*0
            v_i[i]=1
            new.extend([q+dq*v_i for q in q_numbers])
            #print(new)
        q_numbers=new
    weights=[np.exp(-sum([(2.5*(q-q_mean)[i]/q_spread[i])**2 for i in range(len(q))])) for q in q_numbers]
    sum_weights=sum(weights)
    weights=[weight/sum_weights for weight in weights]
    return q_numbers,weights
    
def Packet_box_2d(X,t,w_0,q_mean,q_spread):
    """ 
    X is a 2-element list of values, where the values can be real or np arrays. t is a real #, w_0 is a real #. q_mean and q_spread 
    should be length 2 np arrays of natural numbers. the idea is that q_mean is a point in R2 and the entries in q_spread are the 
    spread in each dimension. these should consist of natural numbers, and should not allow negative values after passing through 
    the gaussian function. returns  a complex # or a complex array if arrays were used for input
    """
    q_num,W=Gaussian(q_mean,q_spread)
    return sum([Psi_box_2d(X,t,w_0,q_num[j][0],q_num[j][1])*W[j] for j in range(len(q_num))])#*

g_2d=lambda v:1/np.sqrt(1-dot(v,v))
tan_to_sin=lambda T: T/np.sqrt(1+T**2)
add_2d=lambda v1,v2:[(v1[i]+v2[i])/(1+dot(v1,v2)) for i in range(len(v1))]
def Packet_2d(X,t,w,V0,dV,N):#,phases=lambda v:0#,phase=phases([dV[0]*q[0],dV[1]*q[1]])
    """ 
    X is a 2-element list of values, where the values can be real or np arrays. t is a real #, w is a real #. 
    V0 and dV should be lists of 2 real #s, the x and y components of the average velocity and delta velocity, respectively. 
    N determines how many delta steps the velocity distribution will be created over. returns  a complex # or a complex 
    array if arrays were used for input
    """
    q_num,W=Gaussian(np.array([0,0]),np.array([N,N]))
    Vg=[[dV[0]*q[0],dV[1]*q[1]] for q in q_num]
    g=[np.sqrt(1+dot(Vg_j,Vg_j))for Vg_j in Vg]
    V=[[Vg_j[0]/g_j,Vg_j[1]/g_j] for Vg_j,g_j in zip(Vg,g)]
    return sum([Psi_2d(X,t,w,add_2d(V0,V_j))*W_j for V_j,W_j in zip(V,W)])
    
# q_m=np.array([3,4])
# q_s=np.array([30,5])
# q_num,W=Gaussian(q_m,q_s)
# plt.plot([q[0] for q in q_num],[q[1]+5000*W[j] for j,q in enumerate(q_num)])



# drawing

def draw_frames(f,X,Y,t,width=4,show_prob=True):
    """
    f should be a function of t only. X and Y should be built into it as global variables or referenced in a lambda function or something, 
    and it should return a complex array of the same shape as X and Y. X,Y should be a meshgrid pair covering the space to be plotted. 
    t should be a list or np array of times at which f is to be evaluated. width is how many image boxes should be shown per row (natural number). 
    if show_prob is True, then after all of the wavefunciton images there will be corresponding probability distrubution images.
    """
    x_start,x_end=X[0][0],X[-1][-1]
    y_start,y_end=Y[0][0],Y[-1][-1]
    n_frames=len(t)
    Psi_i=[f(t_i) for t_i in t]
    print((1+int(show_prob)))
    fig=plt.figure(figsize=[16,n_frames/width**2*16*(1+int(show_prob))])
    for i in range(n_frames):
        ax=fig.add_subplot((n_frames//width)*(1+int(show_prob)),width,1+i)
        ax.imshow(complex_array_to_rgb(Psi_i[i]),origin='lower',aspect= 'auto',interpolation='bilinear')
        #ax.contourf(X,Y,Psi_i[i].real,30,cmap=cm.coolwarm)
        #ax.plot([0,1],[0,1])
        #ax.set_xlim(x_start,x_end)
        #ax.set_ylim(y_start,y_end)
    if show_prob:
        for i in range(n_frames):
            ax2=fig.add_subplot((n_frames//width)*(1+int(show_prob)),width,n_frames+1+i)
            ax2.contourf(X,Y,abs(Psi_i[i])**2,30,cmap=cm.coolwarm)
            #ax.plot([0,1],[0,1])
            ax2.set_xlim(x_start,x_end)
            ax2.set_ylim(y_start,y_end)
            #ax2.plot([0,30],[0,30/v0])
            #ax2.plot([0,30],[0,30])



# Hydrogen ######################################



def Laguerre_coeffs(n,k):
    """returns coefficients of the n,k Laguerre polynomial"""
    p0=[0]*(n+k)+[1]+[0]
    fact_n=1
    for i in range(n):
        fact_n=fact_n*(i+1)
        p0=[-p0[j]+j*p0[j+1] for j in range(len(p0)-1)]+[0]
    p0=p0[k:]
    p0=[c/fact_n for c in p0[:-1]]
    return p0
print(Laguerre_coeffs(3,2))
def fact(n):
    if n==0:
        return 1
    return n*fact(n-1)
def Laguerre(n,k):
    """returns a lambda function that can evaluates the the n,k Laguerre polynomial at any real # or np array of reals"""
    a=Laguerre_coeffs(n,k)
    return lambda x: sum(a[i]*x**i for i in range(len(a)))


# In[ ]:


#radial wavefunction
#U(r)=-e**2/(4*pi*ep_0*r)
#



#pi=
#h_=
#e=
#m=
#ep_0=
#n=
#k=
#l=
#r=
#E_1=-(m/(2*h_**2))*(e**2/(4*pi*ep_0))**2
#E(n)=E_1/n**2
#E=E(n)
#k=(-2*m*E)**.5/h
#rho=kr
#rho_0=(m*e**2)/(2*pi*ep_0*h_**2*k)
#L_n^k(x)=Laguerre(n,k)
#v(x)=L_{1/2*rho_0-(l+1)}^{2*l+1}(x)
#u(rho)=rho**(l+1)*e**(-rho)*v(rho)



#R(r)=u(k*r)/r
#Y(theta,phi)=
#Psi(r,theta,phi)=R(r)*Y(theta,phi)
h=1
ep_0=1
h_=1
m=1
e=1
E_1=-(m/(2*h_**2))*(e**2/(4*np.pi*ep_0))**2
a=(4*np.pi*ep_0*h_**2)/(m*e**2)
def R(r,n,l,L):
    """
    radial part of the wave solution for hydrogen. takes r=radius a real (or np array of reals), n,l quantum #s, and L, the Laguerre polynomial 
    that's supposed to be used in this equation. returns a complex # or array depending on if r is and array.
    """
    try:
        N_n=np.sqrt(
            (2/(a*n))**3
            *fact(n-l-1)/(2*n*(fact(n+l))**3)
        )
    except:
        N_n=1/1000000
    exponential_part=np.exp(-r/(a*n))
    polynomial_part=(2*r/(a*n))**l
    laguerre_part=L(2*r/(a*n))
    return N_n*exponential_part*polynomial_part*laguerre_part
#r=np.linspace(0,300,1000)
#L=Laguerre(3-2-1,2*2+1)
#plt.plot(r,R(r,3,2))
def Y(theta,phi,l,m):
    """
    angular part of the wave solution for hydrogen. takes theta, phi reals (or np arrays of reals with the same shape), l,m quantum #s. returns a 
    complex # or array depending on if the angles are arrays.
    """
    return (np.exp(m*theta*1j))*(np.cos((l-abs(m))*phi))
def time_cycle(t,E,phase=0):
    """
    time part of the wave solution for hydrogen. takes t a real # (or np array of real), E a real number, returns complex # or 
    array depending on if the t is an array.
    """
    return np.exp(-1j*(E*t/h+phase))
def Psi_hydrogen_spherical(r,theta,phi,t,n,l,m,phase=0):
    """
    spherical wave solution for hydrogen. takes r, theta, phi reals (or np arrays of reals with the same shape),t a real #, n,l,m quantum #s. returns a 
    complex # or array depending on if the angles are arrays.
    """
    E=E_1/n**2#(.511e6/13.606)*E_1-
    L=Laguerre(n-l-1,2*l+1)
    return R(r,n,l,L)*Y(theta,phi,l,m)*time_cycle(t,E,phase) 
def Psi_hydrogen_cartesian(x,y,z,t,n,l,m,phase=0):
    """
    cartesian wave solution for hydrogen. takes x,y,z reals (or np arrays of reals with the same shape),t a real #, n,l,m quantum #s. returns a 
    complex # or array depending on if the angles are arrays.
    """
    r=np.sqrt(x**2+y**2+z**2)
    theta=np.arctan2(y,x)
    phi=np.arctan2(np.sqrt(x**2+y**2),z)
    return Psi_hydrogen_spherical(r,theta,phi,t,n,l,m,phase)

# hydrogen weird ############

def Gaussian_2(q_mean,q_spread,initial=None):
    """I think this is the same as the other Gaussian?"""
    q_numbers=[q_mean]
    for i,s_i in enumerate(q_spread):
        new=[]
        for dq in range(-s_i,s_i+1):
            v_i=q_mean*0
            v_i[i]=1
            new.extend([q+dq*v_i for q in q_numbers])
            #print(new)
        q_numbers=new
    weights=[np.exp(-(2.5*(q-q_mean)[i]/(q_spread[i]+1))**2) for q in q_numbers]
    q=q_numbers[0]
    #[print((2.5*(q-q_mean)[i]/(q_spread[i]+1))**2) for i in range(len(q))]
    sum_weights=sum(weights)
    weights=[weight/sum_weights for weight in weights]
    return q_numbers,weights
    
def non_Gaussian(q_mean,q_spread,initial):
    """same as the other Gaussian, excep the weights are complex and defined by how much each one contributes
    to the value at the initial position and time"""
    q_numbers=[q_mean]
    for i,s_i in enumerate(q_spread):
        new=[]
        for dq in range(-s_i,s_i+1):
            v_i=q_mean*0
            v_i[i]=1
            new.extend([q+dq*v_i for q in q_numbers])
            #print(new)
        q_numbers=new
    weights=[Psi_hydrogen_cartesian(initial[0],initial[1],initial[2],initial[3],int(q[0]),int(q[1]),int(q[2])).real for q in q_numbers]
    q=q_numbers[0]
    #[print((2.5*(q-q_mean)[i]/(q_spread[i]+1))**2) for i in range(len(q))]
    sum_weights=sum(weights)
    weights=[weight/sum_weights for weight in weights]
    return q_numbers,weights
    

def Packet_hydrogen(x,y,z,t,n_mean,n_s,l_mean,l_s,m_mean,m_s):
    q_num_unfiltered,W_unfiltered=Gaussian_2(np.array([n_mean,l_mean,m_mean]),np.array([n_s,l_s,m_s]))
    #print(W_unfiltered)
    q_num,W=[],[]
    for q_u,w_u in zip(q_num_unfiltered,W_unfiltered):
        if q_u[1]<q_u[0] and abs(q_u[2])<=q_u[1]:
            q_num.append([int(i) for i in q_u])
            W.append(w_u)
    return sum([Psi_hydrogen_cartesian(x,y,z,t,q_num[j][0],q_num[j][1],q_num[j][2])*W[j] for j in range(len(q_num))])
def Packet_hydrogen2(x,y,z,t,n_mean,n_s,l_mean,l_s,m_mean,m_s,initial):
    q_num_unfiltered,W_unfiltered=non_Gaussian(np.array([n_mean,l_mean,m_mean]),np.array([n_s,l_s,m_s]),initial)
    #print(W_unfiltered)
    q_num,W=[],[]
    for q_u,w_u in zip(q_num_unfiltered,W_unfiltered):
        if q_u[1]<q_u[0] and abs(q_u[2])<=q_u[1]:
            q_num.append([int(i) for i in q_u])
            W.append(w_u)
    return sum([Psi_hydrogen_cartesian(x,y,z,t,q_num[j][0],q_num[j][1],q_num[j][2])*W[j] for j in range(len(q_num))])


# triangle ################################

J=lambda n,k: np.pi*np.sqrt((k-1+n/3)**2+(k-1-n/3)**2/3)#2*np.pi/3*np.sqrt(4*q**2+3*r**2)#
triangle_theta=lambda n,k: np.arctan2(k-1+n/3,(k-1-n/3)/np.sqrt(3))#(2/3*(q+r),-1/np.sqrt(3)*(q-r))#
def step(x):
    ep=1/1000
    return abs(x+ep/4)/ep-abs(x-ep/4)/ep+.5
asdf,qwer=np.meshgrid(np.linspace(-1.4,1.4),np.linspace(-1.4,1.4))
def in_triangle(X,Y):
    return step(Y+1)*step(-Y/2-np.sqrt(3)*X/2+1)*step(-Y/2+np.sqrt(3)*X/2+1)
#plt.contourf(asdf,qwer,in_triangle(asdf,qwer))
def Psi_triangle(X,Y,t,q,r,m):
    X2,Y2=X/3,(Y-2)/3
    j=J(q,r)
    theta=triangle_theta(q,r)
    #print(theta)
    a,b,c=np.pi+2*j*np.sin(theta),np.pi+2*j*np.sin(2*np.pi/3-theta),np.pi+2*j*np.sin(2*np.pi/3+theta)
    #print(a,b,c)
    thetas=[theta,2*np.pi/3-theta,2*np.pi/3+theta,4*np.pi/3-theta,4*np.pi/3+theta,-theta]#[:2]
    pre_coeffs=[0,-c,b+c,b,b+a,-a]#[:2]
    ps=[np.array([j*np.cos(theta_i),j*np.sin(theta_i)]) for theta_i in thetas]
    #plt.scatter([p[0] for p in ps],[p[1] for p in ps])
    vs=[p_to_v(p,m) for p in ps]
    Vs=[[0*X2+v[0],0*Y2+v[1]] for v in vs]
    gs=[1/np.sqrt(1-sum(v*v)) for v in vs]
    #print([np.exp(1j*pre_coeff_i) for pre_coeff_i in pre_coeffs])
    return sum(np.exp(1j*pre_coeff_i)*Psi_2d([X2,Y2],t,m,V) for g,pre_coeff_i,V in zip(gs,pre_coeffs,Vs))*in_triangle(X,Y)

def Psi_triangle_2(X,Y,t,q,r,m):
    return Psi_triangle(X,Y,t,(q+1)*3,r+2,m)
def p_to_v(p,m):
    vg=np.array([p_i/m for p_i in p])
    #g=1/cos(theta)
    #v=c*sin(theta)
    #vg=tan(theta)
    v=vg/np.sqrt(1+sum(vg*vg))
    return v
print(np.array([[abs(triangle_theta(q,r)-np.pi/6)<np.pi/6 for r in range(1,6)] for q in range(-6,1)]))



# attatched stuff ###################


def inner_product(P1_data,P2_data):
    g=P1_data.conjugate()*P2_data
    keepgoing=True
    while keepgoing:
        try:
            g=sum(g)
        except:
            keepgoing=False
    return g+0j

def normed_overlap(P1_data,P2_data):
    norm_1=np.sqrt(inner_product(P1_data,P1_data))
    norm_2=np.sqrt(inner_product(P2_data,P2_data))
    #print(norm_1,'word',norm_2)
    if abs(norm_1*norm_2)==0:#abs(norm_1)==0 or abs(norm_2)==0 or 
        return 0
    result=inner_product(P1_data,P2_data)/(norm_1*norm_2)
    return result
def attatch_coeffs(P1,P2_data,parameters):
    """
    P1 is a function with 1 input (can be a list or whatever), P2_data is an object of the same type as the output of P1 (both objects
    must be able to be used in the above inner product function. typically an np array of complex numbers of any shape.) parameters is
    a list of valid inputs to P1 and is the basis that will be used to approximate P2_data as a linear combination of P1 solutions.
    returns a list of pairs of parameters and coefficients
    """
    result=[]
    for parameter in parameters:
        P1_data_i=P1(parameter)
        coeff_i=normed_overlap(P1_data_i,P2_data)
        result.append([parameter,coeff_i])
    return result
def Packet_attatched(P1,P2_data,data_init,parameters):
    """
    P1 is a function with 2 inputs (can be lists or whatever), P2_data is an object of the same type as the output of P1 (both objects
    must be able to be used in the above inner product function. typically an np array of complex numbers of any shape.) data_init is
    a valid first input to P1 containing the initial state within which comparison to P2_data will be done. parameters is
    a list of valid second inputs to P1 and is the basis that will be used to approximate P2_data as a linear combination of P1 solutions.
    the output is a function that takes in a valid first input to P1 and outputs the same type as P1.
    """
    param_coeffs=attatch_coeffs(lambda parameter: P1(data_init,parameter),P2_data,parameters)
    return lambda data:sum(P1(data,pc[0])*pc[1] for pc in param_coeffs)


# In[ ]:


def vg_to_v_list(vg):
    vg=np.array(vg)
    try: 
        v=vg/(np.sqrt(1+sum(vg*vg)))
    except:
        print(vg,1+sum(vg*vg))
    return [v_i for v_i in v]

# momentum circular quantum states #####
def Psi_m_circular(X,t,w_0,J,n,N):
    """
    X: [np array, np array] # position
    t: real # time
    w_0: real # rest frequency
    J: real # magnitude of momentum
    n: int # m_circular quantum number
    N: natural num # sample density
    """
    thetas=np.linspace(0,2*np.pi,N+1)[:-1]
    return sum([Psi_2d(X,t,w_0,p_to_v([J*np.cos(theta),J*np.sin(theta)],w_0))*np.exp(1j*n*theta)/N for theta in thetas])

# grahm-schmidt #############
# def orthogonal_projection(v,basis,basis_norms):
#     return sum([(inner_product(v_i,v)/norm_i**2)*v_i for v_i,norm_i in zip(basis, basis_norms)])

def attatch_coeffs_gs(P1,P2_data,parameters):
    """
    P1 is a function with 1 input (can be a list or whatever), P2_data is an object of the same type as the output of P1 (both objects
    must be able to be used in the above inner product function. typically an np array of complex numbers of any shape.) parameters is
    a list of valid inputs to P1 and is the basis that will be used to approximate P2_data as a linear combination of P1 solutions.
    returns a list of pairs of parameters and coefficients
    """
    basis_1=[]
    b1_norms=[]
    params_1=[]
    for parameter in parameters:
        v_i=P1(parameter)
        norm_i=np.sqrt(inner_product(v_i,v_i))
        if norm_i!=0:
            basis_1.append(v_i)
            b1_norms.append(norm_i)
            params_1.append(parameter)
    basis_2=[basis_1[0]]
    b2_norms=[]
    coeff_transform=[np.array([1]+[0]*(len(basis_1)-1),dtype=np.complex128)]
    for i_,v1_i in enumerate(basis_1[1:]):
        i=i_+1
        v2_i_to_v1_i=np.zeros(len(basis_1),dtype=np.complex128)
        v2_i_to_v1_i[i]+=1
        j=0
        proj_v=0
        for v2_j,norm2_j in zip(basis_2, b2_norms):
            j+=1
            c=(inner_product(v1_i,v2_j)/norm2_j**2)
            v2_i_to_v1_i-=coeff_transform[j]*c
            proj_v+=c*v2_j
        v2_i=v1_i-proj_v
        norm2_i=np.sqrt(inner_product(v2_i,v2_i))
        if norm2_i!=0:
            basis_2.append(v2_i)
            b2_norms.append(norm2_i)
            coeff_transform.append(v2_i_to_v1_i)
    coeff_transform=np.array(coeff_transform)

    coeffs_2=[]
    for v in basis_2:
        coeff_i=normed_overlap(v,P2_data)
        coeffs_2.append(coeff_i)
    coeffs_1=coeff_transform@np.array(coeffs_2)
    return [[param,c] for c,param in zip(coeffs_1,params_1)]
    
def Packet_attatched_gs(P1,P2_data,data_init,parameters):
    """
    P1 is a function with 2 inputs (can be lists or whatever), P2_data is an object of the same type as the output of P1 (both objects
    must be able to be used in the above inner product function. typically an np array of complex numbers of any shape.) data_init is
    a valid first input to P1 containing the initial state within which comparison to P2_data will be done. parameters is
    a list of valid second inputs to P1 and is the basis that will be used to approximate P2_data as a linear combination of P1 solutions.
    the output is a function that takes in a valid first input to P1 and outputs the same type as P1.
    """
    param_coeffs=attatch_coeffs_gs(lambda parameter: P1(data_init,parameter),P2_data,parameters)
    return lambda data:sum(P1(data,pc[0])*pc[1] for pc in param_coeffs)