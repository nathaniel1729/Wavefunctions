
try:
    # import random
    # import math,time
    import numpy as np
    # from PIL import Image, ImageTk
    # #set up windo_w and a few constants
    # import tkinter as tk
    # from tkinter import messagebox, simpledialog,ttk
    # from colors import colormap_np#, set_colormap, get_color
    # import function_math
    # from function_math import set_function_Magnitude,Newton,exp
    pass
except:
    print("imports failed")
    raise



def unwindow(C,W):
    """For C within the windo_w defined by a center and upper right corner, maps C to be within [0,1]x[0,1]"""
    DD=W[1]-W[0]
    CD=C-W[0]+DD
    return (np.real(CD)/(2*np.real(DD)))+(np.imag(CD)/(2*np.imag(DD)))*1j

def window(C,W):
    """For C within [0,1]x[0,1], maps C to be within the windo_w defined by a center and upper right corner."""
    DD=W[1]-W[0]
    return W[0]-DD+(np.real(C)*np.real(DD)*2)+(np.imag(C)*np.imag(DD)*2)*1j

def decanvasify(cx,cy):
    """takes coordinates in the canvas, returns a complex number in [0,1]x[0,1]"""
    #print((cx-7)/725,1-(cy-31)/723)
    return ((cx-11)/725)+(1-(cy-35)/723)*1j
def canvasify(Z):
    """takes a complex number in [0,1]x[0,1], returns coordinates in the canvas"""
    return int(Z.real*725+11),int((1-Z.imag)*723+35)










