'''#######################################################'''
'''Util funtions for image test functions (letters) input'''
'''#######################################################'''
#created by: Y. Huang

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from ngsolve.webgui import Draw
from ngsolve import GridFunction
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.animation
from matplotlib.animation import FuncAnimation, PillowWriter 

'''################'''
'''Noise corruption'''
'''################'''

def noise(arr,level = 0, std = 1e-3):
    '''add random noise to img array'''
    assert(0 <= level <= 1)
    cp = np.copy(arr)
    sz = len(arr)
    if level > 0:
        num = int(np.ceil(sz*level))
        assert(0 < num < sz)
        noise = np.random.normal(0,std,num)
        ind = np.random.choice(sz-1,num,replace=False)
        cp[ind] = cp[ind] + noise
    return cp

'''###########'''
'''Image input'''
'''###########'''

def rgb2gray(path, check = False, sqxN = 1):
    '''provides image path'''
    #turns img into grayscale
    img = mpimg.imread(path)
    res = np.dot(img[...,:3], [0.2990, 0.5870, 0.1140])
    if check:
        nx,ny = res.shape
        out = np.flip(np.reshape(np.array(res), (nx*ny,)))
        res2heatmap(out, nx, ny, sqxN)
        print('shape: ',res.shape)
    return res

def gray2bw(img, check = False, sqxN = 1):
    '''provides img matrix'''
    #binarize grayscale to black = 1, white = 0
    res = 1.0*(img <= 0.5)
    if check:
        nx,ny = res.shape
        out = np.flip(np.reshape(np.array(res), (nx*ny,)))
        res2heatmap(out, nx, ny, sqxN)
        print('shape: ',res.shape)
    return res

def img2source(obj, path, checkSou = False,
               checkGray = False, checkBw = False, sqxN = 1):
    '''provides image path'''
    #turns img into mx1 ndarray and returns
    img = rgb2gray(path, checkGray, sqxN)
    bw = gray2bw(img, checkBw, sqxN)
    nx,ny = bw.shape
    res = np.flip(np.reshape(np.array(bw), (nx*ny,)))
    if checkSou:
        f1 = obj.gfu.vec.CreateVector()
        f1.FV().NumPy()[:] = res
        gfu_1 = GridFunction(obj.fes)
        gfu_1.vec.data = f1
        Draw(gfu_1,obj.mesh,'sou',order=obj.ord)
    return res

def img2arr(path, checkGray = False, checkBw = False, sqxN = 1):
    '''provides image path'''
    #turns img into mx1 ndarray and returns
    assert(sqxN >= 1 and isinstance(sqxN, int))
    img = rgb2gray(path, checkGray, sqxN)
    bw = gray2bw(img, checkBw, sqxN)
    nx,ny = bw.shape
    res = np.flip(np.array(bw),1)
    res = np.flip(np.reshape(res, (nx*ny,)))
    return res

'''###########'''
'''Image output'''
'''###########'''

def res2heatmap(res, nx, ny, sqxN = 1):
    '''visualizes the result in a heatmap'''
    out = np.reshape(res, (nx,ny))
    out = np.flip(out,0)
    assert(sqxN >= 1 and isinstance(sqxN, int))
    if sqxN > 1:
        fig = plt.figure(figsize=(sqxN*5., 5.))
    else:
        fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 1),  # creates 2x2 grid of axes
                     axes_pad=0.2,  # pad between axes in inch.
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="edge",
                     cbar_size="10%",
                     cbar_pad=0.5)
    for ax in grid:
        img = ax.imshow(out, cmap='inferno', extent = [0, sqxN*1, 0, 1], aspect='auto')
        ax.cax.colorbar(img)
    plt.show()
    
def plot_2D(data_info,title,input_label,output_label,
            axis_bounds=None,xscale=None,yscale=None,bloc='best',anno=None):
    '''
    NOTES: Plots multiple 2D data on one graph.
    INPUT: 
        data_info = list of lists with structure:
            ith list = ith data information, as list
            ith list[0] = [input, output]
            ith list[1] = desired color for ith data
            ith list[2] = legend label for ith data
        title = string with desired title name
        input_label = string with name of input data
        output_label = string with name of output data
        axis_bounds = list with structure: [xmin, xmax, ymin, ymax]
        xscale = string with x axis scale description
        yscale = string with y axis scale description
    '''
    fig = plt.figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)

    for info_cache in data_info:
        if len(data_info) > 1:
            if 'o' in info_cache[1]:
#                 if len(info_cache[0][1]) > 0:
#                     mksize = max(20/int(len(info_cache[0][1])),7)
#                 else:
#                     mksize = 7
                mksize = 7
                alp = 0.5
            else:
                alp = 0.8
                mksize = 10
            plt.plot(info_cache[0][0],info_cache[0][1],
                     info_cache[1],label=info_cache[2],
                     linewidth=3,markersize=mksize, alpha = alp)
             
        else:
            plt.plot(info_cache[0][0],info_cache[0][1],
                     info_cache[1],label=info_cache[2],
                     linewidth=3,markersize=10)
    if anno is not None:
        info_cache = data_info[0]
        plt.text(info_cache[0][0][-1],info_cache[0][1][-1],str(anno[-1])) 
        plt.text(info_cache[0][0][0],info_cache[0][1][0],str(anno[0]))
        
    plt.title(title,fontsize=24)
    plt.xlabel(input_label,fontsize=20)
    plt.ylabel(output_label,fontsize=20)
    if axis_bounds is not None:
        plt.axis(axis_bounds)
    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)
    plt.legend(loc=bloc,fontsize=14)
    plt.show()

def getAnimation(res, nx, ny, fix = False, sqxN = 1):
    '''produces an animated gif for time-step snapshots'''
    assert(sqxN >= 1 and isinstance(sqxN, int))
    if fix:
        fr = len(res)-1
    else:  
        fr = res.shape[1]-1
    if sqxN > 1:
        fig,ax = plt.subplots(figsize = (sqxN*4,4))
    else:
        fig,ax = plt.subplots(figsize = (6,6))
    def animate(i):
        ax.clear()
        ax.axis('off')
        if fix:
            out = np.reshape(res[i], (nx,ny))
            out = np.flip(out,0)
        else:  
            out = np.reshape(res[:,i], (nx,ny))
            out = np.flip(out,0)
        ax.imshow(out, cmap='inferno', extent = [0, sqxN*1, 0, 1], aspect='auto')
        if fix:
            ax.set_title('Step = %03d'%(i))
        else:
            ax.set_title('T = %03d'%(i))

        
    ani =  matplotlib.animation.FuncAnimation(fig,animate,frames=fr,
                                              interval=200,blit=False)

    ani.save('letter.gif',  writer='pillow', fps=7)
    ani.show()