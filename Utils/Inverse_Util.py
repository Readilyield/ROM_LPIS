'''#######################################################'''
'''Util funtions for the linear parabolic equation solvers'''
'''Used NGSolve package'''
'''#######################################################'''
#created by: Y. Huang

from ngsolve import *
from ngsolve.webgui import Draw
from netgen.geom2d import unit_square
from ngsolve.meshes import MakeStructured2DMesh
import scipy.sparse as sp
import numpy as np

'''#######'''
'''Classes'''
'''#######'''

class inverse_obj:
    '''creates an object to hold all variables and flags in one problem'''
    def __init__(self, h, order, dim, boundary, T, dt, u0 = 0, source = 0,
                 showMes = False, showEle = False,
                 showH1 = False, showIni = False, showSou = False,
                 showSol = False, showErr = False, opt = False, sqr = False):
        '''initializes basic properties, type flags, and boolean flags'''
        self.h = h; self.ord = order; self.dim = dim
        self.bdy = boundary; self.T = T; self.dt = dt
        self.u0 = u0; self.sou = source
        if isinstance(u0,fem.CoefficientFunction) or isinstance(u0,int):
            self.u0T = 1  #function->type 1
        elif isinstance(u0,np.ndarray):
            self.u0T = 2  #value array->type 2
        else:
            self.u0T = 0
        if isinstance(source,fem.CoefficientFunction):
            self.soT = 1  #function->type 1
        elif isinstance(source,np.ndarray):
            self.soT = 2  #value array->type 2
        else:
            self.soT = 0
        self.showMes = showMes; self.showEle = showEle
        self.showH1 = showH1; self.showIni = showIni; self.showSou = showSou
        self.showSol = showSol; 
        self.showErr = showErr
        self.sqr = sqr
    
    def get_mesh(self, sqxN = 1):
        '''obtains the mesh and basic matrices via NGSolve'''
        assert(self.u0T != 0)
        assert(self.soT != 0)
        mesh, gfu, fes, m, a, b = initialize(self.h, self.ord, self.dim,
                                            self.bdy, self.u0, self.u0T,
                                            self.sou, self.soT, self.sqr, 
                                            sqxN, self.showMes, 
                                            self.showEle, self.showH1, 
                                            self.showIni, self.showSou)
        self.mesh = mesh; self.gfu = gfu; self.fes = fes
        self.m = m; self.a = a; self.sou = b
        if isinstance(b,fem.CoefficientFunction):
            self.bT = 1  #function->type 1
        elif isinstance(b,np.ndarray):
            self.bT = 2  #value array->type 2

    def solve_fem(self, save = False, anim = False, star = False):
        '''solves the problem using stdfem'''
        assert(hasattr(self,'mesh'))
        #assert(self.soT == 2)
        #self.sou = get_bdy(self.fes, self.sou, self.soT)
        if not star:
            res =  fem_back_solver(self.ord, self.mesh, self.gfu, self.fes, 
                                   self.m, self.a, self.sou, self.T, self.dt,
                                   self.showSol, anim)
        else:
            PT = self.sou
            res =  fem_back_solver_star(self.ord, self.mesh, self.gfu, self.fes, 
                                   self.m, self.a, PT, self.T, self.dt,
                                   self.showSol, anim)
        assert(isinstance(res,np.ndarray))
        if save:
            self.fem_uT = res
        return res

    def solve_rom(self, save = False, star = False):
        '''solves the problem using new rom'''
        assert(hasattr(self,'mesh'))
        #assert(self.soT == 2)
        #self.sou = get_bdy(self.fes, self.sou, self.soT)
        if not star:
            res =  rom_back_solver(self.ord, self.mesh, self.gfu, self.fes, 
                                   self.m, self.a, self.sou, self.T, self.dt,
                                   self.showSol)
        else:
            PT = self.sou
            res =  rom_back_solver_star(self.ord, self.mesh, self.gfu, self.fes, 
                                        self.m, self.a, PT, self.T, self.dt,
                                        self.showSol)
        assert(isinstance(res,np.ndarray))
        if save:
            self.rom_uT = res
        return res

class opt_obj(inverse_obj):
    '''a subclass to inverse_obj, 
    a container for time-dependent optimal control prob'''
    def __init__(self, opt = True, pr_t = None, **kwargs):
        super().__init__(**kwargs)
        self.opt = opt
        self.pr_t = pr_t
    def solve_optfem(self, save = False):
        '''solves the problem using stdfem'''
        assert(hasattr(self,'mesh'))
        if self.opt == True:
            assert(hasattr(self,'pr_t'))
            res =  fem_opt_solver(self.ord, self.mesh, self.gfu, self.fes, 
                           self.m, self.a, self.pr_t, self.sou, self.T, self.dt,
                           self.showSol)
        else:
            res =  fem_back_solver(self.ord, self.mesh, self.gfu, self.fes, 
                                   self.m, self.a, self.sou, self.T, self.dt,
                                   self.showSol)
        assert(isinstance(res,np.ndarray))
        if save:
            self.fem_uT = res
        return res
    
    def solve_optfem_star(self, Un, save = False):
        '''solves the conjugated problem using stdfem'''
        assert(hasattr(self,'mesh'))
        res =  fem_opt_solver_star(self.ord, self.mesh, self.gfu, self.fes, 
                                   self.m, self.a, Un, 
                                   self.sou, self.T, self.dt,
                                   self.showSol)
        return res

'''#######################################'''
'''Initialization and other util functions'''
'''#######################################'''

def sparse_matrix(matrix):
    '''turns NGSolve matrix to a sparse matrix'''
    rows,cols,vals = matrix.mat.COO()
    Mat = sp.csr_matrix((vals,(rows,cols)))
    return Mat

def draw_this(obj, arr):
    '''displays an np.ndarray in the mesh'''
    gf = GridFunction(obj.fes)
    tmp = gf.vec.CreateVector()
    tmp.FV().NumPy()[:] = arr
    gf.vec.data = tmp
    Draw(gf,obj.mesh,'this',order=obj.ord)

def inv_error(obj, sou, res):
    '''displays L2error result v. exact'''
    assert(isinstance(sou,np.ndarray))
    assert(isinstance(res,np.ndarray))
    
    sou2 = obj.gfu.vec.CreateVector()
    sou2.FV().NumPy()[:] = sou
    gfu_sou = GridFunction(obj.fes)
    gfu_sou.vec.data = sou2
    
    res2 = obj.gfu.vec.CreateVector()
    res2.FV().NumPy()[:] = res
    gfu_res = GridFunction(obj.fes)
    gfu_res.vec.data = res2
    
    return sqrt(Integrate((gfu_sou-gfu_res)**2, obj.mesh))

def inv_norm(obj, res):
    '''displays L2norm of the result'''
    assert(isinstance(res,np.ndarray))
    
    res2 = obj.gfu.vec.CreateVector()
    res2.FV().NumPy()[:] = res
    gfu_res = GridFunction(obj.fes)
    gfu_res.vec.data = res2
    
    return sqrt(Integrate((gfu_res)**2, obj.mesh))

def inv_n_norm(obj, res):
    '''displays n-norm of the result'''
    assert(isinstance(res,np.ndarray))
    
    return np.sqrt(np.dot(res,res)*(obj.h**2))
    
    
def initialize(h = 1/2**5, order = 1, dim = 1,
         boundary = None, u_0 = None, u_0Type = 1, 
         source = None, sourceType = 1, square = False, sqxN = 1,
         showMesh = False, showEle = False,
         showH1 = False, showIni = False, showSou = False):
    '''returns mesh, marices, initial condition, and source term'''
    '''returns source term as an array
       square: Make a 2D structured mesh'''
    assert(h > 0)
    if square: 
        nnx = int(np.ceil(1/h))*sqxN-1
        nny = int(np.ceil(1/h))-1
        print('mesh size: ',(nnx+1,nny+1))
        mapping = lambda x,y : (sqxN*x-1,y-1)
        #mesh = MakeStructured2DMesh(quads=False, nx=2*nnx ,ny=2*nny, mapping=mapping)
        mesh = MakeStructured2DMesh(nx=nnx, ny=nny, mapping=mapping)
        #print('num. of vertices:',mesh.nv)
    else:    
        mesh = Mesh(unit_square.GenerateMesh(maxh=h))
    if showMesh:  #displays mesh
        Draw(mesh)
    if showEle: #displays number of vertices and elements in the mesh
        print('num. of vertices:',mesh.nv)
        print('num. of elements:',mesh.ne)
        
    fes = H1(mesh, order=order, dim=dim, dirichlet=boundary)

    if showH1:  #displays FE space configuration
        print(fes)
    u,v = fes.TnT()  #u = trial function, v = test function
    
    #assembling stiffness matrix
    a = BilinearForm(fes, symmetric=False)
    a += grad(u)*grad(v)*dx
    a.Assemble()
    #assembling mass matrix    
    m = BilinearForm(fes, symmetric=False)
    m += u*v*dx
    m.Assemble()
    
    gfu = GridFunction(fes)
    if u_0Type == 1:  #Coe. function type
        gfu.Set(u_0)
        if showIni:
            Draw(gfu,mesh,"u0",order=order)
    elif u_0Type == 2:  #array type
        print('STOP, u0 type: array, undeveloped')
        return
    else:
        print('STOP, u0 type is wrong')
        return

    b = LinearForm(fes)

    if sourceType == 1: #Coe. function type
        b += source*v*dx
        b.Assemble()
        #sou = GridFunction(fes)
        #sou.Set(source)
        #tmp = sou.vec.CreateVector()
        b_val = np.array(b.vec.data)
        if showSou:
            Draw(source,mesh,'sou',order=order)
    elif sourceType == 2:
        m_mat = sparse_matrix(m)
        b_val = m_mat@source
    else:
        print('Error: wrong source type')
        return
    return mesh, gfu, fes, m, a, b_val

# def get_bdy(fes, source, sourceType = 2):
#     '''returns ndarray with value as the load vector'''
#     assert(sourceType == 2)
#     func = GridFunction(fes)
#     func.Set(1,BND)
        
#     q = np.array(func.vec.data)
#     f = np.dot(q,np.dot(q,source))

#     return f

'''##################################################'''
'''Util function for FEM solver'''
'''Using NGSolve package for some matrix computations'''
'''##################################################'''

def fem_back_solver(order = 1, mesh = None, gfu = None, fes = None, 
                    m = None, a = None, 
                    source = None, T = None, dt = None, 
                    showSol = False, anim = False):
    #time-independent by default
    '''solves the parabolic equation by std FEM (BDF2)'''
    assert(isinstance(source,np.ndarray) or 
           isinstance(source,fem.CoefficientFunction) or
           isinstance(source,int) or isinstance(source,float))
    
    #solve FEM by backward Euler and BDF2
    mstar = m.mat.CreateMatrix()
    mstar.AsVector().data =  m.mat.AsVector() + dt*a.mat.AsVector()
    invmstar = mstar.Inverse(freedofs=fes.FreeDofs()) 
    
    mstar2 = m.mat.CreateMatrix()
    mstar2.AsVector().data = (3/2)*m.mat.AsVector() + dt*a.mat.AsVector()
    invmstar2 = mstar2.Inverse(freedofs=fes.FreeDofs()) 

    #print('hey')     
    #print(isinstance(source,np.ndarray))
    if isinstance(source,np.ndarray):
        f = gfu.vec.CreateVector()
        f.FV().NumPy()[:] = source
    elif (isinstance(source,fem.CoefficientFunction) or
         isinstance(source,int) or isinstance(source,float)):
        b = LinearForm(fes)
        u,v = fes.TnT()
        b += source*v*dx
        b.Assemble()
        f = b.vec
    
    res = gfu.vec.CreateVector() 
    res2 = gfu.vec.CreateVector()
    
    U_l = np.array(gfu.vec.data) 
    
    res2.FV().NumPy()[:] = U_l #u_n-2
    #1st step, backward Euler

    gfu_2 = GridFunction(fes)
    res.FV().NumPy()[:] = U_l
    gfu_2.vec.data = invmstar * (dt*f + m.mat*res2)
    U_l = np.c_[U_l, np.array(gfu_2.vec.data)]
    #print(U_l[:,-1].shape)
    res.FV().NumPy()[:] = U_l[:,-1] #u_n-1
    
    Nt = int(np.ceil(T/dt))
    for i in range(1,Nt):
        #BDF2
        gfu_2.vec.data = invmstar2 * (dt*f + 2*m.mat*res - 0.5*m.mat*res2)
        
        U_l = np.c_[U_l, np.array(gfu_2.vec.data)]
        res2.FV().NumPy()[:] = U_l[:,-2] #u_n-2
        res.FV().NumPy()[:] = U_l[:,-1] #u_n-1
    
    res.FV().NumPy()[:] = U_l[:,-1]
    gfu_2.vec.data = res
    
    if showSol:
        Draw(gfu_2,mesh,"sol", order=order)

    
    if anim: ## for sol animation
        return U_l
    else:
        return U_l[:,-1]

def fem_back_solver_star(order = 1, mesh = None, gfu = None, fes = None, 
                         m = None, a = None, source = None, 
                         T = None, dt = None, 
                         showSol = False, anim = False):
    #time-independent source term by default
    '''solves the parabolic conjugate equation by std FEM (BDF2)'''
    ''' -dP/dt - dP^2/dx^2 = 0 => solve dU/dt - dU^2/dx^2 = 0'''
    '''U(t) = P(T-t), U(0) = P(T) = source'''
    assert(isinstance(source,np.ndarray))
    zero = np.zeros(source.shape) #source = zero vector
    #solve FEM by backward Euler only
        #solve FEM by backward Euler and BDF2
    mstar = m.mat.CreateMatrix()
    mstar.AsVector().data =  m.mat.AsVector() + dt*a.mat.AsVector()
    invmstar = mstar.Inverse(freedofs=fes.FreeDofs())  #(M+dtA)^-1
    
    mstar2 = m.mat.CreateMatrix()
    mstar2.AsVector().data = (3/2)*m.mat.AsVector() + dt*a.mat.AsVector()
    invmstar2 = mstar2.Inverse(freedofs=fes.FreeDofs()) 
    
    res = gfu.vec.CreateVector() 
    res2 = gfu.vec.CreateVector()
    
    U_l = source
    
    res2.FV().NumPy()[:] = U_l #u_n-2
    #1st step, backward Euler # M(u1 - u0) + dtAu1 = 0

    gfu_2 = GridFunction(fes)
    res.FV().NumPy()[:] = U_l
    gfu_2.vec.data = invmstar * (m.mat*res2)
    U_l = np.c_[U_l, np.array(gfu_2.vec.data)]
    #print(U_l[:,-1].shape)
    res.FV().NumPy()[:] = U_l[:,-1] #u_n-1
    
    Nt = int(np.ceil(T/dt))
    for i in range(1,Nt):
        #BDF2
        gfu_2.vec.data = invmstar2 * (2*m.mat*res - 0.5*m.mat*res2)
        
        U_l = np.c_[U_l, np.array(gfu_2.vec.data)]
        res2.FV().NumPy()[:] = U_l[:,-2] #u_n-2
        res.FV().NumPy()[:] = U_l[:,-1] #u_n-1
    
    res.FV().NumPy()[:] = U_l[:,-1]
    gfu_2.vec.data = res
    
    if showSol:
        Draw(gfu_2,mesh,"sol", order=order)

    
    if anim: ## for sol animation
        return U_l
    else:
        return U_l[:,-1]


def fem_opt_solver(order = 1, mesh = None, gfu = None, fes = None, 
                    m = None, a = None, Param_t = None,
                    source = None, T = None, dt = None, 
                    showSol = False):
    #time-dependent by default
    '''solves the parabolic equation by std FEM (BDF2)'''
    ''' dU/dt - dU^2/dx^2 = f '''
    assert(isinstance(source,np.ndarray) or 
           isinstance(source,fem.CoefficientFunction) or
           isinstance(source,int) or isinstance(source,float))
    
    #solve FEM by backward Euler and BDF2
    mstar = m.mat.CreateMatrix()

    mstar.AsVector().data = m.mat.AsVector() + dt * a.mat.AsVector()
    invmstar = mstar.Inverse(freedofs=fes.FreeDofs())
    
    mstar2 = m.mat.CreateMatrix()
    mstar2.AsVector().data = (3/2)*m.mat.AsVector() + dt*a.mat.AsVector()
    invmstar2 = mstar2.Inverse(freedofs=fes.FreeDofs()) 

    #print('hey')     
    #print(isinstance(source,np.ndarray))
    ini_time = Param_t.Get()
    Param_t.Set(ini_time+dt)
    if isinstance(source,np.ndarray):
        f = gfu.vec.CreateVector()
        f.FV().NumPy()[:] = source[:,1]
    elif (isinstance(source,fem.CoefficientFunction) or
         isinstance(source,int) or isinstance(source,float)):
        b = LinearForm(fes)
        u,v = fes.TnT()
        b += source*v*dx
        b.Assemble()
        f = b.vec
    
    res = gfu.vec.CreateVector() 
    res2 = gfu.vec.CreateVector()
    
    U_l = np.array(gfu.vec.data) #u_0
    
    res2.FV().NumPy()[:] = U_l #u_n-2
    #1st step, backward Euler

    gfu_2 = GridFunction(fes)

    gfu_2.vec.data = invmstar * (dt*f) #u_1
    U_l = np.c_[U_l, np.array(gfu_2.vec.data)]
    #print(U_l[:,-1].shape)
    res.FV().NumPy()[:] = U_l[:,-1] #u_n-1
    
    tmp_time = Param_t.Get()
    Nt = int(np.ceil(T/dt))
    ###Entering the iteration loop###
    for i in range(1,Nt):
        #BDF2
        if isinstance(source,np.ndarray):
            f.FV().NumPy()[:] = source[:,i+1]
        elif (isinstance(source,fem.CoefficientFunction) or
              isinstance(source,int)):
            tmp_time += dt
            Param_t.Set(tmp_time)
            b.Assemble()
            f = b.vec
        
        #res.data = dt * f - dt * a.mat * U_l[:,-1]
        #gfu_2.vec.data += invmstar * res
        gfu_2.vec.data = invmstar2 * (dt*f + 2*m.mat*res - 0.5*m.mat*res2)
        #u_n, n>= 2
        U_l = np.c_[U_l, np.array(gfu_2.vec.data)]
        res2.FV().NumPy()[:] = U_l[:,-2] #u_n-2
        res.FV().NumPy()[:] = U_l[:,-1] #u_n-1
    
    res.FV().NumPy()[:] = U_l[:,-1]
    gfu_2.vec.data = res
    
    if showSol:
        Draw(gfu_2,mesh,"sol", order=order)
    Param_t.Set(ini_time)
    return U_l

def fem_opt_solver_star(order = 1, mesh = None, gfu = None, fes = None, 
                        m = None, a = None, Un = None,
                        source = None, T = None, dt = None, 
                        showSol = False):
    #time-dependent by default
    '''solves the parabolic conjugate equation by std FEM (BDF2)'''
    ''' -dP/dt - dP^2/dx^2 = U - U_obs '''
    assert(isinstance(source,np.ndarray))
    #solve FEM by backward Euler only
    mstar = m.mat.CreateMatrix()
    #flipped sign for M
    mstar.AsVector().data =  m.mat.AsVector() + dt*a.mat.AsVector()
    invmstar = mstar.Inverse(freedofs=fes.FreeDofs()) 
    
    mstar2 = m.mat.CreateMatrix()
    #flipped sign for M
    mstar2.AsVector().data = (-1/2)*m.mat.AsVector() + dt*a.mat.AsVector()
    invmstar2 = mstar2.Inverse(freedofs=fes.FreeDofs()) 

    ### solve parabolic equation in backwards!!! ###

    f = gfu.vec.CreateVector()
    f.FV().NumPy()[:] = source[:,-1]  #y_n - yd_n
    
    res = gfu.vec.CreateVector()  #u_n-1 or u_k+1 
    res2 = gfu.vec.CreateVector()  #u_n or u_k+2 
    
    res2.FV().NumPy()[:] = Un
    #1st step, backward Euler

    gfu_2 = GridFunction(fes)
    gfu_2.vec.data = invmstar * (dt*f)
    U_l = np.c_[np.array(gfu_2.vec.data), Un]
    res.FV().NumPy()[:] = U_l[:,0]
    
    Nt = int(np.ceil(T/dt))
    ###Entering the iteration loop###
    for i in range(Nt-1,0,-1):  #Not using BDF2
        f.FV().NumPy()[:] = source[:,i]
        #flipped signs in the bracket
        gfu_2.vec.data = invmstar * (dt*f + m.mat*res)
        #gfu_2.vec.data = invmstar2*(dt*f + (3/2)*m.mat*res2 - 2*m.mat*res)
        '''u_k, 0 <= k <= Nt-2'''
        U_l = np.c_[np.array(gfu_2.vec.data), U_l]
        res2.FV().NumPy()[:] = U_l[:,1] #u_k+2
        res.FV().NumPy()[:] = U_l[:,0] #u_k+1
    
    res.FV().NumPy()[:] = U_l[:,1]
    gfu_2.vec.data = res
    
    if showSol:
        Draw(gfu_2,mesh,"sol", order=order)
    #print(U_l.shape)
    return U_l

'''##################################################'''
'''Util functions for the new ROM solver'''
'''Using NGSolve package for some matrix computations'''
'''##################################################'''
#algorithm created by: Y. Zhang
#code created by: Y. Huang

def get_Qmatrix(fes, M, A, b, l, tol = 1e-14):
    '''algorithm 2.1'''
    A2 = A.mat.CreateMatrix()
    gfu_2 = GridFunction(fes)
    res = gfu_2.vec.CreateVector()
    res.data = b
    
    A_star = A2.Inverse(freedofs=fes.FreeDofs())
    gfu_2.vec.data = A_star*res
    tmp = np.array(gfu_2.vec.data)
    U_l = tmp
    rows,cols,vals = A.mat.COO()
    A_mat = sp.csr_matrix((vals,(rows,cols)))
    K_l = U_l.T@A_mat@U_l
    #print(U_l)
    #print('i=0 ',K_l)
    for i in range(1,l):
        res.data = M.mat*gfu_2.vec #Mu_i-1
        gfu_2.vec.data = A_star*res #Au_i = Mu_i-1
        tmp = np.array(gfu_2.vec.data)
        U_l = np.c_[U_l, tmp] #add new u_i to U_l
        #get alpha and beta
        if len(K_l.shape) == 0: #scalar
            alpha = U_l[:,-2].T@A_mat@tmp
        else:
            alpha = np.hstack((K_l[-1, 1:],U_l[:,-2].T@A_mat@tmp))
        beta = tmp.T@A_mat@tmp
        
        #assemble new K_i
        if len(K_l.shape) == 0: #scalar
            K_tmp = np.hstack((K_l, alpha.T))
        else:
            K_tmp = np.c_[K_l, alpha.T]
        botm = np.hstack((alpha, beta))
        K_l = np.vstack((K_tmp, botm))
        #print('i = {}'.format(i))
        #print(K_l)
        eigVal, eigVec = np.linalg.eig(K_l)
        if eigVal[-1] <= tol:
            
            eigVal_tmp = np.sqrt(np.ones(len(eigVal)-1)/eigVal[:-1])
            Q = U_l@eigVec[:,:-1]@np.diag(eigVal_tmp)
            #print('\noptimal r = ',i)
            break
    return Q #Nx(i-1)


def rom_back(fes, M, A, b, delt, Nt, 
             l = 10, tol = 1e-14, star = False, a0 = None):

    '''backward Euler and BDF2'''

    Q = get_Qmatrix(fes, M, A, b, l, tol)
    
    rows,cols,vals = M.mat.COO()
    M_mat = sp.csr_matrix((vals,(rows,cols)))
    rows,cols,vals = A.mat.COO()
    A_mat = sp.csr_matrix((vals,(rows,cols)))

    b_vec = np.array(b.data)
        
    M_r = Q.T@M_mat@Q
    A_r = Q.T@A_mat@Q
    #print(A_r)
    b_r = Q.T@b_vec
    if star:
        assert(isinstance(a0,np.ndarray))
        a_r = Q.T@(M_mat@a0) #a_r0
        b_r = np.array([0]*M_r.shape[0])
    else:  
        a_r = np.array([0]*M_r.shape[0]) #a_r0

    tmp = np.linalg.solve(M_r + delt*A_r, a_r+delt*b_r) #a_r1
    a_r = np.c_[a_r, tmp]
    for i in range(1,Nt):
        
        #a_rn >= 2
        tmp = np.linalg.solve(3/(2*delt) * M_r + A_r,
                              M_r@(2*a_r[:,-1] - a_r[:,-2]/2)/delt + b_r) 
        a_r = np.c_[a_r, tmp]
    
    return Q, a_r[:,-1]

def rom_back_solver(order = 1, mesh = None, gfu = None, fes = None, 
                    m = None, a = None, 
                    source = None, T = None, dt = None, showSol = False):
    #time-independent by default
    '''solves the linear parabolic equation using new ROM'''
    assert(isinstance(source,np.ndarray) or 
           isinstance(source,fem.CoefficientFunction) or
           isinstance(source,int) or isinstance(source,float))
    
    Nt = int(np.ceil(T/dt))

    if isinstance(source,np.ndarray):
        f = gfu.vec.CreateVector()
        f.FV().NumPy()[:] = source
    elif (isinstance(source,fem.CoefficientFunction) or
         isinstance(source,int) or isinstance(source,float)):
        b = LinearForm(fes)
        u,v = fes.TnT()
        b += source*v*dx
        b.Assemble()
        f = b.vec
    
    Q, a_n = rom_back(fes, m, a, f, dt, Nt)
    
    solution = Q@a_n
    #save solution
    sol = gfu.vec.CreateVector()
    sol.FV().NumPy()[:] = solution
    gfu.vec.data = sol
    
    if showSol:
        Draw(gfu,mesh,"sol",order=order)
   
    return solution

def rom_back_solver_star(order = 1, mesh = None, gfu = None, fes = None, 
                         m = None, a = None, 
                         source = None, T = None, dt = None, showSol = False):
    #time-independent by default
    '''solves the linear parabolic equation using new ROM'''
    assert(isinstance(source,np.ndarray))
    
    Nt = int(np.ceil(T/dt))

    if isinstance(source,np.ndarray):
        f = gfu.vec.CreateVector()
        f.FV().NumPy()[:] = [0]*source.shape[0]
    elif (isinstance(source,fem.CoefficientFunction) or
         isinstance(source,int) or isinstance(source,float)):
        b = LinearForm(fes)
        u,v = fes.TnT()
        b += source*v*dx
        b.Assemble()
        f = b.vec
    
    Q, a_n = rom_back(fes, m, a, f, dt, Nt, star = True, a0 = source)
    
    solution = Q@a_n
    #save solution
    sol = gfu.vec.CreateVector()
    sol.FV().NumPy()[:] = solution
    gfu.vec.data = sol
    
    if showSol:
        Draw(gfu,mesh,"sol",order=order)
   
    return solution

