'''###########################################'''
'''Util file for the new ROM linear parabolic equation solver'''
'''Using NGSolve package for initialization and some matrix computations'''
'''###########################################'''
#algorithm created by: Y. Zhang
#code created by: Y. Huang

from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.webgui import Draw
import scipy.sparse as sp
import numpy as np
import time

#default region as a unit square
#default boundary is all four sides with 0
def Init(h = 1/2**7, p = 1, d = 1, bd = None,
         u0 = None, source = None,
         showMesh = False, showEle = False, 
         showH1 = False, showSou = False, showIni = False):
    import time
    time0 = time.time()
    mesh = Mesh(unit_square.GenerateMesh(maxh=h))
    time1 = time.time()
    print('\nbuild mesh used {} sec'.format(time1-time0))
    if showMesh:
        Draw(mesh)
    if showEle:
        mesh.nv, mesh.ne
    fes = H1(mesh, order=p, dim=d, dirichlet=bd)
    if showH1:
        print(fes)
    u,v = fes.TnT() #u = trial, v = test

    #assembling stiffness matrix
    time2 = time.time()
    A = BilinearForm(fes, symmetric=False)
    A += grad(u)*grad(v)*dx
    A.Assemble()
    time3 = time.time()
    #assembling mass matrix    
    M = BilinearForm(fes, symmetric=False)
    M += u*v*dx
    M.Assemble()
    time4 = time.time()
    
    gfu = GridFunction(fes)
    if u0 != None:
        gfu.Set(u0)
        if showIni:
            Draw(gfu,mesh,"u0",order=p)
    else:
        print('STOP, need initial function')
        return
    
    #assembling load vector
    b = LinearForm(fes)
    if source != None:

        b += source*v*dx
        b.Assemble()
        time5 = time.time()
        print('build M, A, b used {} sec'.format(time5-time2))
        if showSou:
            Draw(source,mesh,"b",order=p)
    else:
        print('STOP, need source function')
        return
    
    return mesh, gfu, fes, M, A, b

'''#######################'''
'''New ROM algorithm below'''
'''#######################'''

def Get_Qmatrix(gfu, fes, M, A, b, l, tol = 1e-14):
    '''algorithm 2.1'''
    A2 = A.mat.CreateMatrix()
    #M = M.mat.CreateMatrix()
    #b = b.vec.CreateVector()
    
    res = gfu.vec.CreateVector()
    res.data = b.vec
    #print(res.data[:10])
    A_star = A2.Inverse(freedofs=fes.FreeDofs())
    gfu.vec.data = A_star*res
    tmp = np.array(gfu.vec.data)
    U_l = tmp
    rows,cols,vals = A.mat.COO()
    A_mat = sp.csr_matrix((vals,(rows,cols)))
    K_l = U_l.T@A_mat@U_l
    #print(U_l)
    #print('i=0 ',K_l)
    for i in range(1,l):
        res.data = M.mat*gfu.vec #Mu_i-1
        gfu.vec.data = A_star*res #Au_i = Mu_i-1
        tmp = np.array(gfu.vec.data)
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
            print('\noptimal r = ',i)
            break
    return Q #Nx(i-1)

def Get_Qmatrix_Alt2(gfu, fes, M, A, B, l, tol = 1e-14):
    #alg 4.1
    #silly way of computing A[x1,x2] = B = [b1, b2]
    U = np.array([0])
    #A2 = A.mat.CreateMatrix()
    #A2.AsVector().data = A.mat.AsVector()
    Astar = A.mat.Inverse(freedofs=fes.FreeDofs())
    res = gfu.vec.CreateVector()
    
    rows,cols,vals = M.mat.COO()
    M_mat = sp.csr_matrix((vals,(rows,cols)))
    rows,cols,vals = A.mat.COO()
    A_mat = sp.csr_matrix((vals,(rows,cols)))
    for i in range(B.shape[1]):
        
        res.FV().NumPy()[:] = B[:,i]
        gfu.vec.data = Astar*res #Ax = b_i
        tmp = np.array(gfu.vec.data)
        if U.shape[0]==1:
            U = tmp
        else:
            U = np.c_[U, tmp]
    
    assert(U.shape[0]!=1)
    for i in range(1,l):
        
        for j in range(B.shape[1],0,-1):
            res.FV().NumPy()[:] = M_mat@U[:,-(j)]
            gfu.vec.data = Astar*res #Ax = Mb_i
            tmp = np.array(gfu.vec.data)
            U = np.c_[U, tmp]
    
    K = U.T@A_mat@U
    #print('U',U.shape)
    #print(U)
    #print('K')
    #print(K.shape)
    #print(K)
    eigVal, eigVec = np.linalg.eig(K)
    
    for i in range(1,len(eigVal)+1):
        if sum(eigVal[:i])/sum(eigVal) >= 1-tol:
            print('\noptimal r = ',i)
            eigVal_tmp = np.sqrt(np.ones(i)/eigVal[:i])
            Q = U@eigVec[:,:i]@np.diag(eigVal_tmp)
            break
    return Q #Nx(i-1)

def ROM_Backward_Solver(gfu, fes, M, A, b, delt, Nt, 
                    hasTime = False, Param_t = None, clock = 0, g1 = None,  
                    l = 10, tol = 1e-14):
    import time
    '''backward Euler and BDF2'''
    time0 = time.time()
    if g1 == None:
        Q = Get_Qmatrix(gfu, fes, M, A, b, l, tol)
    else:
        b0 = np.array(gfu.vec.data) #u0
        b1 = np.array(g1.vec.data) #g1
        B = np.c_[b0, b1]
        Q = Get_Qmatrix_Alt2(gfu, fes, M, A, B, l, tol)
    time1 = time.time()
    print('Get Qmatrix used: {} sec'.format(time1-time0))
    
    time0 = time.time()
    rows,cols,vals = M.mat.COO()
    M_mat = sp.csr_matrix((vals,(rows,cols)))
    rows,cols,vals = A.mat.COO()
    A_mat = sp.csr_matrix((vals,(rows,cols)))
    if not hasTime:
        b_vec = np.array(b.vec.data)
    else:
        assert(Param_t != None)
        clock += delt
        Param_t.Set(clock)
        b.Assemble()
        b_vec = np.array(b.vec.data)
        
    M_r = Q.T@M_mat@Q
    A_r = Q.T@A_mat@Q
    #print(A_r)
    b_r = Q.T@b_vec
    
    a_r = np.array([0]*M_r.shape[0]) #a_r0

    tmp = np.linalg.solve(M_r/delt + A_r, b_r.T) #a_r1
    a_r = np.c_[a_r, tmp]
    for i in range(1,Nt):
        if hasTime:#update load vector b(t)
            clock += delt
            Param_t.Set(clock)
            b.Assemble()
            b_vec = np.array(b.vec.data)
            b_r = Q.T@b_vec
        
        #a_rn >= 2
        tmp = np.linalg.solve(3/(2*delt) * M_r + A_r,
                              M_r@(2*a_r[:,-1] - a_r[:,-2]/2)/delt + b_r) 
        a_r = np.c_[a_r, tmp]
    time1 = time.time()
    print('Backward Solver used: {} sec'.format(time1-time0))
    return Q, a_r[:,-1]

def ROM_Solve(h, T = 1, order = 1, dim = 1, boundary = None, sce = None,
              hasTime = False, check = False, 
              showSou = False, showSol = False):
    #solve linear parabolic equation using new ROM
    import time
    time_0 = time.time()
    
    delt = h
    Nt = int(np.ceil(T/delt))
    
    if hasTime:
        t = Parameter(0.0)
        clock = 0.0
        t.Set(0.0)
        Source = exp(t)*(x*(1-x)*y*(1-y)+2*y*(1-y)+2*x*(1-x))
        
        exact = exp(T)*x*(1-x)*y*(1-y)
        #initialization
        if check:
            u0 = x*(1-x)*y*(1-y)
            mesh, gfu, fes, M, A, b = Init(h, p = order, d = dim, bd = boundary,
                        u0 = u0, source = Source, showIni = True)
        time_1 = time.time()
        
        #getting solution
        Q, a_n = ROM_Backward_Solver(gfu, fes, M, A, b, delt, Nt, 
                                     hasTime, t, clock)
    else:
        assert(not check)

        #initialization
        mesh, gfu, fes, M, A, b = Init(h, p = order, d = dim, bd = boundary,
                                       u0 = 0, source = sce, 
                                       showSou = showSou)
        time_1 = time.time()
        
        #getting solution
        Q, a_n = ROM_Backward_Solver(gfu, fes, M, A, b, delt, Nt)
    
    time_2 = time.time()
    
    time_i = time_1-time_0
    time_s = time_2-time_1
    time_t = time_2-time_0
    
    #save solution
    sol = gfu.vec.CreateVector()
    sol.FV().NumPy()[:] = Q@a_n
    gfu.vec.data = sol
    if check:
        err = sqrt(Integrate((gfu-exact)**2, mesh))
        print ("\nL2-error:", err)
    if showSol:
        Draw(gfu,mesh,"sol",order=order)

    print()
    print('When h = 1/{}'.format(Nt))
    print('initialization used {} sec'.format(time_i))
    print('generage Qmatrix + backward solve used {} sec'.format(time_s))
    print('total used {} sec'.format(time_t))
    
    if check:
        return Q, a_n, err, time_t
    else:
        return Q, a_n, time_t

'''############################'''
'''Standard FEM algorithm below'''
'''############################'''  

def StdFEM_Backward_Solver(gfu, fes, M, A, b, delt, Nt,
                  hasTime = False, Param_t = None, clock = 0):
    import time
    #solve FEM by backward Euler and BDF2
    mstar = M.mat.CreateMatrix()
    mstar.AsVector().data = (1/delt) * M.mat.AsVector() +  A.mat.AsVector()
    invmstar = mstar.Inverse(freedofs=fes.FreeDofs()) 
    
    mstar2 = M.mat.CreateMatrix()
    mstar2.AsVector().data = (3/2)*M.mat.AsVector() + delt*A.mat.AsVector()
    invmstar2 = mstar2.Inverse(freedofs=fes.FreeDofs()) 
    
    time0 = time.time()
    res = gfu.vec.CreateVector() 
    res2 = gfu.vec.CreateVector()
    
    if hasTime:
        assert(Param_t != None)
        clock += delt
        Param_t.Set(clock)
        b.Assemble()
    
    U_l = np.array(gfu.vec.data) 
    
    res2.FV().NumPy()[:] = U_l #u_n-2
    #1st step, backward Euler
    
    #res.data = delt * b.vec - delt * A.mat * gfu.vec
    gfu.vec.data = invmstar * b.vec
    
    U_l = np.c_[U_l, np.array(gfu.vec.data)]
    #print(U_l[:,-1].shape)
    res.FV().NumPy()[:] = U_l[:,-1] #u_n-1
    
    for i in range(1,Nt):
        #BDF2
        if hasTime:
            assert(Param_t != None)
            clock += delt
            Param_t.Set(clock)
            b.Assemble()
        gfu.vec.data = invmstar2 * (delt*b.vec + 2*M.mat*res - 0.5*M.mat*res2)
        
        U_l = np.c_[U_l, np.array(gfu.vec.data)]
        res2.FV().NumPy()[:] = U_l[:,-2] #u_n-2
        res.FV().NumPy()[:] = U_l[:,-1] #u_n-1
    
        
    time1 = time.time()
    print('Backward Solver used: {} sec'.format(time1-time0))
    return U_l[:,-1]

def FEM_Solve(h, T = 1, order = 1, dim = 1, boundary = None, sce = None,
              hasTime = False, check = False, 
              showSou = False, showSol = False):
    import time
    #solve linear parabolic equation using standard FEM
    time_0 = time.time()
    
    delt = h
    Nt = int(np.ceil(T/delt))
    if hasTime:
        t = Parameter(0.0)
        clock = 0.0
        t.Set(0.0)
        Source = exp(t)*(x*(1-x)*y*(1-y)+2*y*(1-y)+2*x*(1-x))
        
        exact = exp(T)*x*(1-x)*y*(1-y)
        #initialization
        if check:
            u0 = x*(1-x)*y*(1-y)
            mesh, gfu, fes, M, A, b = Init(h, p = order, d = dim, 
                        bd = boundary, u0 = u0, source = Source, showIni = False)
        time_1 = time.time()
        
        #getting solution
        u_n = StdFEM_Backward_Solver(gfu, fes, M, A, b, delt, Nt, 
                                 hasTime, t, clock)
    else:
        assert(not check)
        #initialization
        mesh, gfu, fes, M, A, b = Init(h, p = order, d = dim, bd = boundary,
                                         u0 = 0, source = sce, showSou = showSou)
        time_1 = time.time()
        
        #getting solution
        u_n = StdFEM_Backward_Solver(gfu, fes, M, A, b, delt, Nt)
    
    time_2 = time.time()
    
    time_i = time_1-time_0
    time_s = time_2-time_1
    time_t = time_2-time_0
    
    #save solution
    sol = gfu.vec.CreateVector()
    sol.FV().NumPy()[:] = u_n
    gfu.vec.data = sol
    if check:
        err = sqrt(Integrate((gfu-exact)**2, mesh))
        print ("\nL2-error:", err)
    if showSol:
        Draw(gfu,mesh,"sol",order=order)
    print()
    print('When h = 1/{}'.format(Nt))
    print('initialization used {} sec'.format(time_i))
    print('generage Qmatrix + backward solve used {} sec'.format(time_s))
    print('total used {} sec'.format(time_t))
    
    if check:
        return u_n, err, time_t
    else:
        return u_n, time_t
    
    