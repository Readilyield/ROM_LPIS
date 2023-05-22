'''########################################################'''
'''Util funtions for the inverse source problem: CG solvers'''
'''Used NGSolve package'''
'''########################################################'''
#created by: Y. Huang

from ngsolve import *
from Inverse_Util import inverse_obj, sparse_matrix, inv_error, inv_norm, inv_n_norm
import scipy.sparse as sp
import numpy as np
import time

'''#########'''
'''CG solvers'''
'''#########'''

def CG_FEM_solver(obj, f_0, lam, tol2 = 1e-10,
                  ite = False, ite_num = 0, anim = False, prin = True, star = False):
    '''Use CG to solve for the source term, 
       incorporated NGSolve for std FEM'''
    #algorithm provided by Y. Zhang
    #assert(isinstance(obj,inverse_obj))
    assert(isinstance(f_0,fem.CoefficientFunction) or isinstance(f_0,int)
           or isinstance(f_0,float) or isinstance(f_0,np.ndarray))
    
    f0s = []
    
    if hasattr(obj,'fem_uT'):
        uT = obj.fem_uT
    else:
        print('No uT, error')
        return
    #get mass matrix
    m_mat = sparse_matrix(obj.m)
    
    tm0 = time.time()
    #Sf0
    if not isinstance(f_0,np.ndarray):
        obj.sou = f_0
        Sf0 = obj.solve_fem()#this is stable
        Sor = GridFunction(obj.fes)
        Sor.Set(f_0)
        #Draw(Sor,obj.mesh,'?',order=obj.ord)
        f0 = np.array(Sor.vec.data)#this is stable
    else:
        obj.sou = m_mat@f_0
        Sf0 = obj.solve_fem()
        f0 = f_0
    f0s.append(f0)
    
    if ite_num == 1:
        return f0
    #S(uT-S(f0))
    #print('\nEr: ',np.sqrt(np.dot((uT-Sf0),m_mat@(uT-Sf0))))
    
    obj.sou = m_mat@(uT-Sf0)
    tmp = obj.solve_fem(star = star)
    r0 = tmp - lam*f0 
    p0 = tmp - lam*f0 
    k = 1; error = 1
    error = np.sqrt(np.dot(p0,m_mat@p0))
    #print('||p0|| ', error)

    if ite_num == 0:
        #print('check')
        while error >= tol2:
            obj.sou = m_mat@p0
            Sp0 = obj.solve_fem()  #Sp0 = S(p0)
            obj.sou = m_mat@Sp0
            SSp0 = obj.solve_fem(star = star)
            Ap0 = lam*p0+SSp0  #Ap0 = lam*p0 + S(S(p0))
            
            alpha = np.dot(r0, m_mat@r0)/np.dot(p0, m_mat@Ap0)

            fnew = f0 + alpha*p0  #update f
            rnew = r0 - alpha*Ap0  #update r

            #beta = np.dot(rnew, m_mat@rnew)/np.dot(r0, m_mat@r0)
            beta = np.dot(rnew, m_mat@Ap0)/np.dot(p0, m_mat@Ap0)
            pnew = rnew - beta*p0
            k += 1
            
            p0 = pnew; f0 = fnew; r0 = rnew
            error = np.sqrt(np.dot(p0,m_mat@p0))
            f0s.append(fnew)
            if k%50 == 0:
                print('k,error = ',(k,error))
    elif ite_num > 1:
        while k < ite_num:
            obj.sou = m_mat@p0
            u1 = obj.solve_fem()
            obj.sou = m_mat@u1
 
            p1 = obj.solve_fem(star = star)
            r1 = lam*p0+p1

            alpha = np.dot(r0, m_mat@r0)/np.dot(r1, m_mat@p0)

            fnew = f0 + alpha*p0 #update f
            rnew = r0 - alpha*r1 #update r

            beta = np.dot(rnew, m_mat@rnew)/np.dot(r0, m_mat@r0)

            pnew = rnew + beta*p0
            k += 1

            p0 = pnew; f0 = fnew; r0 = rnew
            error = np.sqrt(np.dot(p0,m_mat@p0))
            f0s.append(fnew)
            if k%50 == 0:
                print('k,error = ',(k,error))
        
    tm2 = time.time()
    if prin:
        print('\nCG-stdFEM used {} sec'.format(tm2-tm0))
        print('used {} iterations'.format(k))
    if ite:
        if anim:
            return np.array(f0s), k, tm2-tm0
        else:
            return f0, k, tm2-tm0
    else:
        if anim:
            return np.array(f0s)
        else:
            return f0
    
def CG_ROM_solver(obj, f_0, lam, tol2 = 1e-10, 
                  ite = False, ite_num = 0, anim = False, prin = True, star = False):
    '''Use CG to solve for the source term, 
       incorporated NGSolve for new ROM'''
    #algorithm provided by Y. Zhang
    #assert(isinstance(obj,inverse_obj))
    assert(isinstance(f_0,fem.CoefficientFunction) or isinstance(f_0,int)
           or isinstance(f_0,float) or isinstance(f_0,np.ndarray))
    '''initializes u0, r0, p0, dt = h'''
    #u0 = u_k, u1 = u_k', p0 = p_k, p1 = p_k', r0 = r_k, r1 = r_k'
    f0s = []
    
    if hasattr(obj,'fem_uT'):
        uT = obj.fem_uT
    elif hasattr(obj,'rom_uT'):
        uT = obj.rom_uT
    else:
        print('No uT, error')
        return
    #get mass matrix
    m_mat = sparse_matrix(obj.m)
    
    tm0 = time.time()
    #Sf0
    if not isinstance(f_0,np.ndarray):
        obj.sou = f_0
        u_0 = obj.solve_fem()#this is stable
        Sor = GridFunction(obj.fes)
        Sor.Set(f_0)
        #Draw(Sor,obj.mesh,'?',order=obj.ord)
        f0 = np.array(Sor.vec.data)#this is stable
    else:
        obj.sou = m_mat@f_0
        u_0 = obj.solve_fem()
        f0 = f_0
    f0s.append(f0)
    '''initializes u0, r0, p0, dt = h'''
    #u0 = u_k, u1 = u_k', p0 = p_k, p1 = p_k', r0 = r_k, r1 = r_k'
    if ite_num == 1:
        return f0
    #S(uT-u0)
    obj.sou = m_mat@(uT-u_0)
    tmp = obj.solve_rom(star = star)

    r0 = tmp - lam*f0 
    p0 = r0
    k = 1; error = 1

    error = np.sqrt(np.dot(p0,m_mat@p0))
    if ite_num == 0:
        while error > tol2:
            #Sp_k
            obj.sou = m_mat@p0
            u1 = obj.solve_rom()
            #u_k' = Sp_k
            #Su_k
            obj.sou = m_mat@u1
            p1 = obj.solve_rom(star = star)
            #p_k' = Su_k
            r1 = lam*p0+p1 #r_k' = lam*p_k+p_k'

            alpha = np.dot(r0, m_mat@r0)/np.dot(r1, m_mat@p0)

            fnew = f0 + alpha*p0 #update f
            rnew = r0 - alpha*r1 #update r

            beta = np.dot(rnew, m_mat@rnew)/np.dot(r0, m_mat@r0)

            pnew = rnew + beta*p0
            k += 1

            error = np.sqrt(np.dot(p0,m_mat@p0))

            p0 = pnew; f0 = fnew; r0 = rnew
            f0s.append(f0)
            if k%50 == 0:
                print('k,error = ',(k,error))
    elif ite_num > 1:
        while k < ite_num:
            obj.sou = m_mat@p0
            u1 = obj.solve_rom()
            obj.sou = m_mat@u1
            p1 = obj.solve_rom(star = star)
            r1 = lam*p0+p1 #r_k' = lam*p_k+p_k'

            alpha = np.dot(r0, m_mat@r0)/np.dot(r1, m_mat@p0)

            fnew = f0 + alpha*p0 #update f
            rnew = r0 - alpha*r1 #update r

            beta = np.dot(rnew, m_mat@rnew)/np.dot(r0, m_mat@r0)

            pnew = rnew + beta*p0
            k += 1

            error = np.sqrt(np.dot(p0,m_mat@p0))

            p0 = pnew; f0 = fnew; r0 = rnew
            f0s.append(f0)
            if k%50 == 0:
                print('k,error = ',(k,error))
    
    tm2 = time.time()
    if prin:
        print('\nCG-ROM used {} sec'.format(tm2-tm0))
        print('used {} iterations'.format(k))
    if ite:
        if anim:
            return np.array(f0s), k, tm2-tm0
        else:
            return f0, k, tm2-tm0
    else:
        if anim:
            return np.array(f0s)
        else:
            return f0

'''##########'''

def Solve_lam(obj, f_0, tol1 = 1e-6, tol2 = 1e-10, solver = 'fem', 
              out = False, prin = True):
    '''find the optimal lam(regularization param)'''
    #algorithm from Z. Wang (A data-driven model reduction method...)
    n = int(1/obj.h)**2; m_mat = sparse_matrix(obj.m); count = 1; d = 2
    lam_0 = n**(-2/3) #initialization has many ways?
    tm0 = time.time()
    
    if solver == 'fem':
        f_1 = CG_FEM_solver(obj, f_0, lam_0, tol2, 
                  ite = False, ite_num = 0, anim = False)
    elif solver == 'rom':
        f_1 = CG_ROM_solver(obj, f_0, lam_0, tol2, 
                  ite = False, ite_num = 0, anim = False)
    if not isinstance(f_0,np.ndarray):
        obj.sou = f_0
        u_0 = obj.solve_fem()
        Sor = GridFunction(obj.fes)
        Sor.Set(f_0)
        f_0 = np.array(Sor.vec.data)
    norm_0 = inv_norm(obj, f_0); norm_1 = inv_norm(obj, f_1)
    
    obj.sou = m_mat@f_1
    if solver == 'fem':
        Sf_1 = obj.solve_fem()
    elif solver == 'rom':
        Sf_1 = obj.solve_rom()
    resi = inv_error(obj, Sf_1, obj.fem_uT)
    #update lambda
    lam_1 = ((1/np.sqrt(n)) * resi * (1/norm_1))**(4/3)
    
    while np.abs(norm_1-norm_0) >= tol1:
        count += 1; norm_0 = norm_1
        
        if solver == 'fem':
            f_1 = CG_FEM_solver(obj, f_0, lam_1, tol2, 
                  ite = False, ite_num = 0, anim = False, prin = False)
        elif solver == 'rom':
            f_1 = CG_ROM_solver(obj, f_0, lam_1, tol2, 
                      ite = False, ite_num = 0, anim = False, prin = False)
        norm_1 = inv_norm(obj, f_1)
        obj.sou = m_mat@f_1
        if solver == 'fem':
            Sf_1 = obj.solve_fem()
        elif solver == 'rom':
            Sf_1 = obj.solve_rom()
        resi = inv_error(obj, Sf_1, obj.fem_uT)
        n_resi = inv_n_norm(obj,Sf_1-obj.fem_uT)
        #update lambda
        lam_1 = (1/np.sqrt(n)) * resi * (1/norm_1)
        
        if count%5==0: print(f'lam_ite: {count}, diff: {np.abs(norm_1-norm_0)}')
    tm2 = time.time()
    if prin:
        print('\nFind optimal_lam used {} sec'.format(tm2-tm0))
        print('used {} iterations'.format(count))
    if out:
        return lam_1, count, f_1, [resi,n_resi]
    else:
        return f_1
        
'''##########'''

def CG_difference(obj_fem, obj_rom, f_0, l = 10, lam = 1e-6, withM = True, tol2 = 1e-10):
    '''Computes the difference between stdFEM and ROM solver'''
    #algorithm provided by Y. Zhang
    #assert(isinstance(obj_fem,inverse_obj))
    #assert(isinstance(obj_rom,inverse_obj))
    assert(isinstance(f_0,fem.CoefficientFunction) 
           or isinstance(f_0,int) or isinstance(f_0,float))
    
    '''initializes u0, r0, p0, dt = h'''
    #u0 = u_k, u1 = u_k', p0 = p_k, p1 = p_k', r0 = r_k, r1 = r_k'
    assert(hasattr(obj_fem,'fem_uT'))
    uT_fem = obj_fem.fem_uT
    assert(hasattr(obj_rom,'rom_uT'))
    uT_rom = obj_rom.rom_uT
    uT_diff = np.sqrt(np.dot(uT_fem-uT_rom,uT_fem-uT_rom))
    print('uT diff = ',uT_diff)
    #get mass matrix
    m_mat = sparse_matrix(obj_fem.m)
    
    #Sf0
    obj_fem.sou = f_0; obj_rom.sou = f_0
    u0_fem = obj_fem.solve_fem(); u0_rom = obj_rom.solve_rom()
    print('u0 fem = ',np.sqrt(np.dot(u0_fem,u0_fem)))
    u0_diff = np.sqrt(np.dot(u0_fem-u0_rom,u0_fem-u0_rom))
    print('u0 diff = ',u0_diff)
    
    Sor = GridFunction(obj_fem.fes)
    Sor.Set(f_0)
    #Draw(Sor,obj.mesh,'?',order=obj.ord)
    f0_fem = np.array(Sor.vec.data)#this is stable
    f0_rom = np.array(Sor.vec.data)
    f0_diff = np.sqrt(np.dot(f0_fem-f0_rom,f0_fem-f0_rom))
    print('f0 diff = ',f0_diff)
    '''initializes u0, r0, p0, dt = h'''
    #u0 = u_k, u1 = u_k', p0 = p_k, p1 = p_k', r0 = r_k, r1 = r_k'
    
    tm0 = time.time()
    #S(uT-u0)
    obj_fem.sou = m_mat@(uT_fem-u0_fem)
    obj_rom.sou = m_mat@(uT_rom-u0_rom)
    tmp_fem = obj_fem.solve_fem(); tmp_rom = obj_rom.solve_rom()
   
    r0_fem = tmp_fem - lam*f0_fem; r0_rom = tmp_rom - lam*f0_rom
    p0_fem = r0_fem; p0_rom = r0_rom
    print('p0 fem = ',np.sqrt(np.dot(p0_fem,p0_fem)))
    p0_diff = np.sqrt(np.dot(p0_fem-p0_rom,p0_fem-p0_rom))
    print('p0 = r0 diff = ',p0_diff)
    
    k = 0; error = 1
    if withM:
            error_fem = np.sqrt(np.dot(p0_fem,m_mat@p0_fem))
            error_rom = np.sqrt(np.dot(p0_rom,m_mat@p0_rom))
    else:
            error_fem = np.sqrt(np.dot(p0_fem,p0_fem))
            error_rom = np.sqrt(np.dot(p0_rom,p0_rom))
    error = min(error_fem,error_rom)
    print('\n Start iterating: \n')
    while (error > tol2) and (k <= l):
        #Sp_k
        obj_fem.sou = m_mat@p0_fem
        obj_rom.sou = m_mat@p0_rom
        u1_fem = obj_fem.solve_fem()
        print('u1 fem = ',np.sqrt(np.dot(u1_fem,u1_fem)))
        u1_rom = obj_rom.solve_rom()
        u1_diff = np.sqrt(np.dot(u1_fem-u1_rom,u1_fem-u1_rom))
        print('u1 diff = ',u1_diff)
        #u_k' = Sp_k
        
        #Su_k
        obj_fem.sou = m_mat@u1_fem
        obj_rom.sou = m_mat@u1_rom
        p1_fem = obj_fem.solve_fem()
        print('p1 fem = ',np.sqrt(np.dot(p1_fem,p1_fem)))
        p1_rom = obj_rom.solve_rom()
        p1_diff = np.sqrt(np.dot(p1_fem-p1_rom,p1_fem-p1_rom))
        print('p1 diff = ',p1_diff)
        #p_k' = Su_k
        
        r1_fem = lam*p0_fem+p1_fem #r_k' = lam*p_k+p_k'
        r1_rom = lam*p0_rom+p1_rom
        r1_diff = np.sqrt(np.dot(r1_fem-r1_rom,r1_fem-r1_rom))
        print('r1 fem = ',np.sqrt(np.dot(r1_fem,r1_fem)))
        print('r1 diff = ',r1_diff)
        
        p0_diff = np.sqrt(np.dot(p0_fem-p0_rom,p0_fem-p0_rom))
        print('\np0 diff = ',p0_diff)
        
        r0_diff = np.sqrt(np.dot(r0_fem-r0_rom,r0_fem-r0_rom))
        print('r0 diff = ',r0_diff)
        #m_mat@r0 difference is trivial
        a = np.dot(r0_fem, m_mat@r0_fem)
        b = np.dot(r0_rom, m_mat@r0_rom)
        c = np.dot(r1_fem, m_mat@p0_fem)
        d = np.dot(r1_rom, m_mat@p0_rom)
        #print(c,d)
        print('a-b',np.abs(a-b))
        print('c-d',np.abs(c-d))
        print('denom:',c,d)
        if withM:
            alpha_fem = a/c
            alpha_rom = b/d
        else:
            alpha_fem = np.dot(r0_fem, r0_fem)/np.dot(r1_fem, p0_fem)
            alpha_rom = np.dot(r0_rom, r0_rom)/np.dot(r1_rom, p0_rom)
        
        alpha_diff = np.abs(a/c-b/d)
        print(alpha_fem,alpha_rom)
        print('\nalpha diff = ',alpha_diff)
        
        fnew_fem = f0_fem + alpha_fem*p0_fem #update f
        fnew_rom = f0_rom + alpha_rom*p0_rom
        rnew_fem = r0_fem - alpha_fem*r1_fem #update r
        rnew_rom = r0_rom - alpha_rom*r1_rom
        if withM:
            beta_fem = np.dot(rnew_fem, m_mat@rnew_fem)/np.dot(r0_fem, m_mat@r0_fem)
            beta_rom = np.dot(rnew_rom, m_mat@rnew_rom)/np.dot(r0_rom, m_mat@r0_rom)
        else:
            beta_fem = np.dot(rnew_fem, rnew_fem)/np.dot(r0_fem, r0_fem)
            beta_rom = np.dot(rnew_rom, rnew_rom)/np.dot(r0_rom, r0_rom)
        beta_diff = np.abs(beta_fem-beta_rom)
        print('beta diff = ',beta_diff)
        
        pnew_fem = rnew_fem + beta_fem*p0_fem
        pnew_rom = rnew_rom + beta_rom*p0_rom
        
        k += 1
        if withM:
            error_fem = np.sqrt(np.dot(p0_fem,m_mat@p0_fem))
            error_rom = np.sqrt(np.dot(p0_rom,m_mat@p0_rom))
        else:
            error_fem = np.sqrt(np.dot(p0_fem,p0_fem))
            error_rom = np.sqrt(np.dot(p0_rom,p0_rom))
        error = min(error_fem,error_rom)
        p0_fem = pnew_fem; f0_fem = fnew_fem; r0_fem = rnew_fem
        print('p0 fem = ',np.sqrt(np.dot(p0_fem,p0_fem)))
        p0_rom = pnew_rom; f0_rom = fnew_rom; r0_rom = rnew_rom
        f0_diff = np.sqrt(np.dot(f0_fem-f0_rom,f0_fem-f0_rom))
        print('\n f0 diff = ',f0_diff)
        if k%50 == 0:
            print('fem error,',k,error_fem)
            print('rom error,',k,error_rom)
        
    print('iteration num = ',k)
    tm2 = time.time()
    print()
    print('CG-diff used {} sec'.format(tm2-tm0))
    #print('used {} iterations'.format(k))
    return 0
