import numpy as np
import scipy.linalg
from multiprocess import Process, Queue

# Simulation of SGD-DMFT and SME-DMFT restricting to
#   (1) single-index teacher and student, k = k^* = 1
#   (2) fixed learning rate and batch size schedules eta, kappa

# Conventions:
#   Ctheta_sqrt: size (N+1) x m, such that (Ctheta_sqrt)(Ctheta_sqrt)' = Ctheta
#      - first N rows correspond to w^0,w^1,...
#      - last row corresponds to w^*
#   Cf_sqrt: size N x m, such that (Cf_sqrt)(Cf_sqrt)' = Cf
#   Rtheta: lower-triangular N x N matrix (diagonal all 1)
#   Rf: strictly lower-triangular N x N matrix (diagonal all 0)
#
#   args: DMFT parameters
#      - delta: discretization size
#      - N = T/delta: number of time steps
#      - kappa, eta: batch size, learning rate
#      - gamma: limit of n/d
#      For computing Ctheta, Rtheta analytically under ridge penalty:
#      - theta_cov: 2 x 2 matrix for Cov(theta^0,theta^*)
#      - lamb: parameter for ridge penalty (lamb/2)theta^2
#      For computing Ctheta, Rtheta via simulation:
#      - rand_theta: function that generates a random pair (theta^0,theta^*)
#      - g, Dg: functions that compute g(theta) and Dg(theta)
#      For computing Cf, Rf analytically under linear model and squared loss:
#      - eps_var: Var(eps)
#      For computing Cf, Rf via simulation:
#      - rand_eps: function that generates a random eps noise variable
#      - f, Dxif, Dwstarf: functions that compute f(xi,wstar,eps) and its derivatives

class Args(object):
  __slots__ = ('delta','N','kappa','eta','gamma','theta_cov','lamb','rand_theta','g','Dg','eps_var','rand_eps','f','Dxif','Dwstarf')
  def __init__(self,delta,N,kappa,eta,gamma,
          theta_cov=None,lamb=None,rand_theta=None,g=None,Dg=None,
          eps_var=None,rand_eps=None,f=None,Dxif=None,Dwstarf=None):
    self.delta = delta
    self.N = N
    self.kappa = kappa
    self.eta = eta
    self.gamma = gamma
    self.theta_cov = theta_cov
    self.lamb = lamb
    self.rand_theta = rand_theta
    self.g = g
    self.Dg = Dg
    self.eps_var = eps_var
    self.rand_eps = rand_eps
    self.f = f
    self.Dxif = Dxif
    self.Dwstarf = Dwstarf

# Simulate a realization of xi, rfstar, rf under SGD or SME
def simulate_xi(Ctheta_sqrt,Rtheta,args,mode='SGD',seed=None):
  if args.rand_eps is None or args.f is None or args.Dxif is None or \
        args.Dwstarf is None:
    raise RuntimeError('Need to specify rand_eps, f, Dxif, Dwstarf')
  if seed is not None: np.random.seed(seed)
  lr = args.eta/args.kappa
  # Generate eps, dz, and Gaussian process (w,wstar)
  eps = args.rand_eps()
  if mode == 'SGD':
    dz = np.random.poisson(lam=args.delta*args.kappa,size=args.N)
    # SGD shortcuts to sparse computations of int_0^t dz
    dz_inds = np.nonzero(dz)[0]  # Non-zero entries of dz
    dz_vals = dz[dz_inds]        # Values of dz at non-zero entries (usually 1)
  elif mode == 'SME':
    dz = np.random.normal(loc=args.kappa*args.delta,
            scale=np.sqrt(args.kappa*args.delta),size=args.N)
  w = Ctheta_sqrt.dot(np.random.normal(size=Ctheta_sqrt.shape[1]))
  wstar = w[-1]
  w = w[:-1]
  # Compute xi, D_xi f(xi), rfstar
  xi = np.zeros(args.N)
  Dxif = np.zeros(args.N)
  rfstar = np.zeros(args.N)
  if mode == 'SGD':
    fdz = np.zeros(len(dz_inds))      # holds non-zero values of f(xi)dz
    rfstardz = np.zeros(len(dz_inds)) # holds non-zero values of (r_f^*)dz
    idx = 0                           # index into dz_inds
  elif mode == 'SME':
    fdz = np.zeros(args.N)      # holds values of f(xi)dz
    rfstardz = np.zeros(args.N) # holds values of (r_f^*)dz
  for t in range(args.N):
    xi[t] = w[t]
    if mode == 'SGD':
      xi[t] -= lr*Rtheta[t,dz_inds].dot(fdz)
      # Update fdz for time t after above update of xi[t]
      if idx < len(dz_inds) and t == dz_inds[idx]:
        fdz[idx] = args.f(xi[t],wstar,eps)*dz_vals[idx]
    else:
      xi[t] -= lr*Rtheta[t,:t].dot(fdz[:t])
      fdz[t] = args.f(xi[t],wstar,eps) * dz[t]
    Dxif[t] = args.Dxif(xi[t],wstar,eps)
    rfstar[t] = args.Dwstarf(xi[t],wstar,eps)
    if mode == 'SGD':
      rfstar[t] -= Dxif[t]*lr*Rtheta[t,dz_inds].dot(rfstardz)
      # Update rfstardz for time t after above update of rfstar[t]
      if idx < len(dz_inds) and t == dz_inds[idx]:
        rfstardz[idx] = rfstar[t]*dz_vals[idx]
        idx += 1
    else:
      rfstar[t] -= Dxif[t]*lr*Rtheta[t,:t].dot(rfstardz[:t])
      rfstardz[t] = rfstar[t]*dz[t]
  # Compute int_0^t eta/kappa*f(xi,w^*,eps)dz for covariance Cf
  if mode == 'SGD':
    intfdz = np.zeros(args.N)
    intfdz[dz_inds] = args.eta/args.kappa*fdz
  else:
    intfdz = args.eta/args.kappa*fdz
  intfdz = np.concatenate((np.zeros(1),np.cumsum(intfdz)[:-1]))
  # Compute rf
  rf = np.zeros((args.N,args.N))
  if mode == 'SGD':
     # rf[t,s] is only non-zero when dz[s] is non-zero
    for j,s in enumerate(dz_inds):
      rfdz = np.zeros(len(dz_inds))   # holds non-zero values of (r_f[:,s])dz
      idx = j+1
      for t in range(s+1,args.N):
        rf[t,s] = -Dxif[t]*lr*Rtheta[t,s]*Dxif[s]*dz_vals[j]/args.delta
        rf[t,s] -= Dxif[t]*lr*Rtheta[t,dz_inds].dot(rfdz)
        # Update rfdz for time (t,s) after above update of rf[t,s]
        if idx < len(dz_inds) and t == dz_inds[idx]:
          rfdz[idx] = rf[t,s] * dz_vals[idx]
          idx += 1
  else:
    Dxifdz = Dxif*dz                 # holds values of (D_xi f)dz
    rfdz = np.zeros((args.N,args.N)) # holds values of (r_f)dz
    for t in range(1,args.N):
      rf[t,:t] = -Dxif[t]*lr*Rtheta[t,:t]*Dxifdz[:t]/args.delta
      rf[t,:t] -= Dxif[t]*lr*Rtheta[t,:t].dot(rfdz[:t,:t])
      rfdz[t,:t] = rf[t,:t]*dz[t]
  return(xi,intfdz,Dxif,rfstar,rf,wstar,eps)

# Simulate multiple realizations of xi using multiprocess
def simulate_xi_multi(Ctheta_sqrt,Rtheta,args,nsim,mode='SGD',nprocesses=1,verbose=False):
  def worker(start,end,xi_list,wstar_list,eps_list):
    xis = np.zeros((args.N,end-start))
    wstars = np.zeros(end-start)
    epss = np.zeros(end-start)
    for b in range(start,end):
      if verbose and b % 100 == 0: print('  xi simulation %d' % b)
      (xi,intfdz,Dxif,rfstar,rf,wstar,eps) = simulate_xi(Ctheta_sqrt,
              Rtheta,args,mode=mode,seed=b)
      xis[:,b-start] = xi
      wstars[b-start] = wstar
      epss[b-start] = eps
    xi_list.put((xis,start,end))
    wstar_list.put((wstars,start,end))
    eps_list.put((epss,start,end))
  processes = []
  xi_list = Queue()
  wstar_list = Queue()
  eps_list = Queue()
  for i in range(nprocesses):
    start = i * (nsim // nprocesses)
    end = (i+1) * (nsim // nprocesses) if i < nprocesses - 1 else nsim
    process = Process(target=worker,
            args=(start,end,xi_list,wstar_list,eps_list))
    processes.append(process)
    process.start()
  xi = np.zeros((args.N,nsim))
  wstar = np.zeros(nsim)
  eps = np.zeros(nsim)
  for i in range(nprocesses):
    xi_block = xi_list.get()
    start = xi_block[1]
    end = xi_block[2]
    xi[:,start:end] = xi_block[0]
    wstar_block = wstar_list.get()
    start = wstar_block[1]
    end = wstar_block[2]
    wstar[start:end] = wstar_block[0]
    eps_block = eps_list.get()
    start = eps_block[1]
    end = eps_block[2]
    eps[start:end] = eps_block[0]
  for process in processes:
    process.join()
  return(xi,wstar,eps)

# Compute Cf, Gamma, Rfstar, Rf via simulation of xi
def compute_CRf_sim(Ctheta_sqrt,Rtheta,args,nsim,mode='SGD',
      nprocesses=1,verbose=False):
  def worker(start,end,Cf_sqrt_list,Gamma_list,Rfstar_list,Rf_list):
    Cf_sqrt = np.zeros((args.N,end-start))
    Gamma = np.zeros(args.N)
    Rfstar = np.zeros(args.N)
    Rf = np.zeros((args.N,args.N))
    for b in range(start,end):
      if verbose and b % 100 == 0: print('  xi simulation %d' % b)
      (xi,intfdz,Dxif,rfstar,rf,wstar,eps) = simulate_xi(Ctheta_sqrt,
              Rtheta,args,mode=mode,seed=b)
      Cf_sqrt[:,b-start] = intfdz/np.sqrt(nsim)
      Gamma += Dxif/nsim
      Rfstar += rfstar/nsim
      Rf += rf/nsim
    Cf_sqrt_list.put((Cf_sqrt,start,end))
    Gamma_list.put(Gamma)
    Rfstar_list.put(Rfstar)
    Rf_list.put(Rf)
  processes = []
  Cf_sqrt_list = Queue()
  Gamma_list = Queue()
  Rfstar_list = Queue()
  Rf_list = Queue()
  for i in range(nprocesses):
    start = i * (nsim // nprocesses)
    end = (i+1) * (nsim // nprocesses) if i < nprocesses - 1 else nsim
    process = Process(target=worker,
            args=(start,end,Cf_sqrt_list,Gamma_list,Rfstar_list,Rf_list))
    processes.append(process)
    process.start()
  Cf_sqrt = np.zeros((args.N,nsim))
  Gamma = np.zeros(args.N)
  Rfstar = np.zeros(args.N)
  Rf = np.zeros((args.N,args.N))
  for i in range(nprocesses):
    Cf_sqrt_block = Cf_sqrt_list.get()
    start = Cf_sqrt_block[1]
    end = Cf_sqrt_block[2]
    Cf_sqrt[:,start:end] = Cf_sqrt_block[0]
    Gamma += Gamma_list.get()
    Rfstar += Rfstar_list.get()
    Rf += Rf_list.get()
  for process in processes:
    process.join()
  return(Cf_sqrt,Gamma,Rfstar,Rf)

# Compute Cf, Gamma, Rfstar, Rf analytically under linear model and square loss
def compute_CRf_quadratic(Ctheta_sqrt,Rtheta,args):
  if args.eps_var is None:
    raise RuntimeError('Need to specify eps_var')
  # (Cb_sqrt)(Cb_sqrt)' = Cb
  Cb_sqrt = np.hstack((Ctheta_sqrt[:args.N,:]-Ctheta_sqrt[-1,:],
              np.sqrt(args.eps_var)*np.ones((args.N,1))))
  A = -args.eta*Rtheta  # Operator A_xi, without delta scaling
  tmp = scipy.linalg.solve_triangular(np.identity(args.N)-args.delta*A,
          np.hstack((np.ones((args.N,1)),A,Cb_sqrt)),lower=True)
  Me = tmp[:,0]  # (I-A)^{-1} * e (where e = all-1's)
  MA = tmp[:,1:(args.N+1)]  # (I-A)^{-1} * A
  # (MCb_sqrt)(MCb_sqrt)' = (I-A)^{-1}C_b(I-A)^{-1}'
  MCb_sqrt = tmp[:,(args.N+1):]
  # Compute first component of C_f: (CfA_sqrt)(CfA_sqrt)' = BMC_bM'B'
  CfA_sqrt = args.eta*args.delta*np.vstack((np.zeros((1,Cb_sqrt.shape[1])),
              np.cumsum(MCb_sqrt,axis=0)[:(args.N-1),:]))
  # Compute second component of C_f:
  # (CfB_sqrt)(CfB_sqrt)' = (B+BMA)^{odot 2}(Id-(MA)^{odot 2})^{-1}(MC_bM')
  diag = np.sum(MCb_sqrt**2,axis=1)  # diagonal entries of MC_bM'
  tmp = scipy.linalg.solve_triangular(np.identity(args.N)
          -args.delta/args.kappa*MA**2,diag,lower=True)
  # Operator B+BMA, without delta scaling
  BplusBMA = args.eta*np.vstack((np.zeros((1,args.N)),np.cumsum(
                  np.identity(args.N)+args.delta*MA,axis=0)[:(args.N-1),:]))
  CfB_sqrt = BplusBMA*np.sqrt(tmp*args.delta/args.kappa)
  # (Cf_sqrt)(Cf_sqrt)' = C_f
  Cf_sqrt = np.hstack((CfA_sqrt,CfB_sqrt))
  Rfstar = -Me
  Rf = MA
  Gamma = np.ones(args.N)
  return(Cf_sqrt,Gamma,Rfstar,Rf)

# Simulate a realization of theta, rtheta
def simulate_theta(Cf_sqrt,Gamma,Rfstar,Rf,args,seed=None):
  if args.rand_theta is None or args.g is None or args.Dg is None:
    raise RuntimeError('Need to specify rand_theta, g, Dg')
  if seed is not None: np.random.seed(seed)
  lr = args.gamma*args.eta
  # Generate theta_0, theta_star, Gaussian process u
  theta_init,theta_star = args.rand_theta()
  u = Cf_sqrt.dot(np.random.normal(size=Cf_sqrt.shape[1]))
  # Compute theta
  theta = np.zeros(args.N)
  theta[0] = theta_init + np.sqrt(args.gamma)*u[0]
  for t in range(args.N-1):
    theta[t+1] = theta[t]+np.sqrt(args.gamma)*(u[t+1]-u[t])- \
                   (lr*(Gamma[t]*theta[t]+Rf[t,:t].dot(theta[:t])*args.delta \
                   +Rfstar[t]*theta_star)+args.eta*args.g(theta[t]))*args.delta
  # Compute rtheta
  rtheta = np.zeros((args.N,args.N))
  rtheta[0,0] = 1
  for t in range(args.N-1):
    rtheta[t+1,t+1] = 1
    rtheta[t+1,:(t+1)] = rtheta[t,:(t+1)] * \
          (1-(lr*Gamma[t]+args.eta*args.Dg(theta[t]))*args.delta) \
          -lr*Rf[t,:t].dot(rtheta[:t,:(t+1)])*args.delta**2
  return(theta,rtheta,theta_star)

# Simulate multiple realizations of theta using multiprocess
def simulate_theta_multi(Cf_sqrt,Gamma,Rfstar,Rf,args,nsim,nprocesses=1,verbose=False):
  def worker(start,end,theta_list,theta_star_list):
    thetas = np.zeros((args.N,end-start))
    theta_stars = np.zeros(end-start)
    for b in range(start,end):
      if verbose and b % 100 == 0: print('  theta simulation %d' % b)
      (theta,rtheta,theta_star) = simulate_theta(Cf_sqrt,Gamma,Rfstar,Rf,args,seed=b)
      thetas[:,b-start] = theta
      theta_stars[b-start] = theta_star
    theta_list.put((thetas,start,end))
    theta_star_list.put((theta_stars,start,end))
  processes = []
  theta_list = Queue()
  theta_star_list = Queue()
  for i in range(nprocesses):
    start = i * (nsim // nprocesses)
    end = (i+1) * (nsim // nprocesses) if i < nprocesses - 1 else nsim
    process = Process(target=worker,
            args=(start,end,theta_list,theta_star_list))
    processes.append(process)
    process.start()
  theta = np.zeros((args.N,nsim))
  theta_star = np.zeros(nsim)
  for i in range(nprocesses):
    theta_block = theta_list.get()
    start = theta_block[1]
    end = theta_block[2]
    theta[:,start:end] = theta_block[0]
    theta_star_block = theta_star_list.get()
    start = theta_star_block[1]
    end = theta_star_block[2]
    theta_star[start:end] = theta_star_block[0]
  for process in processes:
    process.join()
  return(theta,theta_star)

# Compute Ctheta, Rtheta via simulation of theta
def compute_CRtheta_sim(Cf_sqrt,Gamma,Rfstar,Rf,args,nsim,
      nprocesses=1,verbose=False):
  def worker(start,end,Ctheta_sqrt_list,Rtheta_list):
    Ctheta_sqrt = np.zeros((args.N+1,end-start))
    Rtheta = np.zeros((args.N,args.N))
    for b in range(start,end):
      if verbose and b % 100 == 0: print('  theta simulation %d' % b)
      (theta,rtheta,theta_star) = simulate_theta(Cf_sqrt,Gamma,Rfstar,Rf,args,seed=b)
      Ctheta_sqrt[:,b-start] = np.concatenate((theta,[theta_star]))/np.sqrt(nsim)
      Rtheta += rtheta/nsim
    Ctheta_sqrt_list.put((Ctheta_sqrt,start,end))
    Rtheta_list.put(Rtheta)
  processes = []
  Ctheta_sqrt_list = Queue()
  Rtheta_list = Queue()
  for i in range(nprocesses):
    start = i * (nsim // nprocesses)
    end = (i+1) * (nsim // nprocesses) if i < nprocesses - 1 else nsim
    process = Process(target=worker,
            args=(start,end,Ctheta_sqrt_list,Rtheta_list))
    processes.append(process)
    process.start()
  Ctheta_sqrt = np.zeros((args.N+1,nsim))
  Rtheta = np.zeros((args.N,args.N))
  for i in range(nprocesses):
    Ctheta_sqrt_block = Ctheta_sqrt_list.get()
    start = Ctheta_sqrt_block[1]
    end = Ctheta_sqrt_block[2]
    Ctheta_sqrt[:,start:end] = Ctheta_sqrt_block[0]
    Rtheta += Rtheta_list.get()
  for process in processes:
    process.join()
  return(Ctheta_sqrt,Rtheta)

# Compute Ctheta, Rtheta analytically under ridge penalty
def compute_CRtheta_quadratic(Cf_sqrt,Gamma,Rfstar,Rf,args):
  if args.theta_cov is None or args.lamb is None:
    raise RuntimeError('Need to specify theta_cov, lamb')
  # Operator A_theta, without delta scaling
  A = -args.eta*args.lamb*np.ones((args.N,args.N))
  A -= args.eta*args.gamma*Gamma # Subtract Gamma from each row
  A = np.tril(A,k=-1)
  A -= args.gamma*args.eta*args.delta*np.cumsum(Rf,axis=0)
  S = np.ones((args.N,2))
  S[:,1] = -args.eta*args.gamma*args.delta*np.cumsum(Rfstar)
  thetacov_sqrt = np.linalg.cholesky(args.theta_cov)
  # Covariance of b_theta from (theta^0,theta^*)
  Cb_sqrt = S.dot(thetacov_sqrt)
  # Covariance of b_theta from u^t
  Cb_sqrt = np.hstack((Cb_sqrt,np.sqrt(args.gamma)*Cf_sqrt))
  E = np.tril(np.ones((args.N,args.N)))
  tmp = scipy.linalg.solve_triangular(np.identity(args.N)-args.delta*A,
          np.hstack((E,Cb_sqrt)),lower=True)
  Rtheta = tmp[:,:args.N]
  # (Ctheta_sqrt)(Ctheta_sqrt)' = Ctheta without variable theta^*
  Ctheta_sqrt = tmp[:,args.N:]
  # Add covariance with theta^* to Ctheta
  lastrow = np.hstack((np.array([[0,1]]).dot(thetacov_sqrt),
              np.zeros((1,Ctheta_sqrt.shape[1]-2))))
  Ctheta_sqrt = np.vstack((Ctheta_sqrt,lastrow))
  return(Ctheta_sqrt,Rtheta)

# Initialize Ctheta, Rtheta
def init_CRtheta(args,Ctheta_sqrt_init=None,Rtheta_init=None):
  decay = np.exp(np.linspace(0,-3,args.N))
  if Ctheta_sqrt_init is None:
    if args.theta_cov is not None:
      theta_init_var = args.theta_cov[0,0]
      theta_star_var = args.theta_cov[1,1]
    else:
      theta = np.array([args.rand_theta() for b in range(1000)])
      theta_init_var = np.var(theta[:,0])
      theta_star_var = np.var(theta[:,1])
    # Ctheta[t,s] = e^(-c|t-s|)*Var[theta^0], Ctheta[t,*] = 0
    Ctheta_sqrt = np.zeros((args.N+1,args.N+1))
    Ctheta = theta_init_var*scipy.linalg.toeplitz(c=decay,r=decay)
    Ctheta_sqrt[:args.N,:args.N] = np.linalg.cholesky(Ctheta)
    Ctheta_sqrt[args.N,args.N] = theta_star_var
  else:
    Ctheta_sqrt = Ctheta_sqrt_init
  if Rtheta_init is None:
    # Rtheta[t,s] = e^{-c(t-s)_+}
    Rtheta = scipy.linalg.toeplitz(c=decay,r=np.zeros(args.N))
  else:
    Rtheta = Rtheta_init
  return(Ctheta_sqrt,Rtheta)

# Main DMFT loop
#   sim_theta: set to False to compute Ctheta, Rtheta analytically
#   sim_xi: set to False to compute Cf, Rf, analytically
def DMFT_fixed_point(args,mode='SGD',sim_theta=True,sim_xi=True,nsim_theta=2000,
        nsim_xi=10000,Ctheta_sqrt_init=None,Rtheta_init=None,
        maxiter=20,tol=1e-2,nprocesses=1,verbose=True):
  for it in range(maxiter):
    if verbose:
      print('Starting iteration %d' % it)
    if it == 0:
      ### Initialize Ctheta, Rtheta
      (Ctheta_sqrt,Rtheta) = init_CRtheta(args,Ctheta_sqrt_init,Rtheta_init)
    else:
      ### Compute Ctheta, Rtheta from Cf, Gamma, Rfstar, Rf
      if sim_theta:
        (Ctheta_sqrt,Rtheta) = compute_CRtheta_sim(Cf_sqrt_prev,Gamma_prev,
                Rfstar_prev,Rf_prev,args,nsim_theta,
                nprocesses=nprocesses,verbose=verbose)
      else:
        (Ctheta_sqrt,Rtheta) = compute_CRtheta_quadratic(Cf_sqrt_prev,
                Gamma_prev,Rfstar_prev,Rf_prev,args)
    ### Compute Cf, Gamma, Rfstar, Rf from Ctheta, Rtheta
    if sim_xi:
      (Cf_sqrt,Gamma,Rfstar,Rf) = compute_CRf_sim(Ctheta_sqrt,Rtheta,args,
              nsim_xi,mode=mode,nprocesses=nprocesses,verbose=verbose)
    else:
      (Cf_sqrt,Gamma,Rfstar,Rf) = compute_CRf_quadratic(Ctheta_sqrt,Rtheta,args)
    if verbose: print('Iteration %d complete' % it)
    if it > 0:
      ### Check convergence
      dCtheta = np.max(np.abs(np.sum(Ctheta_sqrt**2,axis=1)[:-1]
                  -np.sum(Ctheta_sqrt_prev**2,axis=1)[:-1]))
      dCthetastar = np.max(np.abs(Ctheta_sqrt[:-1,:].dot(Ctheta_sqrt[-1,:])
                  -Ctheta_sqrt_prev[:-1,:].dot(Ctheta_sqrt_prev[-1,:])))
      dCf = np.max(np.abs(np.sum(Cf_sqrt**2,axis=1)
                  -np.sum(Cf_sqrt_prev**2,axis=1)))
      dRtheta = np.max(np.abs(Rtheta-Rtheta_prev))
      dGamma = np.max(np.abs(Gamma-Gamma_prev))
      dRf = np.max(np.abs(np.cumsum((Rf-Rf_prev)[:,::-1],axis=1)*args.delta))
      dRfstar = np.max(np.abs(Rfstar-Rfstar_prev))
      if verbose:
        print('  max difference of Ctheta[t,t]: %f' % dCtheta)
        print('  max difference of Ctheta[t,*]: %f' % dCthetastar)
        print('  max difference of Cf[t,t]: %f' % dCf)
        print('  max difference of Rtheta[t,s]: %f' % dRtheta)
        print('  max difference of Gamma[t]: %f' % dGamma)
        print('  max difference of int_s^t Rf[t,r] dr: %f' % dRf)
        print('  max difference of Rf[t,*]: %f' % dRfstar)
      if max(dCtheta,dCthetastar,dCf,dRtheta,dGamma,dRf,dRfstar) < tol:
        if verbose: print('CONVERGED')
        break
    Ctheta_sqrt_prev = Ctheta_sqrt
    Cf_sqrt_prev = Cf_sqrt
    Rtheta_prev = Rtheta
    Gamma_prev = Gamma
    Rf_prev = Rf
    Rfstar_prev = Rfstar
  return(Ctheta_sqrt,Rtheta,Cf_sqrt,Gamma,Rf,Rfstar)
