"""

UC Undergraduate Research Capstone Code Implementation (Ryan Kamp; Presented: 4/11/25, Last Updated: 6/4/25)



This file contains my implementation of the following paper:

Notes on Normative Solutions to the Speed-Accuracy Trade-Off in Perceptual Decision-Making (Jan Drugowitsch, 2015).

For more details, please see my presentation of this paper on YouTube: https://youtu.be/H2HssPbUu9k

Links to Jan Drugowitsch:
- Lab: https://www.drugowitschlab.org/
- Google Scholar: https://scholar.google.com/citations?user=fCU×98WAAAAJ
- GitHub: https://github.com/DrugowitschLab
- X: https://x.com/jdrugowitsch

My links (Ryan Kamp):
- LinkedIn: https://www.linkedin.com/in/rjk1999
- Personal Website: https://sites.google.com/view/ryanjosephkamp



Dynamic Programming

Known Evidence Reliability ("KER")



FUNCTIONS:

1. ker_tdtc_dynpro__multi_cost_examples:

    * Time-Dependent Trial Costs ("TDTC") *

    Generate X- and g-data (trajectories)
    for known evidence reliability and time-dependent trial costs,
    where bound functions are computed by dynamic programming
    for five example cost functions:
      c1(t) = e^{-1/(t+1)}
      c2(t) = 1/(1+e^{-t})
      c3(t) = 1/(1+1/(1+t))
      c4(t) = tanh(t)
      c5(t) = (2/π)*arctan(t);

    plot each cost function vs. time,
    each X-bound function vs. time,
    each g-bound function vs. time,
    each full X-trajectory, and
    each full g-trajectory;

    plot each X-trajectory and its X-bound functions together; and

    plot each g-trajectory and its g-bound functions together.

  *** RECOMMENDATION:
      use M_ts >= 1,000 to make max experiment time long enough ***

2. ker_tdtc_dynpro__multi_bib_example__tanh:

    * Time-Dependent Trial Costs ("TDTC") *

    * Biased Initial Beliefs ("BIB") *

    Generate X- and g-data (trajectories) for a specified number of initial
    beliefs within a specified range
    (e.g., 100 initial beliefs between 0.01 and 0.99)
    for known evidence reliability and time-dependent trial costs,
    where bound functions are computed by dynamic programming
    with cost function c(t) = tanh(t);

    plot all X-trajectories and the X-bound functions together;

    plot all g-trajectories and the g-bound functions together; and

    plot the stopping time vs. initial belief for each initial belief.

   *** RECOMMENDATION:
      use M_ts >= 1,000 to make max experiment time long enough ***

3. ker_ctc_dynpro__multi_bib:

    * Constant Trial Costs ("CTC") *

    * Biased Initial Beliefs ("BIB") *

    Generate X- and g-data (trajectories) for a specified number of initial
    beliefs within a specified range
    (e.g., 100 initial beliefs between 0.01 and 0.99)
    for known evidence reliability and
    a specified constant trial cost 0 < c < 1,
    where bounds are computed by dynamic programming with c;

    plot all X-trajectories and the X-bounds together;

    plot all g-trajectories and the g-bounds together; and

    plot the stopping time vs. initial belief for each initial belief.

   *** RECOMMENDATION:
      use M_ts >= 1,000 to make max experiment time long enough ***

4. ker_ctc_dynpro__uib:

    * Constant Trial Costs ("CTC") *

    * Unbiased Initial Beliefs ("UIB") *

    Generate X- and g-data (trajectories) for a single unbiased initial belief
    (i.e., initial belief g_0 = 0.5)
    for known evidence reliability and
    a specified constant trial cost 0 < c < 1,
    where bounds are computed by dynamic programming with c;

    plot the X-trajectory and the X-bounds together; and

    plot the g-trajectory and the g-bounds together.

   *** RECOMMENDATION:
      use M_ts >= 1,000 to make max experiment time long enough ***

"""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from tqdm import tqdm



# Reshape a matrix Mx into a (Mx.shape[0]+M)xM matrix with Mx at beginning:
def expand_mat(Mx, M):
  N = Mx.shape[0]
  Mx_new = np.zeros((N+M, M))
  Mx_new[0:N, 0:M] = Mx
  return Mx_new



# Reshape an array arr into a (arr.shape[0]+M)-D array with arr at beginning:
def expand_arr(arr, M):
  arr_new = np.zeros(arr.shape[0]+M)
  arr_new[0:arr.shape[0]] = arr
  return arr_new



# Reshape a matrix Mx into a (m+1)xM matrix for a specified m >= 0:
def contract_mat(Mx, M, m):
  return Mx[0:m+1, 0:M]



# Reshape an array arr into an (m+1)-D array for a specified m >= 0:
def contract_arr(arr, m):
  return arr[0:m+1]



# Get NumPy array of M_dynpro discretized beliefs:
def disc_beliefs(M_dynpro):
  return np.array([(j+1)/(M_dynpro+1) for j in range(M_dynpro)])



# Get transition probability (see paper or presentation for derivation) from current belief g_cur to next belief g_next:
def p(g_cur, g_next, sig2, dt):
  coeff = np.sqrt(sig2/(8*np.pi*dt*np.exp(np.float128(sig2/dt))))
  f1 = np.sqrt((g_cur*(1-g_cur))/((g_next*(1-g_next))**3))
  f2 = np.exp( (-sig2/(2*dt)) * (np.log(np.sqrt((g_next*(1-g_cur))/(g_cur*(1-g_next))))**2))
  return coeff * f1 * f2



# Get matrix ps of transition probabilities with current beliefs down columns and next beliefs across rows:
def get_ps(gs, M_dynpro, sig2, dt):
  return np.array([[p(gs[j], gs[k], sig2, dt) for j in range(M_dynpro)] for k in range(M_dynpro)])



# Normalize transition probability matrix such that each row sums to 1:
def normalize_ps(ps):
  for j in range(ps.shape[0]):
    psum_j = np.sum(ps[j])
    ps[j] = ps[j] / psum_j
  return ps



# Get the maximum expected reward function Rstop for stopping the experiment at iteration n:
def get_Rstop(gs):
  return np.maximum(gs, 1-gs) # this is the same function for all iterations



# Get the expected reward function Rgo for continuing the experiment at iteration n:
def get_Rgo(gs, V_prev, ps, M_dynpro, c, dt):
  # Integral Approximation:
  Rgo = np.zeros(M_dynpro)
  for j in range(M_dynpro):
    Rgo[j] = np.dot(V_prev, ps[j]) - c*dt
  return Rgo # this function will need to be computed for every iteration n



# Interpolate the elementwise signed difference function between the stopping and continuing expected reward functions at iteration n:
def interp_Rdiff(gs, Rstop, Rgo, interp_kind): # interp_kind: ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next'
  return interp1d(gs, Rstop-Rgo, kind=interp_kind) # this returns a callable interpolant *function*



# Estimate the root of an interpolant function on interval [a, b] using rootfinding method root_method:
def get_interpolant_root(a, b, f, root_method): # root_method: ‘bisect’, ‘brentq’, ‘brenth’, ‘ridder’, ‘toms748’, ‘newton’, ‘secant’, or ‘halley’
  return root_scalar(f, bracket=[a, b], method=root_method).root



# Estimate the belief bounds (g-values on horizontal axis where Rstop and Rgo intersect) for iteration n >= 2:
def get_gbounds(gs, Rstop, Rgo, interp_kind, root_method):
  f = interp_Rdiff(gs, Rstop, Rgo, interp_kind)
  gL = get_interpolant_root(gs[0], 0.5, f, root_method)
  gH = get_interpolant_root(0.5, gs[-1], f, root_method)
  return gH, gL, f



# Estimate the value function output (on vertical axis) at the belief bounds for iteration n >= 2:
def get_V_at_g_bounds(gH, gL, f):
  return f(gH), f(gL), f # note: by symmetry, these should be the same



# Transform the upper and lower g-bounds into corresponding X-bounds:
def get_ker_Xbounds(sig2, gH, gL):
  return (sig2/2)*np.log(gH/(1-gH)), (sig2/2)*np.log(gL/(1-gL))



# Generate the acc. evidence and belief stochastic process data:
def generate_data(z, g_0, sig2, dt, M_ts):

  """

  Parameters:

  - z:          the actual hidden state (z = -1 or z = 1) for data generation

  - g_0:        initial belief that z = 1; 0 < g_0 < 1; use g_0 = 0.5 for unbiased initial belief

  - sig2:       constant noise variance

  - dt:         constant trial duration

  - M_ts:       number of discrete times; empirically, M_ts >= 1,000 appears to be sufficient for dt >= 0.0001

  """


  gs = np.zeros(M_ts)

  if g_0 > 0 and g_0 < 1: gs[0] = g_0
  else: gs[0] = 0.5

  Xs = np.zeros(M_ts)

  # Initialize acc. evidence X to the value corresponding to current initial belief g_0:
  Xs[0] = (sig2/2)*np.log(g_0/(1-g_0))

  # Populate Xs and gs arrays:
  for n in range(1, M_ts):
    # Accumulate evidence from a normal distribution with mean z*dt and variance sig2*dt:
    Xs[n] = Xs[n-1] + np.random.normal(z*dt, np.sqrt(sig2*dt))
    gs[n] = 1/(1+np.exp(((-2*z)/sig2)*Xs[n]))

  return Xs, gs



# Check whether current X-value is on or outside of current bounds:
def boundcheck(X, XH, XL):
  if X >= XH: return XH # if boundcheck(...) == XH: upper bound was hit
  if X <= XL: return XL # if boundcheck(...) == XL: lower bound was hit
  return X              # otherwise:                no bound was hit


# Get the optimal X- and g-bounds at the current cost value:
def ker_ctc__dynpro_bounds(sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, c, t, anim):

  """

  Parameters:



  - sig2 > 0:       noise variance

  - dt > 0:         trial duration (timestep)

  - M_dynpro >= 2:  number of discrete beliefs on *open* interval (0, 1)

  - tol > 0:        value function convergence tolerance

  - interp_kind:    type of interpolation used for expected reward difference function; ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next'

  - root_method:    rootfinding method used to estimate belief bounds; ‘bisect’, ‘brentq’, ‘brenth’, ‘ridder’, ‘toms748’, ‘newton’, ‘secant’, or ‘halley’

  - sfs >= 0:       number of significant figures after the decimal point for numbers in plots

  - 0 < c < 1:      constant cost used in value function iteration

  - t > 0:          current time

  - anim:           Boolean for whether to generate and download animations of value function convergence; True or False

  """


  # Discretize beliefs on (0, 1):
  gs = disc_beliefs(M_dynpro)

  # Get transition probability matrix ps, where ps[j][k] = p(g_k|g_j) = p(g_next|g_cur):
  ps = get_ps(gs, M_dynpro, sig2, dt)

  # Normalize transition probability matrix s.t. Σ[ps[j][k] for k in range(M_dynpro)] = 1:
  ps = normalize_ps(ps)

  # Get expected stopping reward function Rstop (same for all dynamic programming iterations n):
  Rstop = get_Rstop(gs)

  # Initialize "continue" expected reward function matrix:
  Rgos = np.zeros((2, M_dynpro)) # will store the "continue" expected reward functions {Rgo_n(g)} for all iterations n

  # Initialize value function matrix:
  Vs = np.zeros((2, M_dynpro))   # will store the value functions {V_n(g)} for all iterations n

  # Initialize successive value function difference matrix:
  Vdiffs = np.zeros((1, M_dynpro)) # will store the elementwise absolute difference function between consecutive value functions V_n and V_(n-1) for all iterations n >= 1; note: there will only be n-1 of these functions

  # Initialize lpper g- and lower g-bound function NumPy arrays:
  gHs = np.zeros(2)      # will store upper g-values at intersections between Rstop and Rgo functions; note: only meaningful for n >= 2 (since no intersection for n < 2)
  gLs = np.zeros(2)      # will store lower g-values at intersections between Rstop and Rgo functions; note: only meaningful for n >= 2 (since no intersection for n < 2)


  # Initialize NumPy arrays for the output of the value function at each g-bound:
  Vs_at_gH = np.zeros(2) # will store the estimated n-th value function output for the upper g-value at intersection between Rstop and Rgo functions; note: only meaningful for n >= 2 (since no intersection for n < 2)
  Vs_at_gL = np.zeros(2) # will store the estimated n-th value function output for the lower g-value at intersection between Rstop and Rgo functions; note: only meaningful for n >= 2 (since no intersection for n < 2)

  """______________________________ Run the dynamic programming (DP) algorithm: ______________________________"""
  # DP Iteration n = 0: Vs[0] and Rgos[0] are already zeros; ignore gLs[0], gHs[0], Vs_at_gL[0], and Vs_at_gH[0] (all zero)

  # DP Iteration n = 1:
  # Rgos[1] is already zeros

  # Manually update Vs and Vdiffs:
  Vs[1] = Rstop     # since Rgos[1] = np.zeros(M_dynpro)
  Vdiffs[0] = Rstop # since Vs[1] - Vs[0] = Rstop - np.zeros(M_dynpro) = Rstop
  # Ignore gLs[1], gHs[1], Vs_at_gL[1], and Vs_at_gH[1] (all zero)

  # DP Iterations n >= 2:
  n = 2
  while True:

    # Check whether bigger matrices and arrays are needed:
    if n >= Vs.shape[0]:
      # Expand Vs and Rgos matrices to fit M_dynpro more rows:
      Vs = expand_mat(Vs, M_dynpro)
      Rgos = expand_mat(Rgos, M_dynpro)

      # Expand absolute consecutive value function difference matrix to fit M_dynpro more rows:
      Vdiffs = expand_mat(Vdiffs, M_dynpro)

      # Expand belief bounds arrays gLs and gHs to fit M_dynpro more bounds:
      gHs = expand_arr(gHs, M_dynpro)
      gLs = expand_arr(gLs, M_dynpro)


      # Expand value function output at g-bounds arrays to fit M_dynpro more outputs:
      Vs_at_gH = expand_arr(Vs_at_gH, M_dynpro)
      Vs_at_gL = expand_arr(Vs_at_gL, M_dynpro)


    # Get reward function for continuing ("go"):
    Rgos[n] = get_Rgo(gs, Vs[n-1], ps, M_dynpro, c, dt)

    # Compute value function as max of "stop" and "go" functions at each g (elementwise):
    Vs[n] = np.maximum(Rstop, Rgos[n])

    # Compute the absolute consecutive value function difference (elementwise) function:
    Vdiffs[n-1] = np.abs(Vs[n] - Vs[n-1])

    # Get stop-go reward difference interpolant function d, and estimate roots of interpolated difference function on gs interval:
    gHs[n], gLs[n], d = get_gbounds(gs, Rstop, Rgos[n], interp_kind, root_method)

    # Estimate value function outputs at the belief bounds (roots of interpolated difference function):
    Vs_at_gH[n], Vs_at_gL[n], f = get_V_at_g_bounds(gHs[n], gLs[n], interp_Rdiff(gs, Vs[n], np.zeros(M_dynpro), interp_kind))

    # Break if consecutive value functions have small enough maximum difference (assumes that this implies convergence):
    if np.max(Vdiffs[n-1]) < tol:
      break

    # Proceed to the next iteration if no convergence yet:
    n += 1

  # Get NumPy array [0, 1, 2, ..., n] of iteration indices:
  ns = np.arange(n+1)

  # Contract Vs and Rgos matrices to remove extra rows after the nth iteration:
  Vs = contract_mat(Vs, M_dynpro, n)
  Rgos = contract_mat(Rgos, M_dynpro, n)

  # Contract Vdiffs matrix to remove extra rows after the (n-1)th iteration:
  Vdiffs = contract_mat(Vdiffs, M_dynpro, n-1)

  # Remove extra entries after the nth iteration in belief bounds arrays:
  gLs = contract_arr(gLs, n)
  gHs = contract_arr(gHs, n)

  # Remove extra entries after the nth iteration in V output at bounds arrays:
  Vs_at_gH = contract_arr(Vs_at_gH, n)
  Vs_at_gL = contract_arr(Vs_at_gL, n)

  """_________________________________________________________________"""
  # Generate video (animation) of value function convergence if anim == True:
  if anim:

    # Create a figure and axes:
    fig, ax = plt.subplots()

    # Update function:
    def update(frame):

      # Clear the axis for every frame:
      ax.cla()

      # Set figure and font sizes:
      plt.rcParams['figure.figsize'] = [20, 20]
      plt.rcParams['font.size'] = 20

      # Set the horizontal (belief g) and vertical (reward and value) axis bounds:
      ax.set_xlim(0, 1)
      ax.set_ylim(0, 1)

      # Initialization:
      line_1, = ax.plot([], [], lw=5)
      line_2, = ax.plot([], [], lw=5)
      line_3, = ax.plot([], [], lw=5)

      # Plot the data:
      line_1.set_data(gs, Rstop)
      line_2.set_data(gs, Rgos[frame])
      line_3.set_data(gs, Vs[frame])

      # Assign colors and line widths:
      line_1.set_color('red')
      line_2.set_color('blue')
      line_3.set_color('green')

      # Show belief bounds at current iteration (frame) n for iterations n >= 2:
      if frame >= 2:

        # Plot the point (gLs[frame], Vs_at_gL[frame]):
        plt.plot(gLs[frame], Vs_at_gL[frame], marker='o', color='black', markersize=25)
        plt.text(gLs[frame]-0.25, Vs_at_gL[frame]-0.0375, f'$(g_L, V(g_L))=($' + str(np.round(gLs[frame], sfs)) + ", " + str(np.round(Vs_at_gL[frame], sfs)) + ")", color='black', ha='left', va='bottom', fontsize=16)

        # Plot the point (gHs[frame], Vs_at_gH[frame]):
        plt.plot(gHs[frame], Vs_at_gH[frame], marker='o', color='black', markersize=25)
        plt.text(gHs[frame]+0.25, Vs_at_gH[frame]-0.0375, f'$(g_H, V(g_H))=($' + str(np.round(gHs[frame], sfs)) + ", " + str(np.round(Vs_at_gH[frame], sfs)) + ")", color='black', ha='right', va='bottom', fontsize=16)

      # Set title, axis labels, and legend for plot:
      ax.set_title("Expected Reward and Value Functions vs. " + f'$g$' + " for Iteration " + f'$n = $' + str(frame) + " at Time " + f'$t = $' + str(t) + "\nParameters:\n" + f'$M = $' + str(M_dynpro) + " Discrete Beliefs on " + f'$(0, 1)$' + ",\nVariance " + f'$σ_ε^2 = $' + str(sig2) + ", Timestep " + f'$dt = $' + str(dt) + ",\nCost " + f'$c = $' + str(c) + ", Convergence Tolerance " + f'$tol = $' + str(tol))
      ax.set_xlabel('Belief, ' + f'$g$')
      ax.set_ylabel('Expected Reward and Value: ' + f'$\max$' + r'$\{$' + f'$R_n(g,a=-1), R_n(g,a=1)$' + r'$\}$' + ", " + f'$R_n(g,a=0)$' + ", and" + f'$V_n(g)$')
      ax.legend([f'$\max$' + r'$\{$' + f'$R_n(g,a=-1), R_n(g,a=1)$' + r'$\}$', f'$R_n(g,a=0)$', f'$V_n(g)$'])

      return line_1, line_2, line_3, # note the comma after line; this unpacks the tuple

    # Create the animation:
    ani = animation.FuncAnimation(fig, update, frames=n, interval=20, blit=True)

    # Save the animation:
    writer = animation.FFMpegWriter(fps=30)
    ani.save(str(M_dynpro) + 'beliefs__' + str(sig2) + '_sig2__' + str(dt) + '_dt__' + str(c) + '_c__' + str(tol) + '_tol___' + 'TIME_' + str(t) + '.mp4', writer=writer)

  # Define the values to return:
  gH = gHs[-1] # the convergent upper g-bound
  gL = gLs[-1] # the convergent lower g-bound
  XH, XL = get_ker_Xbounds(sig2, gH, gL) # the corresponding convergent upper and lower X-bounds

  return XH, XL, gH, gL





# Compute the KER bound function arrays:
def ker__dynpro_bound_functions(sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, M_ts, c):

  """

  Parameters:

  - sig2 > 0:            noise variance

  - dt > 0:              trial duration (timestep)

  - M_dynpro >= 2:       number of discrete beliefs on *open* interval (0, 1)

  - tol > 0:             value function convergence tolerance

  - interp_kind:         type of interpolation used for expected reward difference function; ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next'

  - root_method:         rootfinding method used to estimate belief bounds; ‘bisect’, ‘brentq’, ‘brenth’, ‘ridder’, ‘toms748’, ‘newton’, ‘secant’, or ‘halley’

  - sfs >= 0:            number of significant figures after the decimal point for numbers in plots

  - M_ts:                number of discrete times; empirically, M_ts >= 1,000 appears to be sufficient for dt >= 0.0001

  - c = c(t):            cost function array used in value function iteration

  """

  # Initialize upper X- and lower X-bound function NumPy arrays:
  XHs = np.zeros(M_ts)
  XLs = np.zeros(M_ts)

  # Initialize lpper g- and lower g-bound function NumPy arrays:
  gHs = np.zeros(M_ts)
  gLs = np.zeros(M_ts)

  # Populate the bound arrays:
  for n in tqdm(range(M_ts)):
    XHs[n], XLs[n], gHs[n], gLs[n] = ker_ctc__dynpro_bounds(sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, c[n], n*dt, False)
  return XHs, XLs, gHs, gLs



# Plot KER trajectories for five example cost functions with unbiased initial beliefs:
def ker_tdtc_dynpro__multi_cost_examples(z, sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, M_ts):

    """

    Parameters:

    - z:                   the actual hidden state (z = -1 or z = 1) for data generation

    - sig2 > 0:            noise variance

    - dt > 0:              trial duration (timestep)

    - M_dynpro >= 2:       number of discrete beliefs on *open* interval (0, 1)

    - tol > 0:             value function convergence tolerance

    - interp_kind:         type of interpolation used for expected reward difference function; ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next'

    - root_method:         rootfinding method used to estimate belief bounds; ‘bisect’, ‘brentq’, ‘brenth’, ‘ridder’, ‘toms748’, ‘newton’, ‘secant’, or ‘halley’

    - sfs >= 0:            number of significant figures after the decimal point for numbers in plots

    - M_ts:                number of discrete times; empirically, M_ts >= 1,000 appears to be sufficient for dt >= 0.0001

    """

    # Initialize accumulated evidence and belief matrices:
    Xs = np.zeros((5, M_ts)) # each row is the X-trajectory for the corresponding cost function
    gs = np.zeros((5, M_ts)) # each row is the g-trajectory for the corresponding cost function

    # Get NumPy array of M_ts total times of increment size (trial duration) dt; empirically, M_ts >= 20,000 appears to be sufficient for dt >= 0.0001
    ts = np.arange(0, M_ts*dt, dt)

    # Initialize cost function matrix:
    cs = np.zeros((5, M_ts)) # each row will be the output of the corresponding cost function over the discrete time inputs

    # Define the five cost functions:
    def c1(t):
      return np.exp(-1/(t+1))
    def c2(t):
      return 1/(1+np.exp(-t))
    def c3(t):
      return 1/(1+1/(1+t))
    def c4(t):
      return np.tanh(t)
    def c5(t):
      return (2/np.pi)*np.arctan(t)

    # Populate the cost function matrix:
    cs[0] = c1(ts)
    cs[1] = c2(ts)
    cs[2] = c3(ts)
    cs[3] = c4(ts)
    cs[4] = c5(ts)

    # Get NumPy array of textual expressions of the cost functions; for use in plotting (e.g., titles, legends):
    cfunstrs = np.array([r'$c(t) = e^{-1/(t+1)}$', r'$c(t) = 1/(1+e^{-t})$', r'$c(t) = 1/(1+1/(1+t))$', r'$c(t) = tanh(t)$', r'$c(t) = \frac{2}{\pi}arctan(t)$'])

    # Define a line color for each cost function; for use in plotting:
    c_colors = np.array(['black', 'blue', 'green', 'purple', 'orange'])


    """

    REMARK:

    Dynamic programming for optimal bounds is ~slow.

    So it is preferable to compute only as many bound function outputs as are
    required for the trajectory to hit the upper or lower bound.

    Therefore, the bound function matrices will be populated while each
    trajectory is being checked for bound-hitting, rather than computing
    the full bound function output rows and then checking for bound-hitting.

    """

    # Get NumPy array of bound-hitting ("stopping") times:
    tstops = np.zeros(5)

    # Get NumPy array of bound-hitting ("stopping") array indices:
    nstops = np.zeros(5, dtype='int')


    # Generate the full X- and g-trajectories from the chosen parameters (sig2, dt):
    Xfull, gfull = generate_data(z, 0.5, sig2, dt, M_ts) # these will be modified for each cost function and stored in respective rows of Xs and gs

    # Initialize upper X- and lower X-bound function matrices:
    XHs = np.zeros((5, M_ts))
    XLs = np.zeros((5, M_ts))

    # Initialize upper g- and lower g-bound function matrices:
    gHs = np.zeros((5, M_ts))
    gLs = np.zeros((5, M_ts))

    """__________________ Run a total of 5 experiments (one for each cost function): __________________"""
    for c_n in range(5):

      # Populate the full X- and g-trajectory rows:
      Xs[c_n] = Xfull
      gs[c_n] = gfull

      # Find the first X-value (and its index and time) in the full trajectory to fall outside of the bounds:
      for n in tqdm(range(M_ts)):

        # Populate the X- and g-bound function rows:
        XHs[c_n, n], XLs[c_n, n], gHs[c_n, n], gLs[c_n, n] = ker_ctc__dynpro_bounds(sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, cs[c_n, n], ts[n], False)

        # Check whether the current X-value is on or outside of the current bound values:
        bc = boundcheck(Xs[c_n][n], XHs[c_n, n], XLs[c_n, n])

        if bc == XHs[c_n, n] or bc == XLs[c_n, n]:
          # EITHER X(t) hit upper bound between t-dt and t OR X(t) hit lower bound between t-dt and t
          # Estimate the bound-hitting time to be the current time (and thus the current X to be the value of the bound that was hit):
          Xs[c_n][n] = bc

          # Update the current belief to that corresponding to the bound that was hit:
          gs[c_n][n] = 1/(1+np.exp(((-2*z)/sig2)*Xs[c_n][n]))

          # Update the stopping index and the stopping time to be the current values:
          nstops[c_n] = n
          tstops[c_n] = ts[n]
          break


      # Plot cost function, bounds, and full trajectories for each cost function:
      fig, axs = plt.subplots(2, 3)

      fig.set_size_inches(24, 16)

      # Plot cost function vs. time:
      axs[0, 0].plot(ts, cs[c_n], label=f'$c(t)$', color=c_colors[c_n], lw=7)
      axs[0, 0].set_xlabel('Time, ' + f'$t$')
      axs[0, 0].set_ylabel('Cost, ' + f'$c(t)$')
      axs[0, 0].set_title('Cost Function ' + cfunstrs[c_n] + ' vs. Time ' + f'$t$')

      # Plot X-bound functions vs. time:
      axs[0, 1].plot(ts[0:nstops[c_n]+1], XHs[c_n][0:nstops[c_n]+1], label=f'$X_H(t)$', lw=7)
      axs[0, 1].plot(ts[0:nstops[c_n]+1], XLs[c_n][0:nstops[c_n]+1], label=f'$X_L(t)$', lw=7)
      axs[0, 1].set_xlabel('Time, ' + f'$t$')
      axs[0, 1].set_ylabel('Accumulated Evidence Bounds, ' + f'$X_H(t)$' + ' and ' + f'$X_L(t)$')
      axs[0, 1].set_title('Accumulated Evidence Bounds vs. Time ' + f'$t$')
      axs[0, 1].legend(['Upper Bound ' + f'$X_H(t)$', 'Lower Bound ' + f'$X_L(t)$'])

      # Plot g-bound functions vs. time:
      axs[0, 2].plot(ts[0:nstops[c_n]+1], gHs[c_n][0:nstops[c_n]+1], label=f'$g_H(t)$', lw=7)
      axs[0, 2].plot(ts[0:nstops[c_n]+1], gLs[c_n][0:nstops[c_n]+1], label=f'$g_L(t)$', lw=7)
      axs[0, 2].set_xlabel('Time, ' + f'$t$')
      axs[0, 2].set_ylabel('Belief Bounds, ' + f'$g_H(t)$' + ' and ' + f'$g_L(t)$')
      axs[0, 2].set_title('Belief Bounds vs. Time ' + f'$t$')
      axs[0, 2].legend(['Upper Bound ' + f'$g_H(t)$', 'Lower Bound ' + f'$g_L(t)$'])

      # Plot full X-trajectory:
      axs[1, 0].plot(ts[0:nstops[c_n]+1], Xs[c_n][0:nstops[c_n]+1], label=f'$X(t)$', color='red', lw=3)
      axs[1, 0].set_xlabel('Time, ' + f'$t$')
      axs[1, 0].set_ylabel('Accumulated Evidence, ' + f'$X(t)$')
      axs[1, 0].set_title('Accumulated Evidence vs. Time ' + f'$t$')

      # Plot g-trajectory:
      axs[1, 1].plot(ts[0:nstops[c_n]+1], gs[c_n][0:nstops[c_n]+1], label=f'$g(t)$', color='red', lw=3)
      axs[1, 1].set_xlabel('Time, ' + f'$t$')
      axs[1, 1].set_ylabel('Belief, ' + f'$g(t)$')
      axs[1, 1].set_title('Belief vs. Time ' + f'$t$')

      plt.tight_layout()
      plt.show()

      # Plot Xs, XHs, and XLs vs. ts on the same graph:
      fig, ax = plt.subplots()
      fig.set_size_inches(24, 14)
      plt.rcParams['font.size'] = 20
      ax.plot(ts[0:nstops[c_n]+1], Xs[c_n][0:nstops[c_n]+1], label=f'$X(t)$', color='red', lw=3)
      ax.plot(ts[0:nstops[c_n]+1], XHs[c_n][0:nstops[c_n]+1], label=f'$X_H(t)$', lw=5)
      ax.plot(ts[0:nstops[c_n]+1], XLs[c_n][0:nstops[c_n]+1], label=f'$X_L(t)$', lw=5)
      ax.set_xlabel('Time, ' + f'$t$')
      ax.set_ylabel('Accumulated Evidence, ' + f'$X(t)$')
      plt.title('Accumulated Evidence ' + f'$X(t)$' + ' vs. Time ' + f'$t$' + ';\tCost Function ' + cfunstrs[c_n] + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))
      ax.plot(ts[nstops[c_n]], Xs[c_n][nstops[c_n]], marker='o', color='black', markersize=12)
      ax.legend(['Accumulated Evidence ' + f'$X(t)$', 'Upper Bound ' + f'$X_H(t)$', 'Lower Bound ' + f'$X_L(t)$', f'$(t, X(t))=($' + str(np.round(ts[nstops[c_n]], 5)) + ', ' + str(np.round(Xs[c_n][nstops[c_n]], 5)) + ')'])
      plt.show()

      # Plot gs, gHs, and gLs vs. ts on the same graph:
      fig, ax = plt.subplots()
      fig.set_size_inches(24, 14)
      ax.plot(ts[0:nstops[c_n]+1], gs[c_n][0:nstops[c_n]+1], label=f'$g(t)$', color='red', lw=3)
      ax.plot(ts[0:nstops[c_n]+1], gHs[c_n][0:nstops[c_n]+1], label=f'$g_H(t)$', lw=5)
      ax.plot(ts[0:nstops[c_n]+1], gLs[c_n][0:nstops[c_n]+1], label=f'$g_L(t)$', lw=5)
      ax.set_xlabel('Time, ' + f'$t$')
      ax.set_ylabel('Belief, ' + f'$g(t)$')
      plt.title('Belief ' + f'$g(t)$' + ' vs. Time ' + f'$t$' + ';\tCost Function ' + cfunstrs[c_n] + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))
      ax.plot(ts[nstops[c_n]], gs[c_n][nstops[c_n]], marker='o', color='black', markersize=12)
      ax.legend(['Belief ' + f'$g(t)$', 'Upper Bound ' + f'$g_H(t)$', 'Lower Bound ' + f'$g_L(t)$', f'$(t, g(t))=($' + str(np.round(ts[nstops[c_n]], 5)) + ', ' + str(np.round(gs[c_n][nstops[c_n]], 5)) + ')'])
      plt.show()
    """____________________________________________________________________________________________________"""

    return ts, cs, Xs, gs, XHs, XLs, gHs, gLs, nstops, tstops



# Plot KER trajectories using cost function c(t)=tanh(t) for a total of M_gs initial beliefs between g_0_start and g_0_stop:
def ker_tdtc_dynpro__multi_bib_example__tanh(z, sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, M_ts, g_0_start, g_0_stop, M_gs):

  """

  Parameters:

  - z:                   the actual hidden state (z = -1 or z = 1) for data generation

  - sig2 > 0:            noise variance

  - dt > 0:              trial duration (timestep)

  - M_dynpro >= 2:       number of discrete beliefs on *open* interval (0, 1)

  - tol > 0:             value function convergence tolerance

  - interp_kind:         type of interpolation used for expected reward difference function; ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next'

  - root_method:         rootfinding method used to estimate belief bounds; ‘bisect’, ‘brentq’, ‘brenth’, ‘ridder’, ‘toms748’, ‘newton’, ‘secant’, or ‘halley’

  - sfs >= 0:            number of significant figures after the decimal point for numbers in plots

  - M_ts:                number of discrete times; empirically, M_ts >= 1,000 appears to be sufficient for dt >= 0.0001

  - g_0_start:           initial belief that z = 1 for first experiment; 0 < g_0_start <= g_0_stop

  - g_0_stop:            initial belief that z = 1 for last experiment;  g_0_start <= g_0_stop < 1

  - M_gs:                number of initial beliefs to generate (number of experiments) between g_0_start and g_0_stop (inclusive)

  """

  # Get NumPy array of M_gs total initial beliefs to plot between g_0_start and g_0_stop:
  g_0s = np.linspace(g_0_start, g_0_stop, M_gs)

  # Initialize accumulated evidence and belief matrices:
  Xs = np.zeros((M_gs, M_ts)) # each row is the X-trajectory for the corresponding initial belief
  gs = np.zeros((M_gs, M_ts)) # each row is the g-trajectory for the corresponding initial belief

  # Get NumPy array of M_ts total times of increment size (trial duration) dt; empirically, M_ts >= 20,000 appears to be sufficient for dt >= 0.0001
  ts = np.arange(0, M_ts*dt, dt)

  # Get NumPy array of chosen trial cost function (here, c(t) = tanh(t)) outputs over time input array ts:
  cs = np.tanh(ts)
  cfunstr = r'$c(t) = tanh(t)$' # textual expression of chosen trial cost function; for use in plotting (e.g., titles, legends)

  # Get NumPy array of bound-hitting ("stopping") times:
  tstops = np.zeros(M_gs)

  # Get NumPy array of bound-hitting ("stopping") array indices:
  nstops = np.zeros(M_gs, dtype='int')

  """___ Generate the X- and g-bound functions from the chosen cost function and parameters (sig2, dt): ___"""
  # Initialize upper X- and lower X-bound function NumPy arrays:
  XHs = np.zeros(M_ts)
  XLs = np.zeros(M_ts)

  # Initialize lpper g- and lower g-bound function NumPy arrays:
  gHs = np.zeros(M_ts)
  gLs = np.zeros(M_ts)

  # Populate the bound arrays:
  for n in tqdm(range(M_ts)):
    XHs[n], XLs[n], gHs[n], gLs[n] = ker_ctc__dynpro_bounds(sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, cs[n], ts[n], False)
  """____________________________________________________________________________________________________"""


  """__________________ Run a total of M_gs experiments (one for each initial belief): __________________"""
  for g_n in tqdm(range(M_gs)):

    # Get full (including after hitting a bound) X and g trajectories for the current initial belief:
    Xs[g_n], gs[g_n] = generate_data(z, g_0s[g_n], sig2, dt, M_ts)

    # Find the first X-value (and its index and time) in the full trajectory to fall outside of the bounds:
    for n in range(M_ts):

      # Check whether the current X-value is on or outside of the current bound values:
      bc = boundcheck(Xs[g_n][n], XHs[n], XLs[n])

      if bc == XHs[n] or bc == XLs[n]:
        # EITHER X(t) hit upper bound between t-dt and t OR X(t) hit lower bound between t-dt and t
        # Estimate the bound-hitting time to be the current time (and thus the current X to be the value of the bound that was hit):
        Xs[g_n][n] = bc

        # Update the current belief to that corresponding to the bound that was hit:
        gs[g_n][n] = 1/(1+np.exp(((-2*z)/sig2)*Xs[g_n][n]))

        # Update the stopping index and the stopping time to be the current values:
        nstops[g_n] = n
        tstops[g_n] = ts[n]
        break

    # Check whether the current initial belief starts outside of the initial bounds, and print an error message if so:
    if nstops[g_n] == 0:
      if g_0s[g_n] >= gHs[0]: print('\n\n\nBiased Initial Belief ' + 'g_0 = ' + str(g_0s[g_n]) + ' is Too High for Cost Function c(t) = tanh(t)!\n\n\n')
      if g_0s[g_n] <= gLs[0]: print('\n\n\nBiased Initial Belief ' + 'g_0 = ' + str(g_0s[g_n]) + ' is Too Low for Cost Function c(t) = tanh(t)!\n\n\n')
      continue
  """____________________________________________________________________________________________________"""


  """____________________________________________ Plot Results: ____________________________________________"""
  # Get the number of trials required for a bound to be hit for the initial belief with the longest experiment:
  max_nstops = np.max(nstops)

  # Plotting line width setup:
  linewidth = 1
  if M_gs <= 5: linewidth = 3
  elif M_gs <= 10: linewidth = 2

  # Plot Xs, XHs, and XLs vs. ts for all initial beliefs on the same graph:
  fig, ax = plt.subplots()
  plt.rcParams['font.size'] = 20
  fig.set_size_inches(24, 14)

  ax.plot(ts[0:max_nstops+1], XHs[0:max_nstops+1], label='Upper Bound ' + f'$X_H(t)$', lw=5)
  ax.plot(ts[0:max_nstops+1], XLs[0:max_nstops+1], label='Lower Bound ' + f'$X_L(t)$', lw=5)

  for g_n in range(M_gs):
    if nstops[g_n] > 0: # avoid plotting a zero-length trajectory
      ax.plot(ts[0:nstops[g_n]+1], Xs[g_n][0:nstops[g_n]+1], label=f'$X(t)_{{g_0={np.round(g_0s[g_n], 5)}}}$', lw=linewidth)
      ax.plot(tstops[g_n], Xs[g_n][nstops[g_n]], marker='o', color='black', markersize=12)

  ax.set_xlabel('Time, ' + f'$t$')
  ax.set_ylabel('Accumulated Evidence, ' + f'$X(t)$')

  # Only show a full legend for a small number (e.g., <=5) of trajectories:
  if M_gs > 5: plt.legend(['Upper Bound ' + f'$X_H(t)$', 'Lower Bound ' + f'$X_L(t)$'])
  else: plt.legend()

  plt.title('Accumulated Evidence ' + f'$X(t)$' + ' vs. Time ' + f'$t$' + ' for ' + str(M_gs) + ' Initial Beliefs Between ' + str(g_0_start) + ' and ' + str(g_0_stop) + '\nCost Function ' + cfunstr + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))

  plt.show()

  # Plot gs, gHs, and gLs vs. ts for all initial beliefs on the same graph:
  fig, ax = plt.subplots()
  plt.rcParams['font.size'] = 20
  fig.set_size_inches(24, 14)

  ax.plot(ts[0:max_nstops+1], gHs[0:max_nstops+1], label='Upper Bound ' + f'$g_H(t)$', lw=5)
  ax.plot(ts[0:max_nstops+1], gLs[0:max_nstops+1], label='Lower Bound ' + f'$g_L(t)$', lw=5)

  for g_n in range(M_gs):
    if nstops[g_n] > 0: # avoid plotting a zero-length trajectory
      ax.plot(ts[0:nstops[g_n]+1], gs[g_n][0:nstops[g_n]+1], label=f'$g(t)_{{g_0={np.round(g_0s[g_n], 5)}}}$', lw=linewidth)
      ax.plot(tstops[g_n], gs[g_n][nstops[g_n]], marker='o', color='black', markersize=12)

  ax.set_xlabel('Time, ' + f'$t$')
  ax.set_ylabel('Belief, ' + f'$g(t)$')

  # Only show a full legend for a small number (e.g., <=5) of trajectories:
  if M_gs > 5: plt.legend(['Upper Bound ' + f'$g_H(t)$', 'Lower Bound ' + f'$g_L(t)$'])
  else: plt.legend()

  plt.title('Belief ' + f'$g(t)$' + ' vs. Time ' + f'$t$' + ' for ' + str(M_gs) + ' Initial Beliefs Between ' + str(g_0_start) + ' and ' + str(g_0_stop) + '\nCost Function ' + cfunstr + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))

  plt.show()

  # Plot stopping time vs. initial belief:
  fig, ax = plt.subplots()
  plt.rcParams['font.size'] = 20
  fig.set_size_inches(24, 14)

  ax.plot(g_0s, tstops, lw=3)

  ax.set_xlabel('Initial Belief, ' + f'$g_0$')
  ax.set_ylabel('Stopping Time, ' + f'$t_{{stop}}$')

  plt.title('Stopping Time ' + f'$t_{{stop}}$' + ' vs. Initial Belief ' + f'$g_0$' + ' for ' + str(M_gs) + ' Initial Beliefs Between ' + str(g_0_start) + ' and ' + str(g_0_stop) + '\nCost Function ' + cfunstr + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))

  plt.show()
  """____________________________________________________________________________________________________"""

  return g_0s, ts, cs, Xs, gs, XHs, XLs, gHs, gLs, nstops, tstops



# Plot KER trajectories using specified constant cost 0 < c < 1 for a total of M_gs initial beliefs between g_0_start and g_0_stop:
def ker_ctc_dynpro__multi_bib(z, sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, M_ts, c, g_0_start, g_0_stop, M_gs):

  """

  Parameters:

  - z:                   the actual hidden state (z = -1 or z = 1) for data generation

  - sig2 > 0:            noise variance

  - dt > 0:              trial duration (timestep)

  - M_dynpro >= 2:       number of discrete beliefs on *open* interval (0, 1)

  - tol > 0:             value function convergence tolerance

  - interp_kind:         type of interpolation used for expected reward difference function; ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next'

  - root_method:         rootfinding method used to estimate belief bounds; ‘bisect’, ‘brentq’, ‘brenth’, ‘ridder’, ‘toms748’, ‘newton’, ‘secant’, or ‘halley’

  - sfs >= 0:            number of significant figures after the decimal point for numbers in plots

  - M_ts:                number of discrete times; empirically, M_ts >= 1,000 appears to be sufficient for dt >= 0.0001

  - 0 < c < 1:           constant trial cost

  - g_0_start:           initial belief that z = 1 for first experiment; 0 < g_0_start <= g_0_stop

  - g_0_stop:            initial belief that z = 1 for last experiment;  g_0_start <= g_0_stop < 1

  - M_gs:                number of initial beliefs to generate (number of experiments) between g_0_start and g_0_stop (inclusive)

  """

  # Get NumPy array of M_gs total initial beliefs to plot between g_0_start and g_0_stop:
  g_0s = np.linspace(g_0_start, g_0_stop, M_gs)

  # Initialize accumulated evidence and belief matrices:
  Xs = np.zeros((M_gs, M_ts)) # each row is the X-trajectory for the corresponding initial belief
  gs = np.zeros((M_gs, M_ts)) # each row is the g-trajectory for the corresponding initial belief

  # Get NumPy array of M_ts total times of increment size (trial duration) dt; empirically, M_ts >= 30,000 appears to be sufficient for dt >= 0.0001
  ts = np.arange(0, M_ts*dt, dt)

  # Get NumPy array of chosen trial cost function (here, c(t) = tanh(t)) outputs over time input array ts:
  cs = c*np.ones(M_ts)
  cfunstr = f'$c(t) = {c}$' # textual expression of chosen trial cost function; for use in plotting (e.g., titles, legends)

  # Get NumPy array of bound-hitting ("stopping") times:
  tstops = np.zeros(M_gs)

  # Get NumPy array of bound-hitting ("stopping") array indices:
  nstops = np.zeros(M_gs, dtype='int')



  """___ Generate the X- and g-bound functions from the chosen cost function and parameters (sig2, dt): ___"""
  # Initialize upper X- and lower X-bound function NumPy arrays:
  XHs = np.zeros(M_ts)
  XLs = np.zeros(M_ts)

  # Initialize lpper g- and lower g-bound function NumPy arrays:
  gHs = np.zeros(M_ts)
  gLs = np.zeros(M_ts)

  # Populate the bound arrays:
  for n in tqdm(range(M_ts)):
    XHs[n], XLs[n], gHs[n], gLs[n] = ker_ctc__dynpro_bounds(sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, cs[n], ts[n], False)
  """____________________________________________________________________________________________________"""


  """__________________ Run a total of M_gs experiments (one for each initial belief): __________________"""
  for g_n in tqdm(range(M_gs)):

    # Get full (including after hitting a bound) X and g trajectories for the current initial belief:
    Xs[g_n], gs[g_n] = generate_data(z, g_0s[g_n], sig2, dt, M_ts)

    # Find the first X-value (and its index and time) in the full trajectory to fall outside of the bounds:
    for n in range(M_ts):

      # Check whether the current X-value is on or outside of the current bound values:
      bc = boundcheck(Xs[g_n][n], XHs[n], XLs[n])

      if bc == XHs[n] or bc == XLs[n]:
        # EITHER X(t) hit upper bound between t-dt and t OR X(t) hit lower bound between t-dt and t
        # Estimate the bound-hitting time to be the current time (and thus the current X to be the value of the bound that was hit):
        Xs[g_n][n] = bc

        # Update the current belief to that corresponding to the bound that was hit:
        gs[g_n][n] = 1/(1+np.exp(((-2*z)/sig2)*Xs[g_n][n]))

        # Update the stopping index and the stopping time to be the current values:
        nstops[g_n] = n
        tstops[g_n] = ts[n]
        break

    # Check whether the current initial belief starts outside of the initial bounds, and print an error message if so:
    if nstops[g_n] == 0:
      if g_0s[g_n] >= gHs[0]: print('\n\n\nBiased Initial Belief ' + 'g_0 = ' + str(g_0s[g_n]) + ' is Too High for Cost Function c(t) = tanh(t)!\n\n\n')
      if g_0s[g_n] <= gLs[0]: print('\n\n\nBiased Initial Belief ' + 'g_0 = ' + str(g_0s[g_n]) + ' is Too Low for Cost Function c(t) = tanh(t)!\n\n\n')
      continue
  """____________________________________________________________________________________________________"""


  """____________________________________________ Plot Results: ____________________________________________"""
  # Get the number of trials required for a bound to be hit for the initial belief with the longest experiment:
  max_nstops = np.max(nstops)

  # Plotting line width setup:
  linewidth = 1
  if M_gs <= 5: linewidth = 3
  elif M_gs <= 10: linewidth = 2

  # Plot Xs, XHs, and XLs vs. ts for all initial beliefs on the same graph:
  fig, ax = plt.subplots()
  plt.rcParams['font.size'] = 20
  fig.set_size_inches(24, 14)

  ax.plot(ts[0:max_nstops+1], XHs[0:max_nstops+1], label='Upper Bound ' + f'$X_H(t)$', lw=5)
  ax.plot(ts[0:max_nstops+1], XLs[0:max_nstops+1], label='Lower Bound ' + f'$X_L(t)$', lw=5)

  for g_n in range(M_gs):
    if nstops[g_n] > 0: # avoid plotting a zero-length trajectory
      ax.plot(ts[0:nstops[g_n]+1], Xs[g_n][0:nstops[g_n]+1], label=f'$X(t)_{{g_0={np.round(g_0s[g_n], 5)}}}$', lw=linewidth)
      ax.plot(tstops[g_n], Xs[g_n][nstops[g_n]], marker='o', color='black', markersize=12)

  ax.set_xlabel('Time, ' + f'$t$')
  ax.set_ylabel('Accumulated Evidence, ' + f'$X(t)$')

  # Only show a full legend for a small number (e.g., <=5) of trajectories:
  if M_gs > 5: plt.legend(['Upper Bound ' + f'$X_H(t)$', 'Lower Bound ' + f'$X_L(t)$'])
  else: plt.legend()

  if M_gs == 1:
    plt.title('Accumulated Evidence ' + f'$X(t)$' + ' vs. Time ' + f'$t$' + ' for Unbiased Initial Belief ' + f'$g_0 = 0.5$' + '\nCost Function ' + cfunstr + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))
  else:
    plt.title('Accumulated Evidence ' + f'$X(t)$' + ' vs. Time ' + f'$t$' + ' for ' + str(M_gs) + ' Initial Beliefs Between ' + str(g_0_start) + ' and ' + str(g_0_stop) + '\nCost Function ' + cfunstr + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))

  plt.show()

  # Plot gs, gHs, and gLs vs. ts for all initial beliefs on the same graph:
  fig, ax = plt.subplots()
  plt.rcParams['font.size'] = 20
  fig.set_size_inches(24, 14)

  ax.plot(ts[0:max_nstops+1], gHs[0:max_nstops+1], label='Upper Bound ' + f'$g_H(t)$', lw=5)
  ax.plot(ts[0:max_nstops+1], gLs[0:max_nstops+1], label='Lower Bound ' + f'$g_L(t)$', lw=5)

  for g_n in range(M_gs):
    if nstops[g_n] > 0: # avoid plotting a zero-length trajectory
      ax.plot(ts[0:nstops[g_n]+1], gs[g_n][0:nstops[g_n]+1], label=f'$g(t)_{{g_0={np.round(g_0s[g_n], 5)}}}$', lw=linewidth)
      ax.plot(tstops[g_n], gs[g_n][nstops[g_n]], marker='o', color='black', markersize=12)

  ax.set_xlabel('Time, ' + f'$t$')
  ax.set_ylabel('Belief, ' + f'$g(t)$')

  # Only show a full legend for a small number (e.g., <=5) of trajectories:
  if M_gs > 5: plt.legend(['Upper Bound ' + f'$g_H(t)$', 'Lower Bound ' + f'$g_L(t)$'])
  else: plt.legend()

  if M_gs == 1:
    plt.title('Belief ' + f'$g(t)$' + ' vs. Time ' + f'$t$' + ' for Unbiased Initial Belief ' + f'$g_0 = 0.5$' + '\nCost Function ' + cfunstr + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))
  else:
    plt.title('Belief ' + f'$g(t)$' + ' vs. Time ' + f'$t$' + ' for ' + str(M_gs) + ' Initial Beliefs Between ' + str(g_0_start) + ' and ' + str(g_0_stop) + '\nCost Function ' + cfunstr + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))

  plt.show()

  # Plot stopping time vs. initial belief if M_gs > 1:
  if M_gs > 1:

    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 20
    fig.set_size_inches(24, 14)

    ax.plot(g_0s, tstops, lw=3)

    ax.set_xlabel('Initial Belief, ' + f'$g_0$')
    ax.set_ylabel('Stopping Time, ' + f'$t_{{stop}}$')
    plt.title('Stopping Time ' + f'$t_{{stop}}$' + ' vs. Initial Belief ' + f'$g_0$' + ' for ' + str(M_gs) + ' Initial Beliefs Between ' + str(g_0_start) + ' and ' + str(g_0_stop) + '\nCost Function ' + cfunstr + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))

    plt.show()
  """____________________________________________________________________________________________________"""

  return g_0s, ts, cs, Xs, gs, XHs, XLs, gHs, gLs, nstops, tstops



# Plot KER trajectories using a specified constant cost 0 < c < 1 for a single unbiased initial belief g_0 = 0.5:
def ker_ctc_dynpro__uib(z, sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, M_ts, c):

  """

  Parameters:

  - z:                   the actual hidden state (z = -1 or z = 1) for data generation

  - sig2 > 0:            noise variance

  - dt > 0:              trial duration (timestep)

  - M_dynpro >= 2:       number of discrete beliefs on *open* interval (0, 1)

  - tol > 0:             value function convergence tolerance

  - interp_kind:         type of interpolation used for expected reward difference function; ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next'

  - root_method:         rootfinding method used to estimate belief bounds; ‘bisect’, ‘brentq’, ‘brenth’, ‘ridder’, ‘toms748’, ‘newton’, ‘secant’, or ‘halley’

  - sfs >= 0:            number of significant figures after the decimal point for numbers in plots

  - M_ts:                number of discrete times; empirically, M_ts >= 1,000 appears to be sufficient for dt >= 0.0001

  - 0 < c < 1:           constant trial cost

  """

  return ker_ctc_dynpro__multi_bib(z, sig2, dt, M_dynpro, tol, interp_kind, root_method, sfs, M_ts, c, 0.5, 0.5, 1)






# Recommended Tests:

#ker_tdtc_dynpro__multi_cost_examples(1, 1, 0.0001, 100, 1e-3, 'quadratic', 'brentq', 5, 1000)

#ker_tdtc_dynpro__multi_bib_example__tanh(1, 1, 0.0001, 100, 1e-3, 'quadratic', 'brentq', 5, 1000, 0.01, 0.99, 100)

#ker_ctc_dynpro__multi_bib(1, 1, 0.0001, 100, 1e-3, 'quadratic', 'brentq', 5, 1000, 0.2, 0.001, 0.999, 1000)

#ker_ctc_dynpro__uib(1, 1, 0.001, 100, 1e-3, 'quadratic', 'brentq', 5, 1000, 0.2)
