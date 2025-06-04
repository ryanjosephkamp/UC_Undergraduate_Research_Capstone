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



Diffusion Process Direct Optimization

Known Evidence Reliability ("KER")



FUNCTIONS:

1. ker_tdtc_difproc__multi_cost_examples:

    * Time-Dependent Trial Costs ("TDTC") *

    Generate X- and g-data (trajectories)
    for known evidence reliability and time-dependent trial costs,
    where bound functions are computed by diffusion process direct optimization
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
      use M_ts >= 20,000 to make max experiment time long enough ***

2. ker_tdtc_difproc__multi_bib_example__tanh:

    * Time-Dependent Trial Costs ("TDTC") *

    * Biased Initial Beliefs ("BIB") *

    Generate X- and g-data (trajectories) for a specified number of initial
    beliefs within a specified range
    (e.g., 100 initial beliefs between 0.01 and 0.99)
    for known evidence reliability and time-dependent trial costs,
    where bound functions are computed by diffusion process direct optimization
    with cost function c(t) = tanh(t);

    plot all X-trajectories and the X-bound functions together;

    plot all g-trajectories and the g-bound functions together; and

    plot the stopping time vs. initial belief for each initial belief.

   *** RECOMMENDATION:
      use M_ts >= 20,000 to make max experiment time long enough ***

3. ker_ctc_difproc__multi_bib:

    * Constant Trial Costs ("CTC") *

    * Biased Initial Beliefs ("BIB") *

    Generate X- and g-data (trajectories) for a specified number of initial
    beliefs within a specified range
    (e.g., 100 initial beliefs between 0.01 and 0.99)
    for known evidence reliability and
    a specified constant trial cost 0 < c < 1,
    where bounds are computed by diffusion process direct optimization with c;

    plot all X-trajectories and the X-bounds together;

    plot all g-trajectories and the g-bounds together; and

    plot the stopping time vs. initial belief for each initial belief.

   *** RECOMMENDATION:
      use M_ts >= 30,000 to make max experiment time long enough ***

4. ker_ctc_difproc__uib:

    * Constant Trial Costs ("CTC") *

    * Unbiased Initial Beliefs ("UIB") *

    Generate X- and g-data (trajectories) for a single unbiased initial belief
    (i.e., initial belief g_0 = 0.5)
    for known evidence reliability and
    a specified constant trial cost 0 < c < 1,
    where bounds are computed by diffusion process direct optimization with c;

    plot the X-trajectory and the X-bounds together; and

    plot the g-trajectory and the g-bounds together.

   *** RECOMMENDATION:
      use M_ts >= 30,000 to make max experiment time long enough ***

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from tqdm import tqdm



# Use diffusion process direct optimization to estimate upper accumulated evidence bound X_H and lower accumulated bound X_L = -X_H:
def get_do_Xbounds(sig2, c_do_t): # Get upper bound on accumulated evidence sum statistic X

  """

  Parameters:

  - sig2:       constant noise variance

  - c_do_t:     cost function evaluated at time t

  """

  # Define the function to maximize:
  def objective_function(X):
    return -(1/(1+np.e**((-2/sig2)*X)) - c_do_t*X*np.tanh(X))  # negative of the function to be maximized

  # Define the bounds for the variable:
  bounds = None

  # Define the initial guess:
  initial_guess = 0

  # Use the minimize function with the negative objective function:
  result = minimize(objective_function, initial_guess, bounds=bounds)

  # Extract the optimal value:
  X_H = result.x[0]

  return X_H, -X_H



# Transform a direct optimization X-bound (either X_H or X_L) into the corresponding bound on belief g (either g_H or g_L, respectively):
def get_do_gbound(sig2, Xbound):
  return 1/(1+np.e**((-2/sig2)*Xbound))



# Use diffusion process direct optimization to estimate upper belief bound g_H and lower belief bound g_L:
def get_do_gbounds(sig2, c_do):
  X_H, X_L = get_do_Xbounds(sig2, c_do)
  return get_do_gbound(sig2, X_H), get_do_gbound(sig2, X_L)



# Generate the acc. evidence and belief stochastic process data:
def generate_data(z, g_0, sig2, dt, M_ts):

  """

  Parameters:

  - z:          the actual hidden state (z = -1 or z = 1) for data generation

  - g_0:        initial belief that z = 1; 0 < g_0 < 1; use g_0 = 0.5 for unbiased initial belief

  - sig2:       constant noise variance

  - dt:         constant trial duration

  - M_ts:       number of discrete times; empirically, M_ts >= 20,000 appears to be sufficient for dt >= 0.0001

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



# Plot KER trajectories for five example cost functions with unbiased initial beliefs:
def ker_tdtc_difproc__multi_cost_examples(z, sig2, dt, M_ts):

  """

  Parameters:

  - z:          the actual hidden state (z = -1 or z = 1) for data generation

  - sig2:       constant noise variance

  - dt:         constant trial duration

  - M_ts:       number of discrete times; empirically, M_ts >= 20,000 appears to be sufficient for dt >= 0.0001

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
    for n in range(M_ts):

      # Populate the X- and g-bound function rows:
      XHs[c_n, n], XLs[c_n, n] = get_do_Xbounds(sig2, cs[c_n, n])
      gHs[c_n, n] = get_do_gbound(sig2, XHs[c_n, n])
      gLs[c_n, n] = get_do_gbound(sig2, XLs[c_n, n])

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
def ker_tdtc_difproc__multi_bib_example__tanh(z, sig2, dt, M_ts, g_0_start, g_0_stop, M_gs):

  """

  Parameters:

  - z:          the actual hidden state (z = -1 or z = 1) for data generation

  - sig2:       constant noise variance

  - dt:         constant trial duration

  - M_ts:       number of discrete times; empirically, M_ts >= 20,000 appears to be sufficient for dt >= 0.0001

  - g_0_start:  initial belief that z = 1 for first experiment; 0 < g_0_start <= g_0_stop

  - g_0_stop:   initial belief that z = 1 for last experiment;  g_0_start <= g_0_stop < 1

  - M_gs:       number of initial beliefs to generate (number of experiments) between g_0_start and g_0_stop (inclusive)

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
  for n in range(M_ts):
    XHs[n], XLs[n] = get_do_Xbounds(sig2, cs[n])
    gHs[n] = get_do_gbound(sig2, XHs[n])
    gLs[n] = get_do_gbound(sig2, XLs[n])
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



# Plot KER trajectories using a specified constant cost 0 < c < 1 for a total of M_gs initial beliefs between g_0_start and g_0_stop:
def ker_ctc_difproc__multi_bib(z, sig2, dt, M_ts, c, g_0_start, g_0_stop, M_gs):

  """

  Parameters:

  - z:          the actual hidden state (z = -1 or z = 1) for data generation

  - sig2:       constant noise variance

  - dt:         constant trial duration

  - M_ts:       number of discrete times; empirically, M_ts >= 30,000 appears to be sufficient for dt >= 0.0001

  - 0 < c < 1:  constant trial cost

  - g_0_start:  initial belief that z = 1 for first experiment; 0 < g_0_start <= g_0_stop

  - g_0_stop:   initial belief that z = 1 for last experiment;  g_0_start <= g_0_stop < 1

  - M_gs:       number of initial beliefs to generate (number of experiments) between g_0_start and g_0_stop (inclusive)

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
  for n in range(M_ts):
    XHs[n], XLs[n] = get_do_Xbounds(sig2, cs[n])
    gHs[n] = get_do_gbound(sig2, XHs[n])
    gLs[n] = get_do_gbound(sig2, XLs[n])
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

  #
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
    plt.title('Accumulated Evidence ' + f'$X(t)$' + ' vs. Time ' + f'$t$' + ' for Unbiased Initial Belief ' + f'$g_0 = 0.5$' + '\nCost Function ' + cfunstr + ';\tParameters: ' + f'$σ_ε^2 = $' + str(sig2) + ', ' + f'$dt = $' + str(dt))
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
def ker_ctc_difproc__uib(z, sig2, dt, M_ts, c):

  """

  Parameters:

  - z:            the actual hidden state (z = -1 or z = 1) for data generation

  - sig2:         constant noise variance

  - dt:           constant trial duration

  - M_ts:         number of discrete times; empirically, M_ts >= 30,000 appears to be sufficient for dt >= 0.0001

  - 0 < c < 1:    constant trial cost

  """

  return ker_ctc_difproc__multi_bib(z, sig2, dt, M_ts, c, 0.5, 0.5, 1)





# Recommended Tests:
#ker_tdtc_difproc__multi_cost_examples(1, 1, 0.0001, 10000)

#ker_tdtc_difproc__multi_bib_example__tanh(1, 1, 0.0001, 20000, 0.01, 0.99, 100)
#ker_tdtc_difproc__multi_bib_example__tanh(1, 1, 0.0001, 20000, 0.0001, 0.9999, 10000)

#ker_ctc_difproc__multi_bib(1, 1, 0.0001, 20000, 0.2, 0.0001, 0.9999, 10000)
#ker_ctc_difproc__multi_bib(1, 1, 0.0001, 30000, 0.2, 0.01, 0.99, 100)

#ker_ctc_difproc__uib(1, 1, 0.0001, 30000, 0.2)
