# %%
#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# =============================================================
# 12 - Tracking linear Levy transition models with time-varying skew
# =============================================================
# In line with the tutorial examples of the Kalman and particle 
# filters in Stone Soup, a simplified single-target tracking 
# example without clutter is provided here to demonstrate the 
# use of linear transition models driven by non-Gaussian Levy 
# noise, as well as how to perform inference tasks on this class 
# of models using the Marginalized Particle Filter (MPF). 
# 
# This tutorial extends the previous tutorial by introducing a 
# secondary driver process allowing the mean of the state 
# transition distribution (mu_W) to vary over time.

# %%
# Consider the scenario where the target evolves according to the Langevin model, driven by a normal sigma-mean mixture with the mixing distribution being the $\alpha$-stable distribution.
import numpy as np
from datetime import datetime, timedelta
# The state of the target can be represented as 2D Cartesian coordinates, $\left[x, \dot x, y, \dot y\right]^{\top}$, modelling both its position and velocity. A simple truth path is created with a sampling rate of 1 Hz.

# %%
from stonesoup.models.transition.linear import ConstantVelocity, RandomWalk
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.base_driver import NoiseCase
from stonesoup.models.driver import AlphaStableNSMDriver 
from stonesoup.models.transition.levy_linear import LevyLangevin, CombinedLinearLevyTransitionModel

# And the clock starts
start_time = datetime.now().replace(microsecond=0)

# %% [markdown]
# 
# The `LevyLangevin` class creates a one-dimensional Langevin model, driven by the $\alpha$-stable NSM mixture process defined in the `AlphaStableNSMDriver` class.
# 
# \begin{equation}
# d \dot{x}(t)=-\theta \dot{x}(t) d t+d W(t), \quad \theta>0
# \end{equation}
# 
# where $\theta$ is the damping factor and $W(t)$ is the non-Gaussian driving process. 
# 
# In the Marginalised
# 
# The noise samples $\mathbf{w}_n$ are drawn from the $\alpha$-stable distribution parameterized by the $\alpha$-stable law, $S_{\alpha}(\sigma, \beta, \mu)$.
# 
# The input parameters to `AlphaStableNSMDriver` class are the stability index $\alpha$, expected jumps per unit time $c$, conditional Gaussian mean $\mu_W$ & variance $\sigma_W^2$, and the type of residuals used for the truncated shot-noise representation, specified by `noise_case`. 
# 
# Without diving into technical details, the scaling factor $\sigma$, skewness parameter $\beta$ and location $\mu$, in the $\alpha$-stable law is a function of the conditional Gaussian parameters $\mu_W, \sigma_W^2$. In general, set $\mu_W=0$ for a symmetric target distribution $\beta=0$, or $\mu_W \neq 0$ to model biased trajectories otherwise. In addition, the size of the resulting trajectories (and jumps) can be adjusted by varying $\sigma_W^2$.
# 
# The available noise cases are:
# 
# 1. No residuals, `NoiseCase.TRUNCATED`, least expensive but drawn noise samples deviate further from target distribution.
# 2. `NoiseCase.GAUSSIAN_APPROX`, the most expensive but drawn noise samples closest target distribution.
# 3. `PartialNoiseCase.GAUSSIAN_APPROX`, a compromise between both cases (1) and (2).
# 
# 
# For interested readers, refer to [1, 2] for more details.
# 
# We must first initialise a mu_driver, which will determine how the x- and y- process means will change as time progresses. 
# 
# Here we have chosen both to be Gaussian random walks, RW_mu_driver= RandomWalk(noise_diff_coeff=q) 
# 
# with the same noise diffusion coefficient parameter, q=2.5e-7.
# 
# We must also change the initial mu_W for each driver, so initial_mu_W_x = +0.001 and initial_mu_W_y= -0.001.
# 
# This is then input into both $\alpha$-stable drivers with the default parameters `mu_W=initial_mu_W_x, sigma_W2=0.0025, alpha=1.9, noise_case=NoiseCase.GAUSSIAN_APPROX(), c=10, mu_W_transition_model=mu_driver`.
# 
# Then, the driver instance is injected into the Langevin model for every coordinate axes (i.e., x and y) during initialisation with parameter `damping_coeff=0.05`.
# 
# Finally, the `CombinedLinearLevyTransitionModel` class takes a set of 1-D models and combines them into a linear transition model of arbitrary dimension, $D$, (in this case, $D=2$).
# 
# 
# <!-- and  in the $\alpha$-stable law is a function of the conditional Gaussian mean $\mu_W(t)$, which now evolves over time.
# 
# where $\beta=\begin{cases} 1, \quad \mu_W \neq 0 \\ 0, \quad \text{otherwise} \end{cases}$ with $\beta=0$ being the a symmetric stable distribution.
# 
# $\sigma=\frac{\mathbb{E}|w|^\alpha \Gamma(2-\alpha) \cos(\pi \alpha / 2))}{1- \alpha}$ represent the scale parameter and $\beta=$ controlling the skewness of the stable distribution. -->
# 

# %%
seed = 1 # Random seem for reproducibility

#time-varying mu_W parameters
q=5e-8
mu_model='RW'
if mu_model=='RW':
    initial_mu_W_x= +0.01
    initial_mu_W_y= -0.01
    RW_mu_driver= RandomWalk(noise_diff_coeff=q) #1D GRW
    mu_driver = RW_mu_driver
elif mu_model=='CV':
    q/=10
    initial_mu_W_x= np.array([[0.00],[0]])
    initial_mu_W_y= np.array([[0.00],[0]])
    CV_mu_driver=ConstantVelocity(noise_diff_coeff=q)
    mu_driver =CV_mu_driver

# Driving process parameters and drivers
sigma_W2 = 0.0025
alpha = 1.9
c=10
noise_case=NoiseCase.GAUSSIAN_APPROX
driver_x = AlphaStableNSMDriver(mu_W=initial_mu_W_x, sigma_W2=sigma_W2, seed=seed, c=c, alpha=alpha, mu_W_transition_model=mu_driver ,mu_W_state=True)
driver_y = AlphaStableNSMDriver(mu_W=initial_mu_W_y, sigma_W2=sigma_W2, seed=seed, c=c, alpha=alpha, mu_W_transition_model=mu_driver ,mu_W_state=True)

# Model parameters
theta=0.05
langevin_x = LevyLangevin(driver=driver_x, damping_coeff=theta)
langevin_y = LevyLangevin(driver=driver_y, damping_coeff=theta)
transition_model = CombinedLinearLevyTransitionModel([langevin_x, langevin_y])

# %%
print(transition_model.mu_W)
print(transition_model.sigma_W2)

# %% [markdown]
# 
# The ground truth is initialised from (0,0).
# 
# The shape of the stored mu_W is num_samples x m x num_drivers.
# 
# num_samples is the number of particles (1 in this groundtruth generation case) and m is the dimension of the mean. 
# 
# num_drivers= 2 here, as we have one for x and y respectively.
# 
# In this example we have used a univariate gaussian random walk, but should we wish for mu_W to evolve according to e.g. the Constant Velocity model, this would be possible.
# 
# In the multivariate mu_W case, the first component (i.e. the position) of the mu_W vector is input as the processes' driving parameter. Our initial_mu inputs would also need to be vectors.
# 
# We here store the mu_W as a 2D groundtruth. 
# 
# This is not done automatically/within the driver framework to maintain simplicity of code and avoid conflicts with the default float input of mu_W

# %%
timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])
if mu_model =='RW':
    initial_mu_W=[initial_mu_W_x,initial_mu_W_y]
elif mu_model== 'CV':
    initial_mu_W= list(initial_mu_W_x) + list(initial_mu_W_y)
mu_W_groundtruth = GroundTruthPath([GroundTruthState(initial_mu_W, timestamp=timesteps[0])])

num_steps = 200
for k in range(1, num_steps + 1):
    timesteps.append(start_time+timedelta(seconds=k))  # add next timestep to list of timesteps
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))
    # Extract the mu_W values (0,0)th index as first two dimensions have no shape in this 1D & groundtruth case
    x_mu = transition_model.mu_W[0,0,0] 
    y_mu = transition_model.mu_W[0,0,1]
    mu_W_groundtruth.append(GroundTruthState([x_mu, y_mu], timestamp=timesteps[k]))

# %% [markdown]
# The simulated ground truth path can be plotted using the in-built plotting classes in Stone Soup.
# 
# In addition to the ground truth, Stone Soup plotting tools allow measurements and predicted tracks (see later) to be plotted and synced together consistently.
# 
# An animated plotter that uses Plotly graph objects can be accessed via the `AnimatedPlotterly` class from Stone Soup.
# 
# Note that the animated plotter requires a list of timesteps as an input, and that `tail_length`
# is set to 0.3. This means that each data point will be on display for 30% of the total
# 
# If a static plotter is preferred, the `Plotterly` class can be used instead
# 
# 

# %%
# from stonesoup.plotter import AnimatedPlotterly
# plotter = AnimatedPlotterly(timesteps, tail_length=1.0, width=600, height=600)
from plotly import colors
from stonesoup.plotter import Dimension, Plotterly
from plotly.subplots import make_subplots

# %% [markdown]
# 1D plotting capabilities and smoother uncertainties
# 
# Plotting a single 1D component against time is possible in Stone Soup by setting the dimension parameter equal to Dimension.ONE.
# 
# These plots may provide a clearer insight into how the underlying process evolves over time.  
# 
# We will look at a 1D plot of the x-position and x-velocity components, using mapping=[0] and mapping=[2] respectively. 
# To look at y-components, change the mapping to [1] and [3]. The mapping for mu, which only has x and y, are [0] and [1] respectively.
# 
# For clarity, we will plot mu_W on a secondary axis. 

# %%
#First initialise the plots.

colorway=colors.qualitative.Plotly[1:]
position_plotter= Plotterly(autosize=False, width=600,height=600,dimension=Dimension.ONE)
position_plotter.fig = make_subplots(specs=[[{"secondary_y": True}]])
position_plotter.fig.update_yaxes(secondary_y=False,title_text="Object Position")
position_plotter.fig.update_yaxes(secondary_y=True,title_text="mu_W")
position_plotter.fig.update_xaxes(title_text="Time")
position_plotter.fig.add_scatter(y=np.zeros_like(truth), x=timesteps, name=f'mu_W, = 0',secondary_y=True,line=dict(color="green",dash='dash'))
position_plotter.fig.layout.colorway=colorway

position_plotter.plot_ground_truths(truth, [0])
position_plotter.plot_ground_truths(mu_W_groundtruth, [0], secondary_y=True, label='mu_W_truth')
position_plotter.fig

# %%
velocity_plotter= Plotterly(autosize=False, width=600,height=600,dimension=Dimension.ONE)
velocity_plotter.fig = make_subplots(specs=[[{"secondary_y": True}]])
velocity_plotter.fig.update_yaxes(secondary_y=False,title_text="Object Velocity")
velocity_plotter.fig.update_yaxes(secondary_y=True,title_text="mu_W")
velocity_plotter.fig.update_xaxes(title_text="Time")
velocity_plotter.fig.add_scatter(y=np.zeros_like(truth), x=timesteps, name=f'mu_W, = 0',secondary_y=True,line=dict(color="green",dash='dash'))
velocity_plotter.fig.layout.colorway=colorway

velocity_plotter.plot_ground_truths(truth, [1])
velocity_plotter.plot_ground_truths(mu_W_groundtruth, [0], secondary_y=True, label='mu_W_truth')
velocity_plotter.fig

# %% [markdown]
# ## Simulate measurements
# 
# Assume a 'linear' sensor which detects the
# position, but not velocity, of a target, such that
# $\mathbf{z}_k = H_k \mathbf{x}_k + \boldsymbol{\nu}_k$,
# $\boldsymbol{\nu}_k \sim \mathcal{N}(0,R)$, with
# 
# \begin{align}H_k &= \begin{bmatrix}
#                     1 & 0 & 0 & 0\\
#                     0  & 0 & 1 & 0\\
#                       \end{bmatrix}\\
#           R &= \begin{bmatrix}
#                   10 & 0\\
#                     0 & 10\\
#                \end{bmatrix} \omega\end{align}
# 
# where $\omega$ is set to 10 initially

# %%
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian

# %% [markdown]
# 
# The linear Gaussian measurement model is set up by indicating the number of dimensions in the
# state vector and the dimensions that are measured (so specifying $H_k$) and the noise
# covariance matrix $R$.

# %%
measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[10, 0],  # Covariance matrix for Gaussian PDF
                          [0, 10]])
    )

# %% [markdown]
# The measurements can now be generated and plotted accordingly.

# %%
measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement,
                                  timestamp=state.timestamp,
                                  measurement_model=measurement_model))

# %%
position_plotter.plot_measurements(measurements, [0], marker=dict(symbol="x"))
position_plotter.fig

# %% [markdown]
# ## Marginalised Particle Filtering
# 
# To start we create a prior estimate. This now includes a prior over mu_x and mu_y
# 
# We first initialise this mu_W prior. 
# 
# Our mu_prior is an array, here sampled from a normal distribution.
# 
# Currently we avoid attaching a specific State type to the prior, again for coherency with the default float value of mu_W
# which allows mu_W to be updated by any type of transition model in a more modular fashion. 
# 
# The code will extract the state_vector array from mu_W (with covar/uncertainty not mattering given we don't explicitly infer over mu_W)
# and returns an numpy array as mentioned earlier. As shown in the groundtruth initialisation, one can also simply input an array, State object is not required.
# 
# We now initialise a new transition model for 2 reasons. 
# 
# Firstly, since the mean evolves over time, we want our initial mu_W for tracking to be our chosen prior,
# rather than the final value of our groundtruth mu. 
# 
# Secondly, we set 'mu_W_state'=None. This essentially tells the driver that we do not want to vary mu_W within the (in this case 1 second) intervals. 
# The default is to recalculate mu_W(t) at each jump point, which can occur many times between the sub intervals, giving us a more granular and accurate simulation of the process.
# 
# However, since each particle will have a different mu_W, and different jump times within the sub-interval due to the differing latents, 
# we have irregular time intervals over which to update each mu_W, and would not be able to vectorise the transition calculations. 
# 
# This greatly slows down the process, so for tracking purposes, we advise setting mu_W_state (which is the array of inter-interval mu_W values) = None
# so that mu_W updates can be vectorised, and tracking with many particles is computationally viable. 
# 
# Setting this parameter =None automatically stops it from being updated between intervals, and vectorises calculations.
# 
# This updated transition model is then fed into the marginalised filtering components
# 

# %%
from stonesoup.types.array import StateVectors
from scipy.stats import multivariate_normal
from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import MarginalisedParticleState

number_particles = 200

# %%
# re-initialise mu_drivers for tracking, before re-assigning transition model
# Sample from the mu_W prior Gaussian distribution, input MxN 
if mu_model=='RW':
    mu_prior_x = np.atleast_2d(multivariate_normal.rvs(initial_mu_W_x,
                                np.diag([q]),
                                size=number_particles))
    mu_prior_y = np.atleast_2d(multivariate_normal.rvs(initial_mu_W_y,
                                np.diag([q]),
                                size=number_particles))
    mu_driver = RW_mu_driver #1D GRW
elif mu_model=='CV':
    mu_prior_x = multivariate_normal.rvs(initial_mu_W_x.flatten(),
                                np.diag([q,0]),
                                size=number_particles).T
    mu_prior_y = multivariate_normal.rvs(initial_mu_W_y.flatten(),
                                np.diag([q,0]),
                                size=number_particles).T
    mu_driver= CV_mu_driver

tracking_driver_x = AlphaStableNSMDriver(mu_W=mu_prior_x, sigma_W2=sigma_W2, seed=seed, c=c, alpha=alpha, mu_W_transition_model=mu_driver,mu_W_state=None)
tracking_driver_y = AlphaStableNSMDriver(mu_W=mu_prior_y, sigma_W2=sigma_W2, seed=seed, c=c, alpha=alpha, mu_W_transition_model=mu_driver,mu_W_state=None)
langevin_x = LevyLangevin(driver=tracking_driver_x, damping_coeff=theta)
langevin_y = LevyLangevin(driver=tracking_driver_y, damping_coeff=theta)
transition_model = CombinedLinearLevyTransitionModel([langevin_x, langevin_y])


# %% [markdown]
# The `MarginalisedParticlePredictor` and `MarginalisedParticleUpdater` classes correspond to the predict and update steps
# respectively.
# Both require a `TransitionModel` and a `MeasurementModel` instance respectively.
# To avoid degenerate samples, the `SystematicResampler` is used which is passed to the updater.
# More resamplers that are included in Stone Soup are covered in the
# [Resampler Tutorial](https://stonesoup.readthedocs.io/en/latest/auto_tutorials/sampling/ResamplingTutorial.html#sphx-glr-auto-tutorials-sampling-resamplingtutorial-py).

# %%

from stonesoup.predictor.particle import MarginalisedParticlePredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.updater.particle import MarginalisedParticleUpdater

predictor = MarginalisedParticlePredictor(transition_model=transition_model)
resampler = SystematicResampler()
updater = MarginalisedParticleUpdater(measurement_model, resampler)


# The object prior is a `MarginalisedParticleState` which describes the state as a distribution of particles.
# 
# The mean priors are randomly sampled from the standard normal distribution.
# 
# The covariance priors is initialised with a scalar multiple of the identity matrix .
#

# Sample from the object's prior Gaussian distribution
states = multivariate_normal.rvs(np.array([0, 1, 0, 1]),
                                  np.diag([1., 1., 1., 1.]),
                                  size=number_particles)
covars = np.stack([np.diag([10., 0.2, 10., 0.2]) for i in range(number_particles)], axis=2) # (M, M, N)

# Create prior particle state.
prior = MarginalisedParticleState(
    state_vector=StateVectors(states.T),
    covariance=covars,
    weight=np.array([Probability(1/number_particles)]*number_particles),
                      timestamp=start_time-timedelta(seconds=1))

# %% [markdown]
# We now run the predict and update steps, propagating the collection of particles and resampling at each step
# 

# %%
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
mu_W_estimate = Track()

for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]
    
    x_mu_estimate= transition_model.mu_W[0,0,0]
    y_mu_estimate= transition_model.mu_W[0,0,1]
    mu_W_state= [x_mu_estimate,y_mu_estimate] # [:,0,:] is n (x 1) x num_drivers: we extract the first component of both mu_x and mu_y, for all n particles
    mu_W_estimate.append(MarginalisedParticleState(
        state_vector= mu_W_state, 
        covariance=np.zeros_like(covars),
        weight=np.array([Probability(1/number_particles)]*number_particles),
        timestamp=measurement.timestamp)) # currently covariance isn't stored to save space and simplify code. This is a dummy covariance.


# %%
print(mu_W_estimate[-1].mean)

# %%
#currently manually inputting track colours, believe is issue with plotter colorway selection
position_plotter.plot_tracks(track,[0], label='filtered',line=dict(color='#00CC96'))
position_plotter.plot_tracks(mu_W_estimate,[0],secondary_y=True,label='mu_W_estimate',line=dict(color='#B6E880'))
position_plotter.fig

# %%
velocity_plotter.plot_tracks(track,[1],label='filtered', line=dict(color='#00CC96'))
velocity_plotter.plot_tracks(mu_W_estimate,[0],secondary_y=True,label='mu_W_estimate',line=dict(color='#B6E880'))
velocity_plotter.fig

# %% [markdown]
# ## Tutorial References
# [1] Lemke, Tatjana, and Simon J. Godsill, 'Inference for models with asymmetric α -stable noise processes', in Siem Jan Koopman, and Neil Shephard (eds), Unobserved Components and Time Series Econometrics (Oxford, 2015; online edn, Oxford Academic, 21 Jan. 2016)
# 
# [2] S. Godsill, M. Riabiz, and I. Kontoyiannis, “The L ́evy state space model,” in 2019 53rd Asilomar Conference on Signals, Systems, and Computers, 2019, pp. 487–494.
# 
# [3] Z. Liu, Z. Tiller, and S. Godsill, “Inference for Non-Gaussian Dynamical Models
# with Time-varying Skew,” in 2024 27th International Conference on Information
# Fusion (FUSION). Venice, Italy: IEEE, Jul. 2024, pp. 1–8.
# 
# 


