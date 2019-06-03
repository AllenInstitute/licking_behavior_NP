import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
plt.ion() # makes non-blocking figures
import fit_tools

# Define which experiment id you want
experiment_id = 715887471

# Get the data
data = fit_tools.get_data(experiment_id, save_dir='./example_data')
licks = data['lick_timestamps']
running_timestamps = data['running_timestamps']
running_speed = data['running_speed']
rewards = np.round(data['reward_timestamps'],2)
rewardsdt = np.round(rewards*(1/dt))

# get start/stop time for session
start_time = 1
dt = 0.01 # 10msec timesteps
stop_time = int(np.round(running_timestamps[-1],2)*(1/dt))
licks = licks[licks < stop_time/100]
licks = np.round(licks,2)
licksdt = np.round(licks*(1/dt))
time_vec = np.arange(0,stop_time/100.0,dt)


### Make mean lick rate example
# simple model with only mean lick rate
nll, latent = fit_tools.licking_model([-.5], licksdt, stop_time, post_lick=False,include_running_speed=False)

# Wrapper function for optimization that only takes one input
def mean_wrapper_func(mean_lick_rate):
    #return mean_lick_model(mean_lick_rate,licksdt,stop_time)[0]
    return fit_tools.licking_model(mean_lick_rate, licksdt, stop_time, post_lick=False,include_running_speed=False)[0]

# optimize
inital_param = 0
res_mean = minimize(mean_wrapper_func, inital_param)

# We get a sensible result!
Average_probability_of_lick = np.exp(res_mean.x)[0]
sanity_check = len(licks)/(stop_time + 0.000001)

def wrapper(mean_lick_rate):
    return fit_tools.licking_model(mean_lick_rate, licksdt, stop_time, post_lick=False,include_running_speed=False)

res_mean = fit_tools.evaluate_model(res_mean,wrapper, licksdt, stop_time)
fit_tools.compare_model(res_mean.latent, time_vec, licks, stop_time)

### Make Post lick filter example
nll, latent = fit_tools.licking_model(np.concatenate(([-.5],np.zeros((10,)))), licksdt, stop_time, post_lick=True,include_running_speed=False)

# Wrapper function for optimization that only takes one input
def post_lick_wrapper_func(params):
    return fit_tools.licking_model(np.concatenate(([-.5],np.zeros((10,)))), licksdt, stop_time, post_lick=True,include_running_speed=False)[0]

# optimize
inital_param = np.concatenate(([-.5],np.zeros((10,)),np.zeros((5,))))
res_post_lick = minimize(post_lick_wrapper_func, inital_param)

def wrapper(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,include_running_speed=False)

res_post_lick = fit_tools.evaluate_model(res_post_lick,wrapper, licksdt, stop_time)
fit_tools.compare_model(res_post_lick.latent, time_vec, licks, stop_time)
fit_tools.build_filter(res_post_lick.x[1:], np.arange(dt,.21,dt), 0.025, plot_filters=True)





### Make running_speed example
nll, latent = fit_tools.licking_model(np.concatenate(([-.5],np.zeros((10,)),np.zeros((5,)))), licksdt, stop_time, post_lick=True,include_running_speed=True, num_running_speed_params=5, running_speed=running_speed)

# Wrapper function for optimization that only takes one input
def running_wrapper_func(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,include_running_speed=True, num_running_speed_params=5,running_speed=running_speed)[0]

# optimize
inital_param = np.concatenate(([-.5],np.zeros((10,)),np.zeros((5,))))
res_running = minimize(running_wrapper_func, inital_param)

def wrapper(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,include_running_speed=True, num_running_speed_params=5,running_speed=running_speed)

res_running = fit_tools.evaluate_model(res_running,wrapper, licksdt, stop_time)
fit_tools.compare_model(res_running.latent, time_vec, licks, stop_time)
fit_tools.build_filter(res_post_lick.x[1:11], np.arange(dt,.21,dt), 0.025, plot_filters=True,plot_nonlinear=True)
fit_tools.build_filter(res_post_lick.x[11:], np.arange(dt,.21,dt), 0.025, plot_filters=True,plot_nonlinear=True)

#### Make Reward Example
nll, latent = fit_tools.licking_model(np.concatenate(([-.5],np.zeros((20,)))), licksdt, stop_time, post_lick=False,include_running_speed=False, num_running_speed_params=5, running_speed=running_speed, include_reward=True, num_reward_params=20, reward_duration=4, reward_sigma=0.5, rewardsdt=rewardsdt)

def reward_wrapper_func(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_speed=False, include_reward=True, num_reward_params=20, reward_duration=4, reward_sigma=0.5,rewardsdt=rewardsdt)[0]

inital_param = np.concatenate(([-.5],np.ones((20,))))
res_reward = minimize(reward_wrapper_func, inital_param)

def wrapper(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_speed=False, include_reward=True, num_reward_params=20, reward_duration=4, reward_sigma=0.5,rewardsdt=rewardsdt)

res_reward = fit_tools.evaluate_model(res_reward, licksdt, stop_time)
fit_tools.compare_model(res_reward.latent, time_vec, licks, stop_time)
fit_tools.build_filter(res_post_lick.x[1:], np.arange(dt,4,dt), 0.5, plot_filters=True,plot_nonlinear=True)

















