import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
plt.ion() # makes non-blocking figures
import fit_tools

# Define which experiment id you want
experiment_id = 715887471

# Get the data
dt = 0.01 # 10msec timesteps
data = fit_tools.get_data(experiment_id, save_dir='./example_data')
licks = data['lick_timestamps']
running_timestamps = data['running_timestamps']
running_speed = data['running_speed']
rewards = np.round(data['reward_timestamps'],2)
flashes=np.round(data['stim_on_timestamps'],2)
dt = 0.01
rewardsdt = np.round(rewards*(1/dt))
flashesdt = np.round(flashes*(1/dt))

stims = data['stim_id']
stims[np.array(stims) == 8 ] = 100
diffs = np.diff(stims)
diffs[(diffs > 50) | (diffs < -50 )] = 0
diffs[ np.abs(diffs) > 0] = 1
diffs = np.concatenate([[0], diffs])

change_flashes = flashes[diffs == 1]
change_flashesdt = np.round(change_flashes*(1/dt))


# get start/stop time for session
start_time = 1
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
Average_licks_per_time_bin = np.exp(res_mean.x)[0]
Average_licks_per_second = Average_licks_per_time_bin*(1/dt)
sanity_check = len(licks)/(stop_time*dt + 0.000001)

def wrapper(mean_lick_rate):
    return fit_tools.licking_model(mean_lick_rate, licksdt, stop_time, post_lick=False,include_running_speed=False)

res_mean = fit_tools.evaluate_model(res_mean,wrapper, licksdt, stop_time)
fit_tools.compare_model(res_mean.latent, time_vec, licks, stop_time)




### Make Post lick filter example
nll, latent = fit_tools.licking_model(np.concatenate(([-.5],np.zeros((10,)))), licksdt, stop_time, post_lick=True,include_running_speed=False)

# Wrapper function for optimization that only takes one input
def post_lick_wrapper_func(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,include_running_speed=False)[0]

# optimize
inital_param = np.concatenate(([-.5],np.zeros((10,))))
res_post_lick = minimize(post_lick_wrapper_func, inital_param)

def wrapper(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,include_running_speed=False)

res_post_lick = fit_tools.evaluate_model(res_post_lick,wrapper, licksdt, stop_time)
fit_tools.compare_model(res_post_lick.latent, time_vec, licks, stop_time)
fit_tools.build_filter(res_post_lick.x[1:], np.arange(dt,.21,dt), 0.025, plot_filters=True)


### Make running_speed example
n_running_bins = 6
nll, latent = fit_tools.licking_model(np.concatenate(([-.5],np.zeros((10,)),np.zeros((n_running_bins,)))), licksdt, stop_time, post_lick=True,include_running_speed=True, num_running_speed_params=n_running_bins, running_speed=running_speed)

# Wrapper function for optimization that only takes one input
def running_wrapper_func(params): return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,include_running_speed=True, num_running_speed_params=n_running_bins,running_speed=running_speed)[0]

# optimize
inital_param = np.concatenate(([-.5],np.zeros((10,)),np.zeros((n_running_bins,))))
res_running = minimize(running_wrapper_func, inital_param)

def wrapper(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,include_running_speed=True, num_running_speed_params=n_running_bins,running_speed=running_speed)

res_running = fit_tools.evaluate_model(res_running,wrapper, licksdt, stop_time)
fit_tools.compare_model(res_running.latent, time_vec, licks, stop_time, running_speed)
fit_tools.build_filter(res_running.x[1:11], np.arange(dt,.21,dt), 0.025, plot_filters=True,plot_nonlinear=True)
#  fit_tools.build_filter(res_running.x[11:], np.arange(dt,.21,dt), 0.025, plot_filters=True,plot_nonlinear=True)


# The combined post-lick / running speed model looks strange, like we have a running speed align. issue. 
### Running speed with no post-lick filter included
n_bins = 6
nll, latent = fit_tools.licking_model(np.concatenate(([-.5],np.zeros((n_bins,)))), licksdt, stop_time, post_lick=False,include_running_speed=True, num_running_speed_params=n_bins, running_speed=running_speed)

# Wrapper function for optimization that only takes one input
def running_wrapper_func(params): return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_speed=True, num_running_speed_params=n_bins,running_speed=running_speed)[0]

# optimize
inital_param = np.concatenate(([-.5],np.zeros((n_bins,))))
res_running = minimize(running_wrapper_func, inital_param)

def wrapper(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_speed=True, num_running_speed_params=n_bins,running_speed=running_speed)

res_running = fit_tools.evaluate_model(res_running,wrapper, licksdt, stop_time)
fit_tools.compare_model(res_running.latent, time_vec, licks, stop_time, running_speed)


#fit_tools.build_filter(, np.arange(dt,.20,dt), 0.025, plot_filters=True,plot_nonlinear=True)

plt.plot(res_running.x[1:21], 'k-')


#### Make Reward Example
def reward_wrapper_full(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_speed=False, include_reward=True, num_reward_params=40, reward_duration=4, reward_sigma=0.1,rewardsdt=rewardsdt)

def reward_wrapper(params):
    return reward_wrapper_full(params)[0]

inital_param = np.concatenate(([-.5],np.ones((40,))))
res_reward = minimize(reward_wrapper, inital_param)

res_reward = fit_tools.evaluate_model(res_reward,reward_wrapper_full, licksdt, stop_time)
fit_tools.compare_model(res_reward.latent, time_vec, licks, stop_time)
x=fit_tools.build_filter(res_reward.x[1:], np.arange(dt,4,dt), 0.1, plot_filters=True,plot_nonlinear=True)

ress =[]
durs = [3,3.5,4,4.5,5,5.5,6,6.5]
num_params = (np.array(durs)*10).astype(int)

for i in range(0, len(durs)):
    def reward_wrapper_full(params):
        return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_speed=False, include_reward=True, num_reward_params=num_params[i], reward_duration=durs[i], reward_sigma=0.1,rewardsdt=rewardsdt)
    def reward_wrapper(params):
        return reward_wrapper_full(params)[0]
    print(str(i))
    if i < 3:
        inital_param = res_reward.x[0:(num_params[i]+1)]
    else:
        inital_param = np.concatenate([res_reward.x, np.zeros((num_params[i]-40,))])
    res = minimize(reward_wrapper, inital_param)
    res = fit_tools.evaluate_model(res,reward_wrapper_full, licksdt, stop_time)
    ress.append(res)





#### Make Lick/Running/Reward Combined Example
def reward_wrapper_full(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,include_running_speed=True, num_running_speed_params=6, running_speed=running_speed, include_reward=True, num_reward_params=40, reward_duration=4, reward_sigma=0.1,rewardsdt=rewardsdt)

def reward_wrapper(params):
    return reward_wrapper_full(params)[0]

inital_param = np.concatenate([[-.5], np.zeros(10), np.zeros(6), np.zeros(40)])
res_reward = minimize(reward_wrapper, inital_param)

res_reward = fit_tools.evaluate_model(res_reward,reward_wrapper_full, licksdt, stop_time)
fit_tools.compare_model(res_reward.latent, time_vec, licks, stop_time, running_speed=running_speed)
x=fit_tools.build_filter(res_reward.x[1:], np.arange(dt,4,dt), 0.1, plot_filters=True,plot_nonlinear=True)


#### Make Flash Example
def flash_wrapper_full(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,include_running_speed=False, include_reward=False, include_flashes=True,flashesdt=flashesdt,num_flash_params=15,flash_sigma=0.025,flash_duration=.7)

def flash_wrapper(params):
    return flash_wrapper_full(params)[0]

inital_param = np.concatenate(([-.5],np.zeros((25,))))
res_flash = minimize(flash_wrapper, inital_param)

res_flash = fit_tools.evaluate_model(res_flash,flash_wrapper_full, licksdt, stop_time)
fit_tools.compare_model(res_flash.latent, time_vec, licks, stop_time)
x = fit_tools.build_filter(res_flash.x[1:11], np.arange(dt,.21,dt), 0.025, plot_filters=True,plot_nonlinear=True)
x = fit_tools.build_filter(res_flash.x[11:], np.arange(dt,.7,dt), 0.025, plot_filters=True,plot_nonlinear=True)


def change_flash_wrapper_full(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,include_running_speed=False, include_reward=False, include_flashes=False, include_change_flashes=True,change_flashesdt=change_flashesdt,num_change_flash_params=15,change_flash_sigma=0.025,change_flash_duration=.7)

def change_flash_wrapper(params):
    return change_flash_wrapper_full(params)[0]

inital_param = np.concatenate(([-.5],np.zeros((25,))))
res_flash = minimize(change_flash_wrapper, inital_param)

res_flash = fit_tools.evaluate_model(res_flash,change_flash_wrapper_full, licksdt, stop_time)
fit_tools.compare_model(res_flash.latent, time_vec, licks, stop_time)
x = fit_tools.build_filter(res_flash.x[1:11], np.arange(dt,.21,dt), 0.025, plot_filters=True,plot_nonlinear=True)
x = fit_tools.build_filter(res_flash.x[11:], np.arange(dt,.7,dt), 0.025, plot_filters=True,plot_nonlinear=True)















