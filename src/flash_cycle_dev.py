import psy_general_tools as pgt
import psy_timing_tools as pt
import matplotlib.pyplot as plt
import flash_cycle_tools as fct
import seaborn as sns
plt.ion()


# Licking wrt flash cycle
ids = pgt.get_active_ids()
dir = '/home/alex.piet/codebase/behavior/model_free/'

# Plot All Sessions Individually
fct.plot_all_sessions(ids,dir)

# Plot all Sessions together
fct.plot_sessions(ids,directory=dir+"all",return_counts=True)

# plot All sessions for each mouse
mice_ids = pgt.get_mice_ids()
fct.plot_all_mouse_sessions(mice_ids, dir)

# get dataframe of peakiness
df = fct.build_session_table(pgt.get_active_ids(),"/home/alex.piet/codebase/behavior/psy_fits_v9/")
plt.figure(); plt.plot(df.peakiness,df.hit_percentage,'ko');    plt.xlabel('PeakScore'); plt.ylabel('Hit Fraction')
plt.figure(); plt.plot(df.peakiness,df.hit_count,'ko');         plt.xlabel('PeakScore'); plt.ylabel('Hit Count')
plt.figure(); plt.plot(df.peakiness,df.licks,'ko');             plt.xlabel('PeakScore'); plt.ylabel('# Lick Bouts')
plt.figure(); plt.plot(df.peakiness,df.task_index,'ko');        plt.xlabel('PeakScore'); plt.ylabel('Timing/Task Index')
plt.figure(); plt.plot(df.peakiness,df.mean_dprime,'ko');       plt.xlabel('PeakScore'); plt.ylabel('mean dprime')

plt.figure(); plt.plot(df.task_index,df.hit_percentage,'ko');   plt.xlabel('Timing/Task Index'); plt.ylabel('Hit Fraction')
plt.figure(); plt.plot(df.task_index,df.hit_count,'ko');        plt.xlabel('Timing/Task Index'); plt.ylabel('Hit Count')
plt.figure(); plt.plot(df.task_index,df.licks,'ko');            plt.xlabel('Timing/Task Index'); plt.ylabel('# Lick Bouts')
plt.figure(); plt.plot(df.task_index,df.peakiness,'ko');        plt.xlabel('Timing/Task Index'); plt.ylabel('PeakScore')
plt.figure(); plt.plot(df.task_index,df.mean_dprime,'ko');      plt.xlabel('Timing/Task Index'); plt.ylabel('mean dprime')


# Dev below here
#####################################################################
# Look at the start of lick bouts relative to flash cycle
all_licks = []
change_licks = []
for id in pgt.get_active_ids():
    print(id)
    try:
        session = pgt.get_data(id)
        pm.annotate_licks(session)
        pm.annotate_bouts(session)
        pm.annotate_bout_start_time(session)
        x = session.stimulus_presentations[session.stimulus_presentations['bout_start']==True]
        rel_licks = (x.bout_start_time-x.start_time).values
        all_licks.append(rel_licks)
        x = session.stimulus_presentations[(session.stimulus_presentations['bout_start']==True) & (session.stimulus_presentations['change'] ==True)]
        rel_licks = (x.bout_start_time-x.start_time).values
        change_licks.append(rel_licks)
    except Exception as e:
        print(" crash "+str(e))

def plt_all_licks(all_licks,change_licks,bins):
    plt.figure()
    plt.hist(np.concatenate(all_licks),bins=bins,color='gray',label='All Flashes')
    plt.hist(np.concatenate(change_licks),bins=bins,color='black',label='Change Flashes')
    plt.ylabel('Count',fontsize=12)
    plt.xlabel('Time since last flash onset',fontsize=12)
    plt.xlim([0, 0.75])
    plt.legend()
    plt.tight_layout()

plt_all_licks(all_licks,change_licks,45)



