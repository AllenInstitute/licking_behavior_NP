import numpy as np
import pandas as pd
import psy_tools as ps
import psy_general_tools as pgt
import psy_training_tools as ptt
import matplotlib.pyplot as plt
import psy_output_tools as po
plt.ion()

# Train Summary is a dataframe with model fit information
train_summary = po.get_training_summary_table(version)
ophys_summary = po.get_ophys_summary_table(version)
mouse_summary = po.get_mouse_summary_table(version)
full_table = ptt.get_full_behavior_table(train_summary, ophys_summary)
full_table_no_lapse = ptt.get_full_behavior_table(train_summary, ophys_summary,filter_lapsed=True)

# Plot Averages by training stage 
ptt.plot_average_by_stage(full_table, metric='strategy_dropout_index')
ptt.plot_all_averages_by_stage(full_table,version)
ptt.plot_all_averages_by_stage(full_table,version,plot_mouse_groups=True)
ptt.plot_all_averages_by_stage(full_table,version,plot_each_mouse=True)
ptt.plot_all_averages_by_stage(full_table,version,plot_cre=True)

# Plot Average by Training session
ptt.plot_all_averages_by_day(full_table, mouse_summary, version)
ptt.plot_all_averages_by_day_mouse_groups(full_table, mouse_summary, version)
ptt.plot_all_averages_by_day_cre(full_table, mouse_summary, version)

# SAC plot
training = po.get_training_summary_table(20)
skip = ['OPHYS_1','OPHYS_3','OPHYS_4','OPHYS_6','OPHYS_0_habituation','TRAINING_5_lapsed','TRAINING_4_lapsed']
ptt.plot_average_by_stage(training, metric='num_hits',filetype='_sac.png',version=20,alpha=1,SAC=True, metric_name='# Hits / Session',skip=skip)