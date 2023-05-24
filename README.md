# licking_behavior
Analysis of mouse licking behavior during visually guided behavior. Primarily, this repo develops a time-varying logistic regression model that learns the probability of licking on a flash by flash basis, using weights that vary over time by following random walk priors. 

This repository is specific to the Visual Behavior Neuropixels dataset. If you want to look at the Visual Behavior Optical Physiology dataset use github.com/alexpiet/licking_behavior. The model works the same but the data is loaded and processed slightly differently due to the VBN sessions being structured differently. 

## Installation
> git clone https://github.com/AllenInstitute/licking_behavior_NP  
> pip install -e . #run in directory with setup.py  

## Time varying regression model

The model predicts the probability of the mouse starting a licking bout on each image presentation. Its described as the sum of several time-varying strategies. 

- Bias, is a strategy that wants to lick on every image
- Visual/Task0, is a strategy that only wants to lick on the image-changes
- Timing1D, is a strategy that wants to lick every 4-5 images after the end of the last licking bout
- Omission0, is a strategy that wants to lick on every omission
- Omission1, is a strategy that wants to lick on the image after every omission

### Fitting the time varying regression model
> import licking_behavior_NP.psy_tools as ps  
> for bsid in behavior_session_ids:  
>    ps.process_session(bsid)  

## Model outputs
The key output dataframes are:

- summary_df, each behavioral session is a row, and columns contain model metrics and other behavioral metrics
- change_df, each image change across all behavioral sessions is a row
- licks_df, each lick across all behavioral sessions is a row
- bouts_df, each licking bout across all behavioral sessions is a row

### Diagram of information flow
![code_diagram](https://user-images.githubusercontent.com/7605170/175404261-4565ab0a-2c82-4215-9840-dffb2b736883.png)

### summary_df
> import licking_behavior_NP.psy_output_tools as po  
> summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)  

The columns of summary_df are: 
- session_roc (float) The cross validated area under the ROC curve for this session  
- strategy_dropout_index (float) The strategy index for this session  
- visual_strategy_session (bool) Whether the visual strategy was dominant in this session  
- strategy_labels (str) Either 'visual' or 'timing'
- prior_bias (float) the smoothing prior for the licking bias strategy  
- prior_omissions (float) the smoothing prior for the licking omissions strategy  
- prior_omissions1 (float) the smoothing prior for the post omissions strategy  
- prior_task0 (float) the smoothing prior for the visual strategy  
- prior_timing1D (float) the smoothing prior for the timing strategy  
- dropout_bias (float) the dropout index for the licking bias strategy  
- dropout_omissions (float) the dropout index for the licking omissions strategy  
- dropout_omissions1 (float) the dropout index for the post omissions strategy  
- dropout_task0 (float) the dropout index for the visual strategy  
- dropout_timing1D (float) the dropout index for the timing strategy  
- dropout_cv_bias (float) the cross validated dropout index for the licking bias strategy  
- dropout_cv_omissions (float) the cross validated dropout index for the licking omissions strategy  
- dropout_cv_omissions1 (float) the cross validated dropout index for the post omissions strategy  
- dropout_cv_task0 (float) the cross validated dropout index for the visual strategy  
- dropout_cv_timing1D (float) the cross validated dropout index for the timing strategy  
- avg_weight_bias (float) the average weight for the licking bias strategy  
- avg_weight_omissions (float) the average weight for the licking omissions strategy  
- avg_weight_omissions1 (float) the average weight for the post omissions strategy  
- avg_weight_task0 (float) the average weight for the visual strategy  
- avg_weight_timing1D (float) the average weight for the timing strategy  
- num_hits (float) number of hits in this session  
- num_miss (float) number of misses in this session  
- num_omission_licks (float) number of licking bouts that started during an omission  
- num_post_omission_licks (float) number of licking bouts that started on the image after an omission  
- num_late_task_licks (float) number of licking bouts that started on the image after an image change
- num_changes (float) number of image changes  
- num_omissions (float) number of omissions  
- num_image_false_alarm (float) number of licking bouts that started on a non-change image
- num_image_correct_reject (float) number of non-change images that did not contain the start of a licking bout  
- num_lick_bouts (float) number of licking bouts
- lick_fraction (float) the percentage of images with the start of a licking bout
- omission_lick_fraction (float) the percentage of omissions with the start of a licking bout
- post_omission_lick_fraction (float) the percentage of post-omission images with the start of a licking bout 
- lick_hit_fraction (float) the percentage of licking bouts that resulted in a reward  
- trial_hit_fraction (float) the percentage of image changes that were rewarded  
- strategy_weight_index (float) the difference in average strategy weights between visual and timing strategies  
- fraction_engaged (float) the percentage of the session when the mouse was engaged

Additionally, there are columns that are split by whether the mouse was engaged or disengaged. These should be self-explanatory based on the corresponding column that isn't split by engagement. 

Finally, there are columns that are lists of length 4800 that correspond to each image presented during the active behavior period:  
- weight_bias (float) the weight of the licking bias strategy  
- weight_omissions (float) the weight of the omission strategy  
- weight_omissions1 (float) the weight of the post omission strategy  
- weight_task0 (float) the weight of the visual strategy  
- weight_timing1D (float) the weight of the timing strategy  
- lick_bout_rate (float) the rate of licking bouts (units?)  
- hit (float) was it a hit? Nan=image repeat, 1=hit, 0=miss  
- RT (float) reaction time from stimulus onset. Nan=no licking bout start
- image_name (str) stimulus name  
- image_correct_reject (float) 0=licked, 1=did not lick, nan=in licking bout, or image change
- image_false_alarm (floaT), 1=licked, 0=did not lick, nan=image change
- engaged (bool) Was the animal engaged?  
- omitted (bool) was the stimulus omitted  
- is_change (bool) was the stimulus and image change  
- lick_bout_start (bool) did a lick bout start on this image  
- miss (float) nan=image repeat, 0=hit, 1=miss  
- reward_rate (float) reward rate (units?)  
- strategy_weight_index_by_image (float) different in weight of visual and timing strategies  
- lick_hit_fraction_rate (float) the rolling percentage of licking bouts that resulted in a reward (units?)  


### change_df
> import licking_behavior_NP.psy_output_tools as po  
> change_df = po.get_change_table(BEHAVIOR_VERSION)  

### licks_df
> import licking_behavior_NP.psy_output_tools as po  
> licks_df = po.get_licks_table(BEHAVIOR_VERSION)  

The columns of licks_df are:  
- timestamps  (float) time of lick  
- pre_ili (float) time from last lick  
- post_ili (float) time until next lick  
- bout_start (bool) whether this lick was the start of a licking bout  
- bout_end (bool) whether this lick was the end of a licking bout  
- bout_number (bool) oridinal numbering of bouts in this session  
- rewarded (bool) whether this lick was rewarded  
- num_rewards (int) number of rewards resulting from this lick. Can be > 1 from auto-rewards getting assigned to nearest lick  
- bout_rewarded (bool) whether this licking bout was rewarded  
- bout_num_rewards (int) number of rewards resulting from this lick bout. Can be > 1 from auto-rewards getting assigned to nearest lick  
- behavior_session_id (int64)   

### bouts_df
> import licking_behavior_NP.psy_output_tools as po  
> licks_df = po.get_licks_table(BEHAVIOR_VERSION)  
> bouts_df = po.build_bout_table(licks_df)  

The columns of bouts_df are:   
- behavior_session_id (int)  
- bout_number (int)           ordinal count within each session  
- bout_length (int)           number of licks in bout  
- bout_duration (float)       duration of bout in seconds  
- bout_rewarded (bool)        whether this bout was rewarded  
- pre_ibi (float)             time from the end of the last bout to the start of this bout  
- post_ibi (float)            time until the start of the next bout from the end of this bout  
- pre_ibi_from_start (float)  time from the start of the last bout to the start of this bout  
- post_ibi_from_start (float) time from the start of this bout to the start of the next  

### figure script
> import licking_behavior_NP.figure_script as f  
> f.make_figure_1_supplement_behavior()  
> f.make_figure_1_timing_end_of_lick_bout()  
> f.make_figure_1_supplement_task()  
> f.make_figure_1_supplement_licking()  
> f.make_figure_2()  
> f.make_figure_2_supplment_model_validation()  
> f.make_figure_2_supplement_strategy_characterization()  
> f.make_figure_2_supplement_strategy_characterization_rates()  
> f.make_figure_2_supplment_pca()  
> f.make_figure_2_novelty()  
> f.make_figure_3()  





