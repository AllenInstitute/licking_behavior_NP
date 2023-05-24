# licking_behavior
Analysis of mouse licking behavior during visually guided behavior. Primarily, this repo develops a time-varying logistic regression model that learns the probability of licking on a flash by flash basis, using weights that vary over time by following random walk priors. 

This repository is specific to the Visual Behavior Neuropixels dataset. If you want to look at the Visual Behavior Optical Physiology dataset use github.com/alexpiet/licking_behavior. The model works the same but the data is loaded and processed slightly differently due to the VBN sessions being structured differently. 

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


### Diagram of information flow
![code_diagram](https://user-images.githubusercontent.com/7605170/175404261-4565ab0a-2c82-4215-9840-dffb2b736883.png)


## Model outputs
The key output dataframes are:

- summary_df, each behavioral session is a row, and columns contain model metrics and other behavioral metrics
- change_df, each image change across all behavioral sessions is a row
- licks_df, each lick across all behavioral sessions is a row
- bouts_df, each licking bout across all behavioral sessions is a row

### summary_df
> import licking_behavior_NP.psy_output_tools as po  
> summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)  

### change_df
> import licking_behavior_NP.psy_output_tools as po  
> change_df = po.get_change_table(BEHAVIOR_VERSION)  

### licks_df
> import licking_behavior_NP.psy_output_tools as po  
> licks_df = po.get_licks_table(BEHAVIOR_VERSION)  

### bouts_df
> import licking_behavior_NP.psy_output_tools as po  
> licks_df = po.get_licks_table(BEHAVIOR_VERSION)  
> bouts_df = po.build_bout_table(licks_df)  

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





