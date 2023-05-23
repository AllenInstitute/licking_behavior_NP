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

