import numpy as np
import matplotlib.pyplot as plt

import licking_behavior_NP.psy_tools as ps
import licking_behavior_NP.psy_analysis as pa
import licking_behavior_NP.psy_general_tools as pgt
import licking_behavior_NP.psy_visualization as pv
import licking_behavior_NP.psy_output_tools as po
import licking_behavior_NP.build_timing_regressor as b
import licking_behavior_NP.psy_metrics_tools as pm

plt.ion()

BEHAVIOR_VERSION=100
FIGTYPE='.svg'

def make_figure_1_examples():
    '''
        Plots licking raster for example epochs
    '''
    raise Exception('Need to pick an example bsid')
    session = pgt.get_data(EXAMPLE_BSID)
    pv.plot_strategy_examples(session, version=BEHAVIOR_VERSION, savefig=True)

def make_figure_1_diagram():
    ''' 
        Plots diagram of full session, and image by image weights
    '''
    raise Exception('Need to pick an example bsid')
    session1a = pgt.get_data(FIG1a_BSID)
    pv.add_fit_prediction(session1a,BEHAVIOR_VERSION)
    pv.plot_session_metrics(session1a, plot_list=['target','prediction'],
        plot_example=True,version=BEHAVIOR_VERSION)
    pv.plot_session_weights_example(session1a, version=BEHAVIOR_VERSION)
    session1b = pgt.get_data(FIG1b_BSID)
    pv.plot_session_diagram(session1b, x=[566.5,579.25],version=BEHAVIOR_VERSION)

def make_figure_1_supplement_behavior():
    '''
        Figure 1 Supplement - basic behavior
            num hits, num false alarms?
            licking rate etc over session
    '''
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)
    pv.plot_session_summary_trajectory(summary_df,'reward_rate',version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE,xaxis_images=False,ylim=[0,None],axline=False,
        width=5)
    pv.plot_session_summary_trajectory(summary_df,'RT',version=BEHAVIOR_VERSION,width=5,
        savefig=True, filetype=FIGTYPE,xaxis_images=False,ylim=[0,.75],axline=False)
    event = ['miss', 'image_false_alarm','image_correct_reject','hit']
    pv.plot_session_summary_multiple_trajectory(summary_df,event,
        version=BEHAVIOR_VERSION, savefig=True,filetype=FIGTYPE,
        event_names='responses',xaxis_images=False,width=5,axline=False)

    pv.histogram_df(summary_df,'num_hits',version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE,xlim=[0,None])
    pv.histogram_df(summary_df,'num_miss',version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE,xlim=[0,None])
    pv.histogram_df(summary_df,'num_omission_licks',version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE,xlim=[0,None])
    pv.histogram_df(summary_df,'num_post_omission_licks',version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE,xlim=[0,None])
    pv.histogram_df(summary_df,'num_image_false_alarm',version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE,xlim=[0,None])
    pv.histogram_df(summary_df,'lick_fraction',version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE,xlim=[0,None])
    pv.histogram_df(summary_df,'omission_lick_fraction',version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE,xlim=[0,None])
    pv.histogram_df(summary_df,'post_omission_lick_fraction',version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE,xlim=[0,None])
    pv.histogram_df(summary_df,'trial_hit_fraction',version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE,xlim=[0,None])


def make_figure_1_timing_regressor():
    raise Exception('Need to pick an example bsid')
    bsid = pgt.get_debugging_id(1)
    session = pgt.get_data(bsid)
    #b.build_timing_schematic(session, version=BEHAVIOR_VERSION, savefig=True)
    pv.plot_segmentation_schematic(session, savefig=True, version=BEHAVIOR_VERSION)
    df = b.build_timing_regressor(version=BEHAVIOR_VERSION, savefig=True)
    b.plot_timing_thumbnail(savefig=True, version=BEHAVIOR_VERSION)

    licks_df = po.get_licks_table(BEHAVIOR_VERSION)
    bouts_df = po.build_bout_table(licks_df)
    pv.plot_chronometric(bouts_df,BEHAVIOR_VERSION,savefig=True)

def make_figure_1_timing_end_of_lick_bout():
    raise Exception('need to compute bout table')
    licks_df = po.get_licks_table(BEHAVIOR_VERSION)
    bouts_df = po.build_bout_table(licks_df)
    pv.plot_interlick_interval(bouts_df,key='pre_ibi',version=BEHAVIOR_VERSION,
        categories='post_reward',savefig=True,filetype=FIGTYPE)
    pv.plot_interlick_interval(bouts_df,key='pre_ibi_from_start',
        version=BEHAVIOR_VERSION,categories='post_reward',savefig=True,filetype=FIGTYPE)
    pv.plot_bout_durations(bouts_df, BEHAVIOR_VERSION,savefig=True,filetype=FIGTYPE)


def make_figure_1_supplement_task():
    change_df = po.get_change_table(BEHAVIOR_VERSION)
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)

    pv.plot_image_pair_repetitions(change_df, BEHAVIOR_VERSION,savefig=True,
        filetype=FIGTYPE)
    pv.histogram_df(summary_df, 'num_changes',version=BEHAVIOR_VERSION,
        savefig=True,filetype=FIGTYPE)
    pv.plot_image_repeats(change_df, BEHAVIOR_VERSION,savefig=True,filetype=FIGTYPE)
    pv.histogram_df(summary_df, 'num_hits',version=BEHAVIOR_VERSION,
        savefig=True,filetype=FIGTYPE)


def make_figure_1_supplement_licking():
    raise Exception('need to compute licks table')
    licks_df = po.get_licks_table(BEHAVIOR_VERSION)
    bouts_df = po.build_bout_table(licks_df)
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)
    summary_df['all'] = True

    pv.plot_interlick_interval(licks_df, version=BEHAVIOR_VERSION,
        savefig=True,filetype=FIGTYPE)
    pv.plot_bout_durations(bouts_df, BEHAVIOR_VERSION, savefig=True,filetype=FIGTYPE)
    pv.RT_by_group(summary_df, BEHAVIOR_VERSION, groups=['all'], labels=[''],
        engaged=None, savefig=True, filetype=FIGTYPE,width=5)


def make_figure_2_raw_data():
    raise Exception('need to pick example bsids')
    timing_bsid = 794071128
    visual_bsid = 943479988
    timing_session = pgt.get_data(timing_bsid)
    visual_session = pgt.get_data(visual_bsid)
    pv.plot_raw_traces(timing_session, x= [350.33],version=BEHAVIOR_VERSION, savefig=True)
    pv.plot_raw_traces(visual_session, x= [3000],version=BEHAVIOR_VERSION, savefig=True,
        top=True)


def make_figure_2():
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)
    summary_df = pv.plot_static_comparison(summary_df,version=BEHAVIOR_VERSION, 
        savefig=True,filetype=FIGTYPE)
    pv.plot_session_summary_dropout(summary_df,version=BEHAVIOR_VERSION,savefig=True,
        cross_validation=False, filetype=FIGTYPE)
    pv.plot_session_summary_weights(summary_df,version=BEHAVIOR_VERSION,savefig=True,
        filetype=FIGTYPE)
    pv.plot_session_summary_priors(summary_df,version=BEHAVIOR_VERSION,savefig=True,
        filetype=FIGTYPE)
    pv.plot_session_summary_weight_avg_scatter_task0(summary_df,version=BEHAVIOR_VERSION,
        savefig=True, filetype=FIGTYPE)
    pv.scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index', 
        version=BEHAVIOR_VERSION,flip1=True,flip2=True,cindex='strategy_dropout_index',
        savefig=True,filetype=FIGTYPE,plot_diag=True)
    pv.scatter_df(summary_df, 'strategy_dropout_index','num_hits',
        cindex='strategy_dropout_index',version=BEHAVIOR_VERSION, 
        savefig=True, filetype=FIGTYPE)
    pv.scatter_df_by_mouse(summary_df,'strategy_dropout_index',version=BEHAVIOR_VERSION,
        savefig=True,filetype=FIGTYPE)


def make_figure_2_supplement_model_validation():
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)
    pv.plot_static_comparison(summary_df,version=BEHAVIOR_VERSION, savefig=True,
        filetype=FIGTYPE)
    pv.scatter_df(summary_df,'strategy_dropout_index','lick_hit_fraction', 
        version=BEHAVIOR_VERSION,savefig=True,filetype=FIGTYPE,figsize=(5,4),
        ylim=[0,None])
    pv.scatter_df(summary_df,'strategy_dropout_index','session_roc', 
        version=BEHAVIOR_VERSION,savefig=True,filetype=FIGTYPE,figsize=(5,4),
        ylim=[0.5,1])
    pv.scatter_df(summary_df,'visual_only_dropout_index','lick_hit_fraction', 
        version=BEHAVIOR_VERSION,savefig=True,filetype=FIGTYPE,figsize=(5,4),
        ylim=[0,None])


def make_figure_2_supplement_strategy_characterization():
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)   
    # Plot session-wise metrics against strategy weights
    event=['hits','miss','lick_fraction']
    for e in event:
        pv.plot_session_summary_weight_avg_scatter_task_events(summary_df,e,
        version=BEHAVIOR_VERSION,savefig=True,filetype=FIGTYPE)


def make_figure_2_supplement_strategy_characterization_rates():
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)   
    # Plot image-wise metrics, averaged across sessions
    events = ['bias','task0','omissions','omissions1','timing1D']
    pv.plot_session_summary_multiple_trajectory(\
        summary_df.query('visual_strategy_session'),events,
        version=BEHAVIOR_VERSION, savefig=True,filetype=FIGTYPE,
        event_names='strategies_visual',xaxis_images=False)
    pv.plot_session_summary_multiple_trajectory(\
        summary_df.query('not visual_strategy_session'),events,
        version=BEHAVIOR_VERSION, savefig=True,filetype=FIGTYPE,
        event_names='strategies_timing',xaxis_images=False)

    #events = ['hit','miss']
    events = ['hit','lick_hit_fraction_rate','lick_bout_rate','reward_rate']
    for key in events:
        pv.plot_session_summary_trajectory(summary_df,key,BEHAVIOR_VERSION,
            categories='visual_strategy_session',savefig=True, filetype=FIGTYPE,
            xaxis_images=False,ylim=[0,None],axline=False)

def make_figure_2_supplement_pca():
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)
    pa.compute_PCA(summary_df, version=BEHAVIOR_VERSION,on='dropout',
        savefig=True)


def make_figure_2_novelty():
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)  
    pv.scatter_df_by_experience(summary_df, ['Familiar','Novel 1'],
        'strategy_dropout_index',experience_type='experience_level',
        version=BEHAVIOR_VERSION,savefig=True, filetype=FIGTYPE) 
    pv.histogram_df_by_experience(summary_df,['Familiar','Novel 1'],
        'strategy_dropout_index',experience_type='experience_level',
        version=BEHAVIOR_VERSION,savefig=True,filetype=FIGTYPE)
    pv.plot_pivoted_df_by_experience(summary_df,'strategy_dropout_index',
        version=BEHAVIOR_VERSION, savefig=True, filetype=FIGTYPE)
    keys = ['lick_hit_fraction_rate','task0','timing1D','lick_bout_rate']
    for key in keys:
        pv.plot_session_summary_trajectory(summary_df,key,BEHAVIOR_VERSION,
            categories='experience_level',savefig=True,filetype=FIGTYPE,xaxis_images=False)


def make_figure_3():
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)
    pv.plot_engagement_analysis(summary_df,BEHAVIOR_VERSION,savefig=True, filetype=FIGTYPE,
        just_landscape=True)
    pv.plot_engagement_landscape_by_strategy(summary_df, z='weight_task0',
        savefig=True, version=BEHAVIOR_VERSION)
    pv.plot_engagement_landscape_by_strategy(summary_df, z='weight_timing1D',
        savefig=True, version=BEHAVIOR_VERSION)
    pv.plot_session_summary_trajectory(summary_df,'engaged',
        version=BEHAVIOR_VERSION, categories='visual_strategy_session',
        savefig=True, filetype=FIGTYPE, ylim=[0,100],axline=False,xaxis_images=False, 
        ylabel_extra='fraction ',paper_fig=True)
    pv.RT_by_engagement(summary_df,BEHAVIOR_VERSION,savefig=True, filetype=FIGTYPE,
        key='engaged')
    pv.RT_by_group(summary_df,BEHAVIOR_VERSION,engaged='engaged',ylim=.004,
        savefig=True, filetype=FIGTYPE,key='engaged',width=5)
    pv.RT_by_group(summary_df,BEHAVIOR_VERSION,engaged='disengaged',ylim=.004,
        savefig=True, filetype=FIGTYPE,key='engaged',width=5)

def make_figure_3_example():
    raise Exception('Need to pick an example bsid')
    session = pgt.get_data(FIG3_BSID)
    pv.plot_session_metrics(session, plot_list=['reward_rate'],
        plot_engagement_example=True,version=BEHAVIOR_VERSION)

def make_figure_4_supplement_strategy_matched():
    raise Exception('Need to define strategy matched subset')
    summary_df = po.get_np_summary_table(BEHAVIOR_VERSION)
    pv.scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index',
        flip1=True, flip2=True,categories='cre_line',savefig=True,  
        version=BEHAVIOR_VERSION,filetype=FIGTYPE,figsize=(5,4))
    pv.histogram_df(summary_df, 'strategy_dropout_index',categories='cre_line',
        savefig=True, version=BEHAVIOR_VERSION,filetype=FIGTYPE,xlim=[-45,45])
    pv.histogram_df(summary_df.query('strategy_matched'), 'strategy_dropout_index',
        categories='cre_line',savefig=True, version=BEHAVIOR_VERSION,
        filetype=FIGTYPE,group='strategy_matched',xlim=[-45,45])


