import numpy as np
import psy_general_tools as pgt
import psy_tools as ps
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd

def get_train_summary():
    train_summary = pd.read_csv('/home/alex.piet/codebase/behavior/model_output/_training_summary_table.csv')
    return train_summary

def plot_strategy_correlation(train_summary):
    donor_ids = train_summary.query('imaging').donor_id.unique()
    mouse_summary = train_summary.pivot(index='donor_id',columns='pre_ophys_number',values=['task_dropout_index'])
    mouse_summary['ophys_index'] = mouse_summary['task_dropout_index'][[-5,-4,-3,-2,-1,0]].mean(axis=1)
    plt.figure(figsize=(10,5))
    plt.axvspan(0,6,color='k',alpha=.1)
    plt.axhline(0, color='k',linestyle='--',alpha=0.5)
    for dex,val in enumerate(train_summary.pre_ophys_number.unique()):
        try:
            if len(mouse_summary['task_dropout_index'][val].unique())> 10:
                plt.plot(-val, mouse_summary['ophys_index'].corr(mouse_summary['task_dropout_index'][val]),'ko')
        except:
            print('crash')
    plt.ylabel('Strategy Index Correlation',fontsize=16)
    plt.xlabel('Sessions before Ophys Stage 1',fontsize=16)
    plt.xlim(right=6)   
    plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/strategy_correlation.svg')
    plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/strategy_correlation.png')

def plot_training(train_summary):
    '''
        train_summary is found in  _training_summary_table.csv
    '''
    donor_ids = train_summary.query('imaging').donor_id.unique()

    plt.figure(figsize=(10,5))
    plt.axhline(0,color='k',linestyle='--',alpha=0.5) 
    plt.axvspan(0,6,color='k',alpha=.1)
    x = []
    y = []
    c = []
    for dex, donor_id in enumerate(donor_ids):
        mouse_table = train_summary.query('donor_id == @donor_id')
        vals = mouse_table.task_dropout_index.values
        xvals = -mouse_table.pre_ophys_number.values
        if len(vals) > 0:
            plt.plot(xvals, vals,'k-',alpha=.05)
            x = x + list(xvals)
            y = y + list(vals)
            c = c + list(np.ones(np.size(vals))*mouse_table.query('imaging').task_dropout_index.mean())

    scat = plt.gca().scatter(x, y, s=80,c =c, cmap='plasma',alpha=0.5)
    plt.ylabel('Strategy Index',fontsize=16)
    plt.xlabel('Sessions before Ophys Stage 1',fontsize=16)
    #plt.xticks(xvals, ['T3','T4','T5','Hab', 'Ophys1','Ophys3','Ophys4','Ophys6'],fontsize=14)
    #plt.yticks(fontsize=14)
    plt.xlim(right=6)
    #plt.axvline(0,linestyle='--',color='k')
    plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_session_number.svg')
    plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_session_number.png')
    #plt.xlim(left=-20)
    #plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_session_number2.svg')
    #plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_session_number2.png')

def plot_training_dropout(train_summary):
    '''
        train_summary is found in  _training_summary_table.csv
    '''
    donor_ids = train_summary.query('imaging').donor_id.unique()

    plt.figure(figsize=(10,5))
    plt.axhline(0,color='k',linestyle='--',alpha=0.5) 
    x = []
    y = []
    c = []
    for dex, donor_id in enumerate(donor_ids):
        mouse_table = train_summary.query('donor_id == @donor_id')
        vals = [mouse_table.query('(not ophys) & (stage == "3")').task_dropout_index.mean(),
        mouse_table.query('(not ophys) & (stage == "4")').task_dropout_index.mean(),
        mouse_table.query('(not ophys) & (stage == "5")').task_dropout_index.mean(),
        mouse_table.query('(ophys) & (stage == "0")').task_dropout_index.mean(),
        mouse_table.query('(ophys) & (stage == "1")').task_dropout_index.mean(),
        mouse_table.query('(ophys) & (stage == "3")').task_dropout_index.mean(),
        mouse_table.query('(ophys) & (stage == "4")').task_dropout_index.mean(),
        mouse_table.query('(ophys) & (stage == "6")').task_dropout_index.mean()]
        xvals = [-3,-2,-1,0,1,2,3,4]
        plt.plot(xvals, vals,'k-',alpha=.05)
        x = x + xvals
        y = y + vals
        c = c + list(np.ones(np.size(vals))*mouse_table.query('imaging').task_dropout_index.mean())

    scat = plt.gca().scatter(x, y, s=80,c =c, cmap='plasma',alpha=0.5)

    vals = [train_summary.query('(not ophys) & (stage == "3")').task_dropout_index.mean(),
    train_summary.query('(not ophys) & (stage == "4")').task_dropout_index.mean(),
    train_summary.query('(not ophys) & (stage == "5")').task_dropout_index.mean(),
    train_summary.query('(ophys) & (stage == "0")').task_dropout_index.mean(),
    train_summary.query('(ophys) & (stage == "1")').task_dropout_index.mean(),
    train_summary.query('(ophys) & (stage == "3")').task_dropout_index.mean(),
    train_summary.query('(ophys) & (stage == "4")').task_dropout_index.mean(),
    train_summary.query('(ophys) & (stage == "6")').task_dropout_index.mean()]
    plt.plot(xvals, vals, 'k-',linewidth=2)


    plt.ylabel('Strategy Index',fontsize=16)
    plt.xlabel('Stage',fontsize=16)
    plt.xticks(xvals, ['T3','T4','T5','Hab', 'Ophys1','Ophys3','Ophys4','Ophys6'],fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_stage.svg')
    plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_stage.png')

def plot_training_roc(train_summary):
    '''
        train_summary is found in  _training_summary_table.csv
    '''
    donor_ids = train_summary.query('ophys & (stage > "0")').donor_id.unique()

    plt.figure(figsize=(10,5))
    plt.axhline(train_summary.session_roc.mean(),color='k',linestyle='--',alpha=0.5) 
    
    x = []
    y = []
    c = []
    for dex, donor_id in enumerate(donor_ids):
        mouse_table = train_summary.query('donor_id == @donor_id')
        vals = [mouse_table.query('(not ophys) & (stage == "3")').session_roc.mean(),
        mouse_table.query('(not ophys) & (stage == "4")').session_roc.mean(),
        mouse_table.query('(not ophys) & (stage == "5")').session_roc.mean(),
        mouse_table.query('(ophys) & (stage == "0")').session_roc.mean(),
        mouse_table.query('(ophys) & (stage == "1")').session_roc.mean(),
        mouse_table.query('(ophys) & (stage == "3")').session_roc.mean(),
        mouse_table.query('(ophys) & (stage == "4")').session_roc.mean(),
        mouse_table.query('(ophys) & (stage == "6")').session_roc.mean()]
        xvals = [-3,-2,-1,0,1,2,3,4]
        plt.plot(xvals, vals,'k-',alpha=.1)
        x = x + xvals
        y = y + vals
        c = c + vals 

    scat = plt.gca().scatter(x, y, s=80,c =c, cmap='plasma',alpha=0.5)

    vals = [train_summary.query('(not ophys) & (stage == "3")').session_roc.mean(),
    train_summary.query('(not ophys) & (stage == "4")').session_roc.mean(),
    train_summary.query('(not ophys) & (stage == "5")').session_roc.mean(),
    train_summary.query('(ophys) & (stage == "0")').session_roc.mean(),
    train_summary.query('(ophys) & (stage == "1")').session_roc.mean(),
    train_summary.query('(ophys) & (stage == "3")').session_roc.mean(),
    train_summary.query('(ophys) & (stage == "4")').session_roc.mean(),
    train_summary.query('(ophys) & (stage == "6")').session_roc.mean()]
    plt.plot(xvals, vals, 'm-',linewidth=2)

    plt.ylabel('Session ROC',fontsize=16)
    plt.xlabel('Stage',fontsize=16)
    plt.xticks(xvals, ['T3','T4','T5','Hab', 'Ophys1','Ophys3','Ophys4','Ophys6'],fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0.6,1)

    plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/roc_by_stage.svg')
    plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/roc_by_stage.png')


