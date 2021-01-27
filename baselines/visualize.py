import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pilotscripts.data_utils import dropout


def who_dropout(var,frame, i, total_i=4, ylim= False, scatter=False, bar=True):
    
    """
    Given a parameter, generate two plots:
    1. individuals who dropouts
    2. dropout and remain groups
    
    ------------------
    Required Parameters:
    var: variable under investigation
    frame: dataframe
    i: which time to investigate

    Optional Parameters:
    ylim: set a limit for scatter plots
    scatter: if True, generate the plots
    bar: if true, generate the bar plots

    """
    if i >= total_i:
        return 

    this_var = ['_'.join([var,str(i)])]
    next_vars = ['_'.join([var, str(j)]) for j in range(i+1,total_i+1)]
    this_var_appear, next_vars_disappear = dropout(this_var, next_vars, frame)
    frame['drop'] = next_vars_disappear
    pd= frame[this_var_appear][['sid']+this_var+['drop']]
    
    print('total:', pd.shape[0])
    print('dropout people:',pd['drop'].sum())
    print(pd[this_var].isna().sum())
    if scatter:
        sns.scatterplot(pd.sid, pd[this_var[0]],hue=pd['drop'])
        if ylim:
            (a,b) = ylim
            plt.ylim(a,b)
        plt.show()
    
    if bar:
        sns.barplot(pd['drop'],pd[this_var[0]])
        plt.show()


def ploting_the_change(live_outcome, var, variables, total_i=4,RANGE=None):
    
    """
    Given a parameter, generate three plots:
    1. individuals who dropouts
    2. dropout and remain groups
    
    ------------------
    Required Parameters:
    var: variable under investigation
    live_outcome: dataframe
    i: which time to investigate

    Optional Parameters:
    ylim: set a limit for scatter plots
    scatter: if True, generate the plots
    bar: if true, generate the bar plots

    """
    
    #var = var+'_'
    #variables = [''.join([var,str(j)]) for j in range(1,total_i+1)]
    print(var, variables)

    frame = live_outcome[['sid']+variables]
    print(frame.describe())
    # changes at each of the time
    fig, axs = plt.subplots(ncols=len(variables)-1,figsize=(20,5))
    for i in range(1,len(variables)):
        selected = [var+str(i),var+str(i+1)]
        pd=frame[(frame[selected] != 0).all(axis=1)][['sid']+selected].dropna()
        num_data = pd.shape[0]
        print(selected, num_data)
        pd['change'] = pd[selected[1]]-pd[selected[0]]
        pd['decrease'] = pd['change']<0
        sns.scatterplot(pd.sid.str[:-3],pd.change, hue=pd['decrease'],ax= axs[i-1],s=50)
        axs[i-1].legend(loc=4)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    plt.show()

    # changes at each of the time
    fig, axs = plt.subplots(ncols=len(variables)-1,figsize=(20,5))
    for i in range(1,len(variables)):
        selected = [var+str(i),var+str(i+1)]
        pd=frame[(frame[selected] != 0).all(axis=1)][['sid']+selected].dropna()
        num_data = pd.shape[0]
        print(selected, num_data)
        pd['change'] = pd[selected[1]]-pd[selected[0]]
        pd['decrease'] = pd['change']<0
        sns.barplot(pd.change, hue=pd['decrease'],orient="v",ax= axs[i-1] )
        axs[i-1].set_xticklabels(axs[i-1].get_xticklabels(),rotation=90)
        if RANGE:
            (MIN,MAX) = RANGE
            axs[i-1].set_ylim(MIN,MAX)
        axs[i-1].legend(loc=4)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    plt.show()


    newframe = frame[(frame[variables] != 0).all(axis=1)]
    print(newframe.dropna().shape)
    newframe_long = newframe.melt(id_vars='sid',value_vars=variables)
    sns.lineplot('variable','value',data = newframe_long.dropna(),err_style="bars", ci=95)
