import pandas as pd
import wandb

from pilotscripts import config, data_utils, baseline_model as md


# MODEL PARAMETERS EXPLORATION
def parameters_exploration(config, write=True,fname ='params_result_report.csv',splits=10):
    train, _  = data_utils.load_data(config, preprocessing=False)
    #print('train sets',train['label'][True].value_counts())
    #print('test sets', test['label'][True].value_counts())
    
    # fix the training sets but sampled different parameters for models
    t, d = data_utils.data_split(train['data'],train['label'],random_seed=None,save=False)
    print('train sets',t['label'][True].value_counts())
    print('dev sets', d['label'][True].value_counts())


    res = []
    for i in range(splits):
        bmds = md.baseline_models(config)
        bmds.initialize()
        bmds.train(t['data'], t['label'])
        bmds.evaluate(d['data'],d['label'])
        res+= bmds.report(False)

    resdf = pd.DataFrame(res).sort_values(by=['f1','accuracy'],ascending=False)
    
    if write:
        resdf.to_csv(fname)



def result_estimation(config, write=True, fname='stable_report_test.csv'):
    train, _  = data_utils.load_data(config, preprocessing=False)
    res = []
    for i in range(config.data_splits):
        t, d = data_utils.data_split(train['data'],train['label'],random_seed=None,save=False)
        bmds = md.baseline_models(config)
        bmds.initialize(config.best_params)
        bmds.train(t['data'], t['label'])
        bmds.evaluate(d['data'],d['label'])
        res+= bmds.report(False, idx=i,baseline_label=d['label'])
    resdf = pd.DataFrame(res).sort_values(by=['model_type','id'],ascending=True)
    if write:
        resdf.to_csv(fname)
        
    return 