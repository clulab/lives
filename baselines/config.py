datafile = 'data/LIVES_NLP_BehaviorOutcomes_BL-24M_01062021.xls'
datapath = '/media/yiyunzhao/yiyun/project_storation/Lives/processed_data/'
# dropout criterion
dropout_criterion_var = 'totmet_hrs_weekx_GE3MET_1'
dropout_criterion_nextvars = ['totmet_hrs_weekx_GE3MET_2','totmet_hrs_weekx_GE3MET_3','totmet_hrs_weekx_GE3MET_4']


## Feature definitions
label_col = ['drop']
categorical_features =  ['diabetes', 'education','high_lipids', 'hypertension',
                        'hyperthyroidism', 'hypothyroidism', 'marital_status', 'osteoporosis',
                        'race','stroke','ethnicity', 'ever_smoked']
continuous_features =   ['bmi_1','ener_1','fruit_serv_whole_1','tfat_1','tfat_pcal_1', 'tfib_1',
                        'totmet_hrs_weekx_GE3MET_1','totsed_hrsday_v2_adj_1','veg_serv_whole_1']

model_parameters ={'naive_bayes':{},
    'logistic_regression':{'penalty':['l2','l1'],'C':[0.1, 0.5, 1, 5, 10, 20, 30, 60, 100],'solver':['liblinear']},
    'k_neighbors':{'n_neighbors': range(1, 10)},
    'decision_tree':{'min_samples_split': range(2,10),'criterion':["gini","entropy"]},
    'random_forest':{'n_estimators':[20, 50, 80, 100, 120, 150, 180]},
    'SVC':{'C':[0.1, 0.5, 1, 5, 10, 20, 30, 60, 100],'kernel':['linear', 'poly','rbf'] },
    'MLP':{'hidden_layer_sizes': [20, 50, 80, 100, 120, 150, 180], 'activation':['relu', 'tanh', 'logistic']},
    'XGB':{'n_estimators': [100, 300, 500, 700, 800, 900], 'learning_rate':[0.01, 0.03, 0.05, 0.07,0.09,0.1]},
    'XGBRF':{'objective':['binary:logistic']}
}


baseline_models = ['naive_bayes','logistic_regression','k_neighbors','SVC','decision_tree']

metrics = ['f1','accuracy','confusion_matrix']


best_params = {'naive_bayes': {},
    'logistic_regression':{'penalty':'l1','C':0.5,'solver':'liblinear'},
    'k_neighbors':{'n_neighbors': 1},
    'decision_tree':{'min_samples_split': 9,'criterion':"gini"},
    'random_forest':{'n_estimators':9},
    'SVC':{'C':60,'kernel':'rbf' },
    #'MLP':{'hidden_layer_sizes': [20, 50, 80, 100, 120, 150, 180], 'activation':['relu', 'tanh', 'logistic']},
    #'XGB':{'n_estimators': [100, 300, 500, 700, 800, 900], 'learning_rate':[0.01, 0.03, 0.05, 0.07,0.09,0.1]},
    #'XGBRF':{'objective':['binary:logistic']}
}
'''
best_params  = {'naive_bayes':{},
    'logistic_regression':{'penalty':['l2','l1'],'C':[0.1, 0.5, 1, 5, 10, 20, 30, 60, 100],'solver':['liblinear']},
    'k_neighbors':{'n_neighbors': range(1, 10)},
    'decision_tree':{'min_samples_split': range(2,10),'criterion':["gini","entropy"]},
    'random_forest':{'n_estimators':[20, 50, 80, 100, 120, 150, 180]},
    'SVC':{'C':[0.1, 0.5, 1, 5, 10, 20, 30, 60, 100],'kernel':['linear', 'poly','rbf'] },
    'MLP':{'hidden_layer_sizes': [20, 50, 80, 100, 120, 150, 180], 'activation':['relu', 'tanh', 'logistic']},
    'XGB':{'n_estimators': [100, 300, 500, 700, 800, 900], 'learning_rate':[0.01, 0.03, 0.05, 0.07,0.09,0.1]},
    'XGBRF':{'objective':['binary:logistic']}
}
''' 
data_splits = 30
splits=10
