from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn import preprocessing


from pilotscripts.data_utils import sample_parameters
from pilotscripts import config



class baseline_models:
    
    def __init__(self, config):
        self.model_types= {
        'naive_bayes':GaussianNB,
        'logistic_regression':LogisticRegression,
        'k_neighbors':KNeighborsClassifier,
        'decision_tree':DecisionTreeClassifier,
        'random_forest':RandomForestClassifier,
        'SVC':SVC,
        'MLP':MLPClassifier
        }

        self.metric_types = {'f1': f1_score, 'accuracy': accuracy_score, 'confusion_matrix': confusion_matrix}

        self.config = config
        self.model_collections = []
    
    def initialize(self, fix_params = None):
        model_collections = []
        for md in config.baseline_models:
            if not fix_params:
                md_params = sample_parameters(config.model_parameters[md])
            else:
                md_params = fix_params[md]
            #print(md_params, md)
            md_baseline = self.model_types[md](**md_params) 
            this_model = {"params": md_params, 'model': md_baseline, 'model_type': md}
            model_collections.append(this_model)
        self.model_collections = model_collections

    def train(self, data, label):
        for idx, baseline_model in enumerate(self.model_collections):
            try:
                baseline_model['model'] = baseline_model['model'].fit(data, label)
            except:
                print('failed models', idx,baseline_model)
                baseline_model['model'] = None
                pass

    def evaluate(self, data, label):
        for baseline_model in self.model_collections:
            if baseline_model['model'] is not None:
                for metric in config.metrics:
                    baseline_model[metric] = self.metric_types[metric](label, baseline_model['model'].predict(data))

    def get_baseline(self, label):
        dic = {}
        dic['model_type'] = 'baseline'
        for metric in config.metrics:
            dic[metric] = self.metric_types[metric](label, len(label)*[1])
        return [dic]

    def report(self, verbal=True,idx=None, baseline_label=None):
        reports = []
        bl = []
        if baseline_label is not None: bl= self.get_baseline(baseline_label)
        for i in self.model_collections + bl :
            new_dict = {k:i[k] for k in i if k != 'model'}
            if idx is not None: new_dict['id'] = idx
            reports.append(new_dict)
        if verbal:
            print(reports)
        return reports



