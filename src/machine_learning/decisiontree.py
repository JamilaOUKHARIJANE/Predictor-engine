import copy
import timeit
import pdb
from src.enums import LabelThresholdType, ConfusionMatrix
from src.models import EvaluationResult
import csv
import numpy as np
import settings
from sklearn import metrics
from pm4py.objects.conversion.log import converter as log_converter
import sys
import pickle
from src.dataset_manager.datasetManager import DatasetManager
from src.machine_learning import *
import time

import graphviz
import os
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from src.models import *
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from sklearn.model_selection import GridSearchCV
from src.machine_learning.utils import *
from pm4py.objects.log import obj as log
from pm4py.objects.log.util.get_prefixes import get_log_with_log_prefixes
from src.dataset_manager.datasetManager import DatasetManager
from pm4py.objects.conversion.log import converter as log_converter
from src.machine_learning.labeling import *


class ParamsOptimizer:
    def __init__(self, data, train_val_df, val_df, train_df, parameters, labeling, checkers, rules,
                 min_prefix_length, max_prefix_length):
        self.parameter_names = parameters.keys()
        self.val_df = val_df
        self.data = data
        self.train_val_df = train_val_df
        self.param_grid = [element for element in itertools.product(*parameters.values())]
        self.train_df = train_df
        self.parameters = parameters
        self.labeling = labeling
        self.checkers = checkers
        self.rules = rules
        self.model_grid = []
        self.min_prefix_length = min_prefix_length
        self.max_prefix_length = max_prefix_length


def train_path_recommender(method,X_train, y_train, labeling, support_threshold, dataset_name, output_dir,  feat_strategy):
     
    print(f"Generating decision tree with params optimization for {dataset_name} using {method} encoding...")
    if settings.optmize_dt:
        best_model_dict, feature_names = find_best_dt(method, dataset_name, X_train, y_train, labeling, support_threshold, settings.print_dt, feat_strategy)
    with open(os.path.join(output_dir, 'model_params.csv'), 'a') as f:
        w = csv.writer(f, delimiter='\t')
        row = list(best_model_dict.values())
        w.writerow(row[:-1]) # do not print the model
    
    return best_model_dict['model'],feature_names

def find_best_dt(method, dataset_name, X_train, y_train, labeling, support_threshold_dict, render_dt,
                 num_feat_strategy):
    
    dataset_manager = DatasetManager(dataset_name.lower())   
    
    print(f"DT params optimization for {dataset_name} using {method} encoding...")
    categories = [TraceLabel.FALSE.value, TraceLabel.TRUE.value]
    model_dict = {'dataset_name': dataset_name, 'parameters': (),
                  'f1_score_val': None, 'f1_score_train': None, 'f1_prefix_val': None, 'max_depth': 0,
                  'model': None}

    features = X_train.columns
  
    '''if num_feat_strategy == 'sqrt':
        num_feat = int(math.sqrt(len(features)))
    else:
        num_feat = int(num_feat_strategy * len(features))

    sel = SelectKBest(mutual_info_classif, k=num_feat)
    X_train = sel.fit_transform(X_train, y_train)
    cols = sel.get_support(indices=True)
    
    features = np.array(features)[cols]'''

    print(f"Grid search for {dataset_name} using {method} encoding...")
    search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=settings.dt_hyperparameters,
                          scoring="f1", return_train_score=True, cv=5)
    search.fit(X_train, y_train)

    model_dict['model'] = search.best_estimator_
    f1_score_train = round(100*search.cv_results_['mean_train_score'][search.best_index_], 2)
    model_dict['f1_score_val'] = round(100*search.best_score_, 2)
    model_dict['f1_score_train'] = f1_score_train
    model_dict['f1_prefix_val'] = -1
    model_dict['max_depth'] = search.best_estimator_.tree_.max_depth
    model_dict['parameters'] = tuple(search.best_params_.values())

    if render_dt:
        dot_data = tree.export_graphviz(search.best_estimator_, out_file=None, impurity=True,
                                        feature_names=features, node_ids=True, filled=True)
                                        # class_names=['regular', 'deviant'])
        graph = graphviz.Source(dot_data, format="pdf")
        graph.render(os.path.join(settings.output_dir, f'DT_{dataset_name}_{num_feat_strategy}'))
    return model_dict, features

