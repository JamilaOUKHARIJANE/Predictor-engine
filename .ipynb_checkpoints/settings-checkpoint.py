import os
from src.enums.ConstraintChecker import ConstraintChecker

# ================ thresholds ================
support_threshold_dict = {'min': 0.0, 'max': 1.75}
sat_threshold = 0.75
top_K_paths = 6
reranking = False
sat_type = 'count_occurrences'  # count_occurrences or count_activations or strong
fitness_type = 'mean'  # mean or wmean
cumulative_res = True
optmize_dt = True
print_dt = True
compute_gain = False
smooth_factor = 1
num_classes = 2
train_prefix_log = False
one_hot_encoding = True
use_score = True
prova = False
compensation= False #True

# ================ folders ================
output_dir = "media/output"
results_dir = os.path.join(output_dir, "result")
dataset_folder = "media/input/processed_benchmark_event_logs"
models_path = "models"


checkers = {"latestindex": []}

constr_family_list = checkers.keys()
#constr_family_list = ["existence"]

# ================ datasets ================
datasets_labels = {"FMPlog": "fmplog"}

datasets_names = ["FMPlog"]


# ================ hyperparameters ================
"""
hyperparameters = {'support_threshold': [support_threshold_dict['min']-0.2, support_threshold_dict['min']-0.1,
                                         support_threshold_dict['min'],
                                         support_threshold_dict['min']+0.1],
                   'class_weight': [None, 'balanced'],
                   'min_samples_split': [2]}
"""
hyperparameters = {"class_weight": ['balanced', None]}
dt_hyperparameters = {'criterion': ['entropy', 'gini'],
                      'class_weight': ['balanced', None],
                      'max_depth': [4, 6, 8, 10, None],
                      'min_samples_split': [0.1, 2, 0.2, 0.3],
                      'min_samples_leaf': [10, 1, 16]}

num_feat_strategy = ['sqrt',  0.3, 0.5]
#num_feat_strategy = [0.3, 0.5]
sat_threshold_list = [0.55, 0.65, 0.75, 0.85]
#sat_threshold_list = [0.85]
weight_combination_list = [(0.2, 0.4, 0.4), (0.6, 0.2, 0.2), (0.4, 0.4, 0.2), (0.4, 0.2, 0.4), (0.8, 0.1, 0.1), (0.4, 0.3, 0.3), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)]
#weight_combination_list = [(0.4, 0.4, 0.2)]

# ================ plots ================
method_label = {'existence': r'$\mathcal{E}$', 'choice': r'$\mathcal{\widehat{C}}$',
                'positive relations': r'$\mathcal{\widehat{PR}}$', 'negative relations': r'$\mathcal{\widehat{NR}}$',
                'all': r'$\mathcal{A}$', 'boolean':'Boolean','frequency':'Frequency-based', 'latestindex': 'Latest index', 'new':'Hybrid' }
method_marker = {'existence': 'x', 'choice': '1', 'positive relations': '.', 'negative relations': '', 'all': '+', 'boolean':'*', 'frequency':'+', 'simpleindex': '', 'complex': '.', 'latestindex':'*', 'new':'+'}
method_color = {'existence': 'mediumpurple', 'choice': 'deepskyblue', 'positive relations': 'orange',
                'negative relations': 'crimson', 'all': 'forestgreen', 'boolean':'red', 'frequency':'blue', 'simpleindex': 'green', 'complex': 'yellow', 'latestindex': 'gray','new':'black'}
method_style = {'existence': 'solid', 'choice': (0, (1, 1)), 'positive relations': 'dashdot',
                'negative relations': (0, (5, 10)), 'all': 'dashdot', 'boolean':'solid', 'frequency':'solid', 'simpleindex': 'solid', 'complex': 'solid','latestindex': 'dashdot', 'new':'dashdot'}