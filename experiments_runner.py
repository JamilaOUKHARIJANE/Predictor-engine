import pdb
import sys
import pickle
from src.dataset_manager.datasetManager import DatasetManager
from src.machine_learning import *
import argparse
import time
import numpy as np
import csv
from src.machine_learning.conformal_prediction import conformal_prediction
from pm4py.objects.conversion.log import converter as log_converter


def rec_sys_exp(dataset_name):
    dataset_manager = DatasetManager(dataset_name.lower())
    data = dataset_manager.read_dataset(os.path.join(os.getcwd(), settings.dataset_folder))
    constr_family = "latestindex"
    method = "latestindex"

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    max_prefix_data =min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    
    
    data = dataset_manager.generate_prefix_data(data=data, min_length=1, max_length=max_prefix_data, gap=1)
    #print(data.groupby(dataset_manager.case_id_col).tail(1))
    encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                    'static_cat_cols':dataset_manager.static_cat_cols,
                    'static_num_cols':dataset_manager.static_num_cols,
                    'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                    'dynamic_activity_col':dataset_manager.dynamic_activity_col, 
                    'dynamic_num_cols':dataset_manager.dynamic_num_cols, 
                    'label_col': dataset_manager.label_col,
                    'pos_label': dataset_manager.pos_label,
                    'fillna':True}      
    encoder = get_encoder(method, settings.support_threshold_dict, **encoder_args)
    X1, y = encoder.fit_transform(data) 
    '''X1=X
    X=X.drop(dataset_manager.label_col, axis=1)
    X.drop("prefix", axis=1) 
    print(X)'''
    
    data = data.rename(columns={dataset_manager.timestamp_col: 'time:timestamp',
                                dataset_manager.case_id_col: 'case:concept:name',
                                dataset_manager.activity_col: 'concept:name'}) 
    data = log_converter.apply(data)
    features, encoded_data = freq_encode_traces(data)
    X2 = pd.DataFrame(encoded_data, columns=features)
    X= pd.concat([X1, X2], axis=1)
    print(X)

        
    # split into training and test
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train_val, y_train_val, test_size=0.2,random_state=42)
   
    
    labeling = {
        "type": LabelType.TRACE_CATEGORICAL_ATTRIBUTES,
        "threshold_type": "",
        "target": TraceLabel.TRUE,  # lower than a threshold considered as True
        "trace_lbl_attr": dataset_manager.label_col,
        "trace_label": dataset_manager.pos_label,
        "custom_threshold": 0.0
    }
   
    # generate recommendations and evaluation
    results = [] 
    time_results = [["dataset_name", "case_id", "prefix_length", "total_recommendation_time"]]
    train_opt = [["dataset_name", "train_opt_time"]]
    recommendations_res=[]

    if constr_family == "latestindex":         
        start_train_opt_time = timeit.default_timer()
       

        if load_model:
            try:
                with open(os.path.join(settings.models_path, f'{dataset_name}_{constr_family}.pickle'), 'rb') as file:
                    model, best_hyperparams_combination = pickle.load(file)
                    #print(len(paths))
                    print(f"Model {dataset_name}_{constr_family}.pickle loaded")
                    
            except FileNotFoundError as not_found:
                print(f"Model {dataset_name}_{constr_family}.pickle not found. Invalid path or you "
                      f"have to train a model before loading.")
                sys.exit(2)
        else:
            feat_strategy_paths_dict = {strategy: None for strategy in settings.num_feat_strategy}
            hyperparams_evaluation_list = []
            results_hyperparams_evaluation = {}
            hyperparams_evaluation_list_baseline = []

            for v1 in settings.sat_threshold_list:
                # the baseline chooses the path with highest probability
                hyperparams_evaluation_list_baseline.append((v1,) + (0, 0, 1))
                for v2 in settings.weight_combination_list:
                    hyperparams_evaluation_list.append((v1,) + v2)

            for feat_strategy in settings.num_feat_strategy:                
                model,feature_names = train_path_recommender(method, X_train_val, y_train_val, labeling=labeling, support_threshold=settings.support_threshold_dict, dataset_name=dataset_name,output_dir=settings.output_dir, feat_strategy=feat_strategy) 
                
                feat_strategy_paths_dict[feat_strategy] = model

                #discovering on val set with best hyperparams_evaluation setting
                print(f"Hyper params for evaluation for {dataset_name} ...")
                if compute_baseline:
                    hyperparams_evaluation_list = hyperparams_evaluation_list_baseline

                for hyperparams_evaluation in hyperparams_evaluation_list:
                    res_val_list = []
                    y_pred = model.predict(X_cal)
                    f1 = f1_score(y_cal, y_pred, average='weighted')
                    results_hyperparams_evaluation[(feat_strategy, ) + hyperparams_evaluation] = f1

            results_hyperparams_evaluation = dict(sorted(results_hyperparams_evaluation.items(), key=lambda item: item[1]))
            best_hyperparams_combination = list(results_hyperparams_evaluation.keys())[-1]
            model = feat_strategy_paths_dict[best_hyperparams_combination[0]]
            best_hyperparams_combination = best_hyperparams_combination[1:]
            print(f"BEST HYPERPARAMS COMBINATION {best_hyperparams_combination}")
            with open(os.path.join(settings.models_path, f'{dataset_name}_{constr_family}.pickle'), 'wb') as file:
                pickle.dump((model, best_hyperparams_combination), file)
        
        
        # saving training/optimization times
        training_optimization_time = timeit.default_timer() - start_train_opt_time
        train_opt += [[dataset_name, constr_family, training_optimization_time]]
        
    ################################################## Conformal Prediction ###########################################################################
    from nonconformist.icp import IcpClassifier
    from nonconformist.nc import ClassifierNc, MarginErrFunc
    from nonconformist.icp import IcpClassifier
    from nonconformist.base import ClassifierAdapter
    from nonconformist.nc import NcFactory
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    # Create a nonconformity measure
    nc = NcFactory.create_nc(model)

    # Wrap the classifier in an IcpClassifier
    icp = IcpClassifier(nc)

    # Fit the IcpClassifier
    icp.fit(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    rf_model = icp.nc_function.model.model

    # Removing cal data from the model
    icp.cal_x=[]
    icp.cal_y=[]

    test_pval = icp.predict(X_test.values)
    #print(test_pval)
    sys.path.append("plot_utils/python/src/")
    from pharmbio.cp import metrics
    from pharmbio.cp import plotting
    
    test_pval = pd.DataFrame(test_pval, columns=['p_value_0', 'p_value_1'])  

    results_df = pd.DataFrame({
        'true_labels': y_test.values,
        'p_value_0': test_pval['p_value_0'],
        'p_value_1': test_pval['p_value_1']
    })


    results_df.to_csv(settings.output_dir+'/C_DT_predictions.csv', sep=';', index=False)

    print(f"Done with: {dataset_name}...\n")
      

    
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments for predictive analysis")
    parser.add_argument('--log', default=None, help='input log')
    parser.add_argument('--load_model', action='store_true', help='Use trained model')
    parser.add_argument('--baseline', action='store_true', help='Use baseline model')
        
    args = parser.parse_args()
    dataset = args.log
    load_model = args.load_model
    compute_baseline = args.baseline

    start_time = time.time()
    res_obj = rec_sys_exp(dataset)
    print(f"Simulations took {(time.time() - start_time) / 3600.} hours for {dataset}")