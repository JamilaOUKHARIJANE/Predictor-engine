import argparse
import os
import pickle
import time
from csv import DictWriter

import numpy as np
import pandas as pd
from nonconformist.icp import IcpClassifier
from nonconformist.nc import NcFactory
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.dataset_manager import DatasetManager, latest_IndexBasedTransformer

scoring = { "precision": make_scorer(precision_score, pos_label=1, average="binary", zero_division=0),
            "recall":    make_scorer(recall_score,    pos_label=1, average="binary", zero_division=0),
            "f1":        make_scorer(f1_score,        pos_label=1, average="binary", zero_division=0),
            }

# ================ folders ================
output_dir = "media/output"
results_dir = os.path.join(output_dir, "result")
dataset_folder = "media/input/processed_benchmark_event_logs"
support_threshold_dict = {'min': 0.0, 'max': 1.75}
#=========================


def train_svm(X_train, y_train):
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Feature scaling
        ('svc', SVC(probability=True))  # SVM classifier
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring=scoring, refit="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search

def train_decision_tree(X_train, y_train):
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'class_weight': ['balanced', None],
        'max_depth': [4, 6, 8, 10, None],
        'min_samples_split': [0.1, 2, 0.2, 0.3],
        'min_samples_leaf': [10, 1, 16]
    }

    dt = DecisionTreeClassifier()

    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=3,
        scoring=scoring,     # compute all 3 metrics
        refit="f1",          # select best params by F1
        n_jobs=-1,
        return_train_score=False
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def train_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 1, 5],
        'scale_pos_weight': [1, 10, 25]  # Useful for imbalanced datasets
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # Disable deprecation warnings
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring=scoring, refit="f1",n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def select_best_model(metrics, grids):
    best_name = max(metrics, key=lambda n: metrics[n]["f1_mean"])
    best_grid = grids[best_name]
    best_model = best_grid.best_estimator_

    print("Best model:", best_name)
    print("Best CV F1:", metrics[best_name]["f1_mean"], "+/-", metrics[best_name]["f1_std"])
    print("Best params:", best_grid.best_params_)
    return best_model, best_name

def conformal_metrics_from_pvalues(p_values, y_true, alpha, class_labels=("noAdapt", "Adapt")):
    p_values = np.asarray(p_values)
    y_true = np.asarray(y_true, dtype=int)
    n, K = p_values.shape

    adapt_idx = class_labels.index("Adapt")
    noadapt_idx = class_labels.index("noAdapt")

    pred_set = (p_values > alpha)
    coverage = pred_set[np.arange(n), y_true].mean()
    avg_set_size = pred_set.sum(axis=1).mean()
    # Predict Adapt only if set is singleton {Adapt}; otherwise predict noAdapt
    y_pred = np.where(pred_set[:, adapt_idx] & (~pred_set[:, noadapt_idx]), 1, 0)  # Adapt=1, noAdapt=0

    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return float(coverage), float(avg_set_size), float(precision), float(recall), float(f1)


def evaluate_predictive_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} Results:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\n")
    return precision, recall, f1

def run_experiment(log_data, dataset_name,X_train_cal, y_train_cal,X_test,y_test,fold):

    # split into training and calibration sets
    X_train, X_cal, y_train, y_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.2, random_state=42)

    print("start training")
    print("train DT")
    start_time= time.time()
    gs_dt = train_decision_tree(X_train, y_train)
    end_time_DT = time.time() - start_time
    with open(os.path.join(output_dir, f'{dataset_name}_DT.pickle'),
              'wb') as file:
        pickle.dump((gs_dt.best_estimator_), file)

    print("train Xgboost")
    start_time = time.time()
    gs_xgboost = train_xgboost(X_train, y_train)
    end_time_Xgboost = time.time() - start_time
    with open(os.path.join(output_dir, f'{dataset_name}_xgboost_fold{fold}.pickle'),
              'wb') as file:
        pickle.dump((gs_xgboost.best_estimator_), file)

    print("start cross-validation")
    # Validate models
    grids = { "DT": gs_dt, "XGBoost": gs_xgboost}

    metrics = {}
    for name, gs in grids.items():
        best = gs.best_index_
        cv = gs.cv_results_

        metrics[name] = {
            "precision_mean": cv["mean_test_precision"][best],
            "precision_std": cv["std_test_precision"][best],
            "recall_mean": cv["mean_test_recall"][best],
            "recall_std": cv["std_test_recall"][best],
            "f1_mean": cv["mean_test_f1"][best],
            "f1_std": cv["std_test_f1"][best],
        }

    # best performing model
    best_model, name= select_best_model(metrics,grids)

    # Conformal calibration
    print("start Calibration....")
    cal_time = {}
    conformal_metrics={}
    times={}

    start_time = time.time()
    nc = NcFactory.create_nc(best_model)
    # Fit predictive model inside ICP
    icp = IcpClassifier(nc)
    icp.fit(X_train, y_train)
    # Calibrate on D_cal
    icp.calibrate(X_cal, y_cal)
    cal_time[name] = time.time() - start_time
    start_time = time.time()
    test_pval = icp.predict(X_test.values)
    prediction_time = (time.time()-start_time) / len(X_test)
    times[name]= prediction_time
    # Save p-values
    test_pval = pd.DataFrame(test_pval, columns=['p_value_0', 'p_value_1'])
    results_df = pd.DataFrame({
                'true_labels': y_test.values,
                'p_value_0': test_pval['p_value_0'],
                'p_value_1': test_pval['p_value_1']
        })
    results_df.to_csv(output_dir + f'{dataset_name}1_p_values{name}_predictions_fold{fold+1}.csv', sep=';', index=False)
    prec, rec, f1 =evaluate_predictive_model(best_model, X_test, y_test,name)
    conformal_metrics[name] = {"precision": prec, "recall": rec, "f1": f1}
    print(f"Precision (Adapt): {prec:.3f}")
    print(f"Recall (Adapt): {rec:.3f}")
    print(f"F1 (Adapt): {f1:.3f}")

    output_file = f'{output_dir}/experiments_results_fold.csv'
    with open(output_file, 'a+', newline='') as output_file:
        fieldnames = ["dataset", "training_DT", "training_Xgboost", "best_model"
                      "cal_time",
                      "DT_prec_mean", "DT_prec_std", "DT_rec_mean", "DT_rec_std", "DT_f1_mean", "DT_f1_std",
                      "XGBoost_prec_mean", "XGBoost_prec_std", "XGBoost_rec_mean", "XGBoost_rec_std", "XGBoost_f1_mean", "XGBoost_f1_std",
                      "prec_test", "rec_test", "f1_test", "pred_test_time",
                      ]

        dict_writer = DictWriter(output_file, fieldnames=fieldnames)
        if output_file.tell() == 0:
            dict_writer.writeheader()
        dict_writer.writerow({
            "dataset": dataset_name,
            "training_DT": round(end_time_DT, 3),
            "training_Xgboost": round(end_time_Xgboost, 3),
            "best_model":name,
            "cal_time": round(cal_time[name], 3),
            "DT_prec_mean": metrics["DT"]["precision_mean"], "DT_prec_std": metrics["DT"]["precision_std"],
            "DT_rec_mean": metrics["DT"]["recall_mean"], "DT_rec_std": metrics["DT"]["recall_std"],
            "DT_f1_mean": metrics["DT"]["f1_mean"], "DT_f1_std": metrics["DT"]["f1_std"],
            "XGBoost_prec_mean": metrics["XGBoost"]["precision_mean"], "XGBoost_prec_std": metrics["XGBoost"]["precision_std"],
            "XGBoost_rec_mean": metrics["XGBoost"]["recall_mean"], "XGBoost_rec_std": metrics["XGBoost"]["recall_std"],
            "XGBoost_f1_mean": metrics["XGBoost"]["f1_mean"], "XGBoost_f1_std": metrics["XGBoost"]["f1_std"],
           "prec_test":conformal_metrics[name]["precision"] , "rec_test": conformal_metrics[name]["recall"], "f1_test": conformal_metrics[name]["f1"],
            "pred_test_time": times[name]
        })

    print(f"Done with: {dataset_name}...\n")

if __name__ == "__main__":
    # Load your data
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=None, help='input log')
    parser.add_argument('--maxlen', default=None, help='max length of log traces')
    args = parser.parse_args()
    dataset_name = args.log
    max_prefix_data = args.maxlen

    #for dataset_name, max_prefix_data in zip(['hospital_billing', 'sepsis_cases',"FMPlog"], [25,61,8]):
    dataset_manager = DatasetManager(dataset_name.lower())
    data = dataset_manager.read_dataset(os.path.join(os.getcwd(), dataset_folder))
    print("splitting Data")
    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    train_data, test_data = dataset_manager.split_data(data, train_ratio=0.8, split="temporal")

    if dataset_name != "FMPlog": v_star1, v_star2, v_star3 = dataset_manager.get_dominant_variant(train_data)

    train_data = dataset_manager.generate_prefix_data(data=train_data, min_length=1, max_length=max_prefix_data, gap=1)#,static_case_cols=static_case_cols,dynamic_event_cols=dynamic_event_cols)
    test_data = dataset_manager.generate_prefix_data(data=test_data, min_length=1, max_length=max_prefix_data, gap=1)#,static_case_cols=static_case_cols,dynamic_event_cols=dynamic_event_cols)

    if dataset_name != "FMPlog":
        train_data = dataset_manager.label_data_by_dominant_variant(v_star1, v_star2,v_star3, train_data)
        test_data = dataset_manager.label_data_by_dominant_variant(v_star1, v_star2,v_star3, test_data)
    labelled_data= pd.concat([test_data,train_data], ignore_index=True)
    if dataset_name != "FMPlog":
        test_data = labelled_data
    pos_labels= len(labelled_data[labelled_data[dataset_manager.label_col] == dataset_manager.pos_label].groupby(dataset_manager.case_id_col).size())
    neg_labels = len(labelled_data[labelled_data[dataset_manager.label_col] == dataset_manager.neg_label].groupby(dataset_manager.case_id_col).size())
    print('pos labels', pos_labels)
    print('neg labels', neg_labels)

    print("start encoding")
    all_cat_cols = dataset_manager.static_cat_cols + dataset_manager.dynamic_cat_cols
    cat_maps_ ={}
    for col in all_cat_cols:
        vals = sorted(data[col].astype(str).fillna("__MISSING__").unique().tolist())
        cat_maps_[col] = {v: i + 1 for i, v in enumerate(vals)}
    encoder_args = {'case_id_col': dataset_manager.case_id_col,
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols,
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_activity_col': dataset_manager.activity_col,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                        'label_col': dataset_manager.label_col,
                        'pos_label': dataset_manager.pos_label,
                        'cat_maps_': cat_maps_,
                        'max_events':max_prefix_data,
                        'fillna': True}
    encoder = latest_IndexBasedTransformer(**encoder_args, support_threshold_dict=support_threshold_dict)
    X_train_cal, y_train_cal = encoder.fit_transform(train_data)
    X_test, y_test = encoder.fit_transform(test_data)

    # Run the experiment
    for fold in range(3):
        run_experiment(data, dataset_name,X_train_cal, y_train_cal,X_test,y_test,fold)
