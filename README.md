# Outcome-Oriented Prescriptive Process Monitoring
This repository contains the source code of a prescriptive process monitoring system that provides recommendations for achieving a positive outcome of an ongoing process using all the proposed encoding types in the literature: Declare, boolean, frequency, simple index, latest index and complex-based index encodings.

## Repository Structure
- `media/input` contains the input logs in `.csv` format. Before reproducing the experiments it is necessary to download 
  and unzip the log folder from [here](https://drive.google.com/file/d/1DDP7OKQhD8cno2tbSpLlIPZ-Mh5y-XUC/view?usp=sharing);
- `media/output` contains the numeric results regarding the performance of the prescriptive system;
- `models` contains the pre-trained models trained on the datasets described in the paper.
- `src` contains the backbone code;
- `settings.py` contains the main settings for the experiments as described in the paper below;
- `dataset_figures.py` is a Python script to extract the dataset figures and save them in a `.csv` file in the 
  `media/output` folder;
- `experiments_runner.py` is the main Python script for running the experiments;
- `gather_results.py` is a Python script for aggregating the results of each dataset and presenting in a more 
  understandable format.
  
## Requirements
The following Python packages are required:

-   [numpy](http://www.numpy.org/) tested with version 1.25.0;
-   [PM4PY](https://pm4py.fit.fraunhofer.de/) tested with version 2.7.4;
-   [sklearn](https://scikit-learn.org/stable/) tested with version 1.2.2;
-   [pandas](https://pandas.pydata.org/) tested with version 2.0.2.
-   [matplotlib](https://matplotlib.org/) tested with version 3.7.1;
-   [imbalanced-learn](https://pypi.org/project/imbalanced-learn/) tested with version 0.10.1;
-   [https://seaborn.pydata.org/](https://seaborn.pydata.org/) tested with version 0.12.2.

## Usage
The system has been tested with Python 3.11.4. After installing the requirements, please download this repository.

### Running the code
To run the evaluation for a given (pretrained) dataset, type:
```
python3 experiments_runner.py --log=Production
```
if you don't want to train again your model, you need to set the `--load_model` option:
```
$ python experiments_runner.py --log=Production --load_model
```
The encoding types are grouped into ten families. The first five encoding types, proposed by [Leontjeva et al., (2015)](https://www.researchgate.net/publication/281446750_Complex_Symbolic_Sequence_Encodings_for_Predictive_Monitoring_of_Business_Processes), are: boolean, frequency, simple index, latest index, and complex-based index encodings. The last five encoding families, representing the DECLARE constraints proposed by [Donadello et al., (2023)](https://www.sciencedirect.com/science/article/pii/S0952197623010837), are: existence, choice, positive relations, negative relations, and all. If you need to use a subset of these encoding families (e.g., existence and frequency) use:
```
$ python experiments_runner.py --log=Production --decl_list="existence,frequency"
```
You can also train a model on your own dataset `my_event_log` saved in the standard `.csv` format for event log.
First of all, define you have to add the needed keys to the configuration dictionaries in the `src/dataset_manager/DatasetManager.py` file:
```
dataset = "my_event_log"
filename[dataset] = os.path.join(logs_dir, "my_event_log.csv")
case_id_col[dataset] = "CaseID_my_event_log"
activity_col[dataset] = "ActivityID_my_event_log"
resource_col[dataset] = "ResourceID_my_event_log"
timestamp_col[dataset] = "CompleteTimestamp_my_event_log"
label_col[dataset] = "labelID_my_event_log"
pos_label[dataset] = "pos_label_my_event_log"
neg_label[dataset] = "neg_label_my_event_log"
```
then, you just need to run
```
$ python3 experiments_runner.py --log=my_event_log --load_model=False
```
Type:
```
./run_experiments.sh
```
to run the experiments on the whole pool of datasets in parallel, in Unix systems. For Windows systems, you
need to run `.bat` file.

### Gathering the results
After running the experiments, type:
```
$ python plot_time_performance.py
```
to aggregate the computational times of generating the recommendations for all datasets. The results will be in the 
file `aggregated_recommendation_times.pdf` in `media/output/result`. 

Type
```
$ python gather_results.py
```
to have an aggregation of the results for each dataset. Such aggregation are found in the files in the `media/output` 
folder.
