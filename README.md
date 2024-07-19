# Predictive Analysis  Service for Proactive Business Process Adaptation Need Detection
This repository contains the source code of a predictor engine that provides service for proactive process adaptation need detection.

## Repository Structure
- `media/input` contains the input logs in `.csv` format; 
- `media/output` contains the numeric results regarding the performance of the conformal prediction;
- `src` contains the backbone code for training and conformal prediction;
- `plot_utils` contains the backbone code for plotting;
- `settings.py` contains the main settings for the experiments as described in the paper below;
- `experiments_runner.py` is the main Python script for running the experiments;
  
## Requirements
The following Python packages are required:

-   [numpy](http://www.numpy.org/) tested with version 1.25.0;
-   [sklearn](https://scikit-learn.org/stable/) tested with version 1.2.2;
-   [pandas](https://pandas.pydata.org/) tested with version 2.0.2.
-   [matplotlib](https://matplotlib.org/) tested with version 3.7.1;
-   [imbalanced-learn](https://pypi.org/project/imbalanced-learn/) tested with version 0.10.1;
-   [https://seaborn.pydata.org/](https://seaborn.pydata.org/) tested with version 0.12.2.

## Usage
The system has been tested with Python 3.11.4. After installing the requirements, please download this repository.

### Running the code
To run the evaluation for a given dataset, type:
```
python experiments_runner.py --log=FMPlog
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
$ python experiments_runner.py --log=my_event_log
```
### Gathering the results
After running the experiments, collect results according to EQ1 by running the following "plot_EQ1.ipynb" notebook.
