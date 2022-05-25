# QMR execution settings
plot_sne = False                         # t-SNE plot? (requires a bit of RAM)
plot_metrics_comp = True                 # metrics comparison?
use_existing_metrics = True              # read existing metrics output data files instead of processing them?
regressor_quality_metrics = ['sigma','snr','rer','sharpness','scale','score']

# Imports
import os
import argparse
import shutil
import mlflow
import pandas as pd

from iquaflow.datasets import DSModifier, DSWrapper, DSModifier_sr
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.experiment_visual import ExperimentVisual
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.quality_metrics import RERMetrics, SNRMetrics, GaussianBlurMetrics, NoiseSharpnessMetrics, GSDMetrics

#Define name of IQF experiment
experiment_name='EOQMR'

# Remove previous mlflow records of previous executions of the same experiment
try: # rm_experiment
    mlflow.delete_experiment(ExperimentInfo(f"{experiment_name}").experiment_id)
    # Clean mlruns and __pycache__ folders
    shutil.rmtree("mlruns/",ignore_errors=True)
    os.makedirs("mlruns/.trash", exist_ok=True)
    shutil.rmtree(f"{data_path}/.ipynb_checkpoints",ignore_errors=True)
    [shutil.rmtree(x) for x in glob(os.path.join(os.getcwd(), "**", '__pycache__'), recursive=True)]
except:
    pass

#Define path of the original (reference) datasets
data_paths = {
"inria-aid10": "./Data/inria-aid_short/test/images_short",
"UCMerced380": "./Data/test-ds/test",
}
dataframes = []

for ids, database_name in enumerate(list(data_paths.keys())):

    data_path = data_paths[database_name]
    ml_models_path = "./test_ml_models"
    mock_model_script_name = 'sr.py'

    #DS wrapper is the class that encapsulate a dataset
    ds_wrapper = DSWrapper(data_path=data_path)

    # plot SNE of existing images
    if plot_sne:
        plotSNE(database_name, data_path, (232,232), 6e4, True, True, "plots/")

    # Define and execute script (sr.py: copy image files to testing folder)
    python_ml_script_path = os.path.join(ml_models_path, mock_model_script_name)
    ds_wrapper = DSWrapper(data_path=data_path)
    ds_modifiers_list = [DSModifier()]
    task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
    experiment = ExperimentSetup(
        experiment_name=experiment_name,
        task_instance=task,
        ref_dsw_train=ds_wrapper,
        ds_modifiers_list=ds_modifiers_list,
        ref_dsw_val=ds_wrapper,
        repetitions=1
    )

    #Execute the experiment
    experiment.execute()
    experiment_info = ExperimentInfo(experiment_name)
    # ExperimentInfo is used to retrieve all the information of the whole experiment. 
    # It contains built in operations but also it can be used to retrieve raw data for futher analysis

    print('Calculating Quality Metric Regression...'+",".join(regressor_quality_metrics)) #default configurations
    path_regressor_quality_metrics = f'./{database_name}_regressor_quality_metrics.csv'
    if use_existing_metrics and os.path.exists(path_regressor_quality_metrics):
        df = pd.read_csv(path_regressor_quality_metrics)
    else:
        _ = experiment_info.apply_metric_per_run(ScoreMetrics(), ds_wrapper.json_annotations)
        _ = experiment_info.apply_metric_per_run(RERMetrics(), ds_wrapper.json_annotations)
        _ = experiment_info.apply_metric_per_run(SNRMetrics(), ds_wrapper.json_annotations)
        _ = experiment_info.apply_metric_per_run(GaussianBlurMetrics(), ds_wrapper.json_annotations)
        _ = experiment_info.apply_metric_per_run(NoiseSharpnessMetrics(), ds_wrapper.json_annotations)
        _ = experiment_info.apply_metric_per_run(GSDMetrics(), ds_wrapper.json_annotations)
        df = experiment_info.get_df(
            ds_params=["modifier"],
            metrics=regressor_quality_metrics,
            dropna=False
        )
        df.to_csv(path_regressor_quality_metrics)

    # check results
    df["modifier"] = database_name
    print(df)

    # plot metric comparison
    if plot_metrics_comp:
        metric_comp(df,regressor_quality_metrics,True,"plots/")

    # append table of current dataset
    dataframes.append(df)

# concat all dataset tables
df = pd.concat(dataframes)
print(df)