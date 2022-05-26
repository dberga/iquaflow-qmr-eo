#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# QMR execution settings
plot_sne = False                         # t-SNE plot? (requires a bit of RAM)
plot_metrics_comp = True                 # metrics comparison?
use_existing_metrics = True              # read existing metrics output data files instead of processing them?
regressor_quality_metrics = ['sigma','snr','rer','sharpness','scale','score']

#Define path of the original (reference) datasets
data_paths = {
"inria-aid10": "./Data/inria-aid_short",
"UCMerced380": "./Data/test-ds",
}
image_folders = {
    "inria-aid10": "test/images_short",
    "UCMerced380": "test",
}


# In[ ]:


# Imports
import os
import argparse
import shutil
import mlflow
import pandas as pd
from pdb import set_trace as debug # debugging

from custom_modifiers import DSModifierFake
from iquaflow.datasets import DSModifier, DSWrapper
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.experiment_visual import ExperimentVisual
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.quality_metrics import RERMetrics, SNRMetrics, GaussianBlurMetrics, NoiseSharpnessMetrics, GSDMetrics, ScoreMetrics


# In[ ]:


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


# In[ ]:


# Save Figs if python extension, show if notebook
_, extension = os.path.splitext(__file__)
savefig = False if extension == "ipynb" else True


# In[ ]:


dataframes = []
for ids, database_name in enumerate(list(data_paths.keys())):

    data_path = data_paths[database_name]
    images_path = os.path.join(data_paths[database_name],image_folders[database_name])
    python_ml_script_path = 'sr.py'  

    #Define name of IQF experiment
    experiment_name='eoqmr'
    experiment_name += f"_{database_name}"
    
    # set output folders
    plots_folder = "plots/"+experiment_name+"/"
    results_folder = "results/"+experiment_name+"/"
    
    # plot SNE of existing images
    if plot_sne:
        plotSNE(database_name, data_path, (232,232), 6e4, True, savefig, plots_folder)

    #DS wrapper is the class that encapsulate a dataset
    ds_wrapper = DSWrapper(data_path=data_path)
    
    # Define and execute script (sr.py: copy image files to testing folder)
    ds_wrapper = DSWrapper(data_path=data_path)
    ds_modifiers_list = [DSModifierFake(name="base",images_dir=images_path)]
    task = PythonScriptTaskExecution( model_script_path = python_ml_script_path )
    experiment = ExperimentSetup(
        experiment_name=experiment_name,
        task_instance=task,
        ref_dsw_train=ds_wrapper,
        ds_modifiers_list=ds_modifiers_list,
        ref_dsw_val=ds_wrapper,
        repetitions=1,
        extra_train_params={"trainds": [data_path], "traindsinput": [images_path], "valds": [data_path],"valdsinput": [images_path]}
    )
    # Enforce replacement of paths in experiment wrappers
    '''
    experiment.ref_dsw_train.data_input = images_path
    if experiment.ref_dsw_test is not None:
        experiment.ref_dsw_test.data_input = images_path
    if experiment.ref_dsw_val is not None:
        experiment.ref_dsw_val.data_input = images_path 
    '''
    
    #Execute the experiment
    experiment.execute()
    experiment_info = ExperimentInfo(experiment_name)
    # ExperimentInfo is used to retrieve all the information of the whole experiment. 
    # It contains built in operations but also it can be used to retrieve raw data for futher analysis

    print('Calculating Quality Metric Regression...'+",".join(regressor_quality_metrics)) #default configurations
    path_regressor_quality_metrics = f'./{results_folder}_regressor_quality_metrics.csv'
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

    # plot metric comparison
    if plot_metrics_comp:
        metric_comp(df,regressor_quality_metrics,savefig,plots_folder)
    
    print(f"writing {path_regressor_quality_metrics}")
    df.to_csv(path_regressor_quality_metrics)
    
    # append table of current dataset
    dataframes.append(df)


# In[ ]:


# concat all dataset tables
df = pd.concat(dataframes)
path_all_datasets = f"{results_folder}/results.csv"
print(f"writing {path_all_datasets}")
df.to_csv(path_all_datasets)


# In[ ]:


df

