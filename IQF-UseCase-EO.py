#!/usr/bin/env python
# coding: utf-8

# In[1]:


# QMR execution settings
plot_sne = False                         # t-SNE plot? (requires a bit of RAM)
plot_metrics_comp = False                 # metrics comparison?
savefig = False                           # save figs or show in line
use_existing_metrics = True              # read existing metrics output data files instead of processing them?
regressor_quality_metrics = ['sigma','snr','rer','sharpness','scale','score']

#Define path of the original (reference) datasets
data_paths = {
    "DeepGlobe469": "./Data/DeepGlobe",
    "UCMerced2100": "./Data/UCMerced_LandUse",
    "UCMerced380": "./Data/test-ds",
    "USGS279": "./Data/USGS",
    "shipsnet-ships4000": "./Data/shipsnet",
    "shipsnet-scenes7": "./Data/shipsnet",
    "inria-test10": "./Data/inria-aid_short",
    "inria-test180": "./Data/AerialImageDataset",
    "inria-train180": "./Data/AerialImageDataset",
    "XView-train846": "./Data/XView",
    "XView-val281": "./Data/XView",
}
image_folders = {
    "DeepGlobe469": "images",
    "UCMerced2100": "Images/ALL_CATEGORIES",
    "UCMerced380": "test",
    "USGS279": "hr_images",
    "shipsnet-ships4000": "shipsnet",
    "shipsnet-scenes7": "scenes/scenes",
    "inria-test10": "test/images_short",
    "inria-test180": "test/images",
    "inria-train180": "train/images",
    "XView-train846": "train_images",
    "XView-val281": "val_images",
}

image_resolutions = {
    "DeepGlobe469": 1024,
    "UCMerced2100": 256,
    "UCMerced380": 256,
    "USGS279": 1024,
    "shipsnet-ships4000": 64,
    "shipsnet-scenes7": 1024,
    "inria-test10": 1024,
    "inria-test180": 1024,
    "inria-train180": 1024,
    "XView-train846": 1024,
    "XView-val281": 1024,
}


# In[2]:


# Imports
import os
import argparse
import shutil
import mlflow
import pandas as pd
from pdb import set_trace as debug # debugging

# display tables of max 50 columns
pd.set_option('display.max_columns', 50)

from custom_modifiers import DSModifierFake
from iquaflow.datasets import DSModifier, DSWrapper
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.experiment_visual import ExperimentVisual
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.quality_metrics import RERMetrics, SNRMetrics, GaussianBlurMetrics, NoiseSharpnessMetrics, GSDMetrics, ScoreMetrics
from visual_comparison import metric_comp, plotSNE


# In[3]:


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


# In[4]:


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
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    
    # plot SNE of existing images
    if plot_sne:
        plotSNE(database_name, data_path, (232,232), 6e4, True, savefig, plots_folder)

    #DS wrapper is the class that encapsulate a dataset
    ds_wrapper = DSWrapper(data_path=data_path)
    
    # Define and execute script (sr.py: copy image files to testing folder)
    ds_wrapper = DSWrapper(data_path=data_path)
    ds_modifiers_list = [DSModifier()] # DSModifierFake(name="base",images_dir=images_path)
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
    path_regressor_quality_metrics = f'./{results_folder}regressor_quality_metrics.csv'
    if use_existing_metrics and os.path.exists(path_regressor_quality_metrics):
        df = pd.read_csv(path_regressor_quality_metrics, index_col='ds_modifier')
    else:
        _ = experiment_info.apply_metric_per_run(ScoreMetrics(input_size=image_resolutions[database_name]), ds_wrapper.json_annotations)
        _ = experiment_info.apply_metric_per_run(RERMetrics(input_size=image_resolutions[database_name]), ds_wrapper.json_annotations)
        _ = experiment_info.apply_metric_per_run(SNRMetrics(input_size=image_resolutions[database_name]), ds_wrapper.json_annotations)
        _ = experiment_info.apply_metric_per_run(GaussianBlurMetrics(input_size=image_resolutions[database_name]), ds_wrapper.json_annotations)
        _ = experiment_info.apply_metric_per_run(NoiseSharpnessMetrics(input_size=image_resolutions[database_name]), ds_wrapper.json_annotations)
        _ = experiment_info.apply_metric_per_run(GSDMetrics(input_size=image_resolutions[database_name]), ds_wrapper.json_annotations)
        df = experiment_info.get_df(
            ds_params=["ds_modifier"],
            metrics=regressor_quality_metrics,
            dropna=False
        )
        df.to_csv(path_regressor_quality_metrics)

    # check results
    df["ds_modifier"] = database_name
    
    # Clean dataframe
    df = df[['ds_modifier']+regressor_quality_metrics]
    df = df.set_index('ds_modifier')
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"writing {path_regressor_quality_metrics}")
    df.to_csv(path_regressor_quality_metrics)
    
    # append table of current dataset
    dataframes.append(df)


# In[5]:


# concat all dataset tables
#df = pd.concat(dataframes)
results_folder = "results/"
path_all_datasets = f"{results_folder}results.csv"
print(f"writing csv: {path_all_datasets}")
df = pd.concat(dataframes,axis=0)
{df.drop(str(field),inplace=True,axis=1) for field in df if "Unnamed" in str(field)}
df.to_csv(path_all_datasets)


# In[6]:


df


# In[7]:


# plot metric comparison
if plot_metrics_comp:
    metric_comp(df,regressor_quality_metrics,savefig,plots_folder)

