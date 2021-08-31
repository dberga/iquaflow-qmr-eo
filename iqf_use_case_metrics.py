import os
import argparse

from iq_tool_box.datasets import DSModifier, DSWrapper, DSModifier_sr
from iq_tool_box.experiments import ExperimentInfo, ExperimentSetup
from iq_tool_box.experiments.experiment_visual import ExperimentVisual
from iq_tool_box.experiments.task_execution import PythonScriptTaskExecution
from iq_tool_box.quality_metrics import RERMetrics, SNRMetrics, GaussianBlurMetrics, NoiseSharpnessMetrics, ResolScaleMetrics

def main(
    data_path = "./tests/test_datasets/inria-aid_short/test/images_short", 
    ml_models_path = "./tests/test_ml_models",
    mock_model_script_name = "sr.py"
    ):

    python_ml_script_path = os.path.join(ml_models_path, mock_model_script_name)

    ds_wrapper = DSWrapper(data_path=data_path)
    ds_modifiers_list = [DSModifier()]

    task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)

    experiment_name='regressor'
    experiment = ExperimentSetup(
        experiment_name=experiment_name,
        task_instance=task,
        ref_dsw_train=ds_wrapper,
        ds_modifiers_list=ds_modifiers_list,
        ref_dsw_val=ds_wrapper,
        repetitions=1
    )

    experiment.execute()
    experiment_info = ExperimentInfo(experiment_name)

    metric = RERMetrics()
    results_run = experiment_info.apply_metric_per_run(metric, str(None))

    metric = SNRMetrics()
    results_run = experiment_info.apply_metric_per_run(metric, str(None))

    metric = GaussianBlurMetrics()
    results_run = experiment_info.apply_metric_per_run(metric, str(None))

    metric = NoiseSharpnessMetrics()
    results_run = experiment_info.apply_metric_per_run(metric, str(None))

    metric = ResolScaleMetrics()
    results_run = experiment_info.apply_metric_per_run(metric, str(None))

    df = experiment_info.get_df(
        ds_params=["modifier"],
        metrics=[
            "rer",
            "snr",
            "sigma",
            "sharpness",
            "scale"
        ]
        )

    print("\n\n\************************************\n\n")
    print('METRICS:\n')
    print(df)
    print("\n\n\************************************\n\n")

if __name__=="__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--imf',
                        help='Images folder of the dataset. Its parent directory contains the annotations.json file',
                        default="./tests/test_datasets/inria-aid_short/test/images_short",
                        type=str)
    
    args = parser.parse_args()

    # root_path = "./" 
    # tests_path = os.path.join(root_path, "tests")
    # ml_models_path = os.path.join(tests_path, "test_ml_models")

    # base_ds = os.path.join(tests_path, "test_datasets")
    # ds_path = os.path.join(base_ds, "inria-aid_short")
    # data_path = os.path.join(ds_path, "test/images_short")
    
    main(
        data_path = args.imf, 
        ml_models_path = "./tests/test_ml_models",
        mock_model_script_name = "sr.py"
        )