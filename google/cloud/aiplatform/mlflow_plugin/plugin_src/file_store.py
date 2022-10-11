from mlflow.store.tracking import file_store

from google.cloud.aiplatform.compat.types import execution as gca_execution

import uuid
from google.cloud import aiplatform


class VertexMlflowTracking(file_store.FileStore):
    """FileStore provided through entrypoints system"""

    def __init__(self, store_uri=None, artifact_uri=None):
        self.autolog_setting = store_uri.split('/')[2]

        current_experiment = aiplatform.metadata.metadata._experiment_tracker.experiment.name
        current_run = aiplatform.metadata.metadata._experiment_tracker.experiment_run

        self.vertex_experiment = current_experiment
        self.vertex_experiment_run = current_run

        super(VertexMlflowTracking, self).__init__()

    def create_run(self, experiment_id, user_id, start_time, tags):
        framework = ""

        for tag in tags:
            if tag.key == "mlflow.autologging":
                framework = tag.value

        # Create a new run for the user only if they've called aiplatform.autolog()
        if self.autolog_setting == "global_autolog":
            new_vertex_run = f"{framework}-{uuid.uuid4()}"
            self.vertex_experiment_run = aiplatform.start_run(run=new_vertex_run)
    
        return super().create_run(experiment_id, user_id, start_time, tags)

    def update_run_info(self, run_id, run_status, end_time):

        # a run_status of 3 means the run has finished
        # see here: https://www.mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunStatus
        if run_status == 3 and self.autolog_setting == "global_autolog":
            self.vertex_experiment_run.end_run()

        return super().update_run_info(run_id, run_status, end_time)



    def log_batch(self, run_id, metrics, params, tags):
        summary_metrics = {}
        params = {}
        time_series_metrics = {}

        if len(metrics) > 0:
            for metric in metrics:
                if metric.step:
                    if metric.step not in summary_metrics:
                        time_series_metrics[metric.step] = {metric.key:metric.value}
                    else:
                        time_series_metrics[metric.step][metric.key] = metric.value
                else:
                    summary_metrics[metric.key] = metric.value

        if len(params) > 0:
            for param in params:
                params[param.key] = param.value

        if summary_metrics:
            self.vertex_experiment_run.log_metrics(metrics=summary_metrics)

        # TODO: if there are ts metrics but no summary metrics, should we log the metrics from the last step as summary metrics?
        # if summary_metrics is None and time_series_metrics is not None:
        
        if params:
            self.vertex_experiment_run.log_params(params=params)

        if time_series_metrics:
            for step, ts_metrics in time_series_metrics.items():
                aiplatform.log_time_series_metrics(ts_metrics, step)

    def log_metric(self, run_id, metric):

        print('in log metric')
        return self.log_batch(run_id, metric)
        