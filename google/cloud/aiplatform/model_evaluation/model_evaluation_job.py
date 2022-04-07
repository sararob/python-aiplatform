# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from google.auth import credentials as auth_credentials

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import models
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform import pipeline_based_service

from typing import (
    List,
    Optional,
)

from google.cloud.aiplatform_v1.types import model_evaluation

_LOGGER = base.Logger(__name__)

_MODEL_EVAL_PIPELINE_TEMPLATE = "/Users/sararob/Dev/sara-fork/python-aiplatform/google/cloud/aiplatform/model_evaluation/sdk_pipeline_experimental.json"


class ModelEvaluationJob(pipeline_based_service._VertexAiPipelineBasedService):

    _template_ref = _MODEL_EVAL_PIPELINE_TEMPLATE

    @property
    def metadata_output_artifact(self) -> Optional[str]:
        """The resource uri for the ML Metadata output artifact from the last component of the Model Evaluation pipeline"""
        return super().metadata_output_artifact
        # set if pipeline is complete

    def __init__(
        self,
        evaluation_pipeline_run: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves a ModelEvaluationJob and instantiates its representation.

        Example Usage:

            my_evaluation = aiplatform.ModelEvaluationJob(
                pipeline_job_id = "projects/123/locations/us-central1/pipelineJobs/456"
            )

            my_evaluation = aiplatform.ModelEvaluationJob(
                pipeline_job_id = "456"
            )

        Args:
            evaluation_pipeline_run (str):
                Required. A fully-qualified pipeline job run ID.
                Example: "projects/123/locations/us-central1/pipelineJobs/456" or
                "456" when project and location are initialized or passed.
            project (str):
                Optional. Project to retrieve pipeline job from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve pipeline job from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this pipeline job. Overrides
                credentials set in aiplatform.init.
        """
        super().__init__(
            pipeline_job_id=evaluation_pipeline_run,
            project=project,
            location=location,
            credentials=credentials,
        )

    @classmethod
    def submit(
        cls,
        model: models.Model,
        prediction_type: str,
        pipeline_root: str,  # rename this in the caller to evaluation_staging_bucket?
        target_column_name: str,
        gcs_source_uris: List[str],
        class_names: Optional[List[str]],
        instances_format: Optional[str] = "jsonl",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "ModelEvaluationJob":
        """Submits a Model Evaluation Job using aiplatform.PipelineJob and returns
        the ModelEvaluationJob resource.

        Example usage:

        my_evaluation = ModelEvaluationJob.submit(
            model=aiplatform.Model(model_name="..."),
            prediction_type="classification",
            pipeline_root="gs://my-pipeline-bucket/runpath",
            gcs_source_uris=["gs://test-prediction-data"],
            class_names=["cat", "dog"],
            target_column_name=["animal_type"],
            instances_format="jsonl",
        )

        my_evaluation = ModelEvaluationJob.submit(
            model=aiplatform.Model(model_name="..."),
            prediction_type="regression",
            pipeline_root="gs://my-pipeline-bucket/runpath",
            gcs_source_uris=["gs://test-prediction-data"],
            target_column_name=["price"],
            instances_format="jsonl",
        )

        Args:
            model (str):
                Required. The aiplatform.Model resource to run the ModelEvaluationJob on.
            prediction_type (str):
                Required. The type of prediction performed by the Model. One of "classification" or "regression".
            pipeline_root (str):
                Required. The GCS directory to store output from the model evaluation PipelineJob.
            gcs_source_uris (List[str]):
                Required. A list of GCS URIs containing your input data for batch prediction.
                TODO add details on input source reqs.
            target_column_name (str):
                Required. The name of your prediction column.
            class_names (List[str]):
                Required when `prediction_type` is "classification". A list of all possible class names
                for your classification model's output, in the same order as they appear in the batch prediction input file.
            instances_format (str):
                The format in which instances are given, must be one of the Model's supportedInputStorageFormats. If not set, defaults to "jsonl".
        Returns:
            model: Updated model resource.
        Raises:
            ValueError: If `labels` is not the correct format.
        """

        # TODO: validate the passed in Model resource can be used for model eval

        template_params = {
            "batch_predict_gcs_source_uris": gcs_source_uris,
            "batch_predict_instances_format": instances_format,
            "class_names": class_names,
            "model_name": model.resource_name,
            "prediction_type": prediction_type,
            "project": project or initializer.global_config.project,
            "location": location or initializer.global_config.location,
            "root_dir": pipeline_root,
            "target_column_name": target_column_name,
        }

        eval_pipeline_run = cls._create_and_submit_pipeline_job(
            template_params=template_params,
            pipeline_root=pipeline_root,
            project=project,
            location=location,
            credentials=credentials,
        )

        return eval_pipeline_run

    def get_model_evaluation(
        self,
        display_name: str,
    ) -> Optional[model_evaluation.ModelEvaluation]:
        """Creates a ModelEvaluation resource and instantiates its representation.
        Args:
            display_name (str):
                Required. The display name for your model evaluation resource.
        Returns:
            aiplatform.ModelEvaluation: Instantiated representation of the ModelEvaluation resource.
        Raises:
            RuntimeError: If the ModelEvaluationJob pipeline failed.
        """
        eval_job_state = self.backing_pipeline_job.state

        if eval_job_state in pipeline_jobs._PIPELINE_ERROR_STATES:
            raise RuntimeError(
                f"Evaluation job failed. For more details see the logs: {self.pipeline_console_uri}"
            )
        elif eval_job_state not in pipeline_jobs._PIPELINE_COMPLETE_STATES:
            _LOGGER.info(
                f"Your evaluation job is still in progress. For more details see the logs {self.pipeline_console_uri}"
            )
        else:
            # TODO: waiting for updated pipeline template that creates the ModelEvaluation resource
            _LOGGER.info(
                f"Your evaluation job ran successfully. Creating Model Evaluation with name {display_name}"
            )

            # TODO: set ModelEvaluation properties for BP job, eval metrics
            for component in self._gca_resource.job_detail.task_details:
                for key in component.outputs:
                    if key == "batchpredictionjob":
                        batch_pred_mlmd_uri = component.outputs[key].artifacts[0].name
                        batch_pred_resource_uri = (
                            component.outputs[key].artifacts[0].metadata["resourceName"]
                        )
                    if key == "evaluation_metrics":
                        eval_metrics_mlmd_uri = component.outputs[key].artifacts[0].name
                        # eval_metrics_resource_uri = component.outputs[key].artifacts[0].metadata['resourceName'] # not available yet
