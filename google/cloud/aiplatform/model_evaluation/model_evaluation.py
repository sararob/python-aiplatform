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
from google.cloud import aiplatform
from google.protobuf import field_mask_pb2

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform import jobs

from google.cloud.aiplatform.utils import model_evaluation_utils

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)


class ModelEvaluation(base.VertexAiResourceNounWithFutureManager):

    client_class = utils.ModelClientWithOverride
    _resource_noun = "evaluations"
    # _delete_method = "delete_pipeline_job"
    _getter_method = "get_model_evaluation"
    _list_method = "list_model_evaluations"
    _parse_resource_name_method = "parse_model_evaluation_path"
    _format_resource_name_method = "model_evaluation_path"

    @property
    def evaluation_metrics(self) -> Optional[Dict[str, Any]]:
        """Gets the evaluation metrics from the Model Evaluation.
        Returns:
            A dict with model metrics created from the system.Metrics 
            pipeline output artifact. Returns None if the underlying 
            PipelineJob has not yet completed.
        """

    @property
    def batch_prediction_job(self) -> jobs.BatchPredictionJob:
        """The Batch Prediction job used for the Model Eval"""

    @property
    def backing_pipeline_job(self) -> pipeline_jobs.PipelineJob:
        """The PipelineJob resource that ran this model evaluation."""

    def get_from_pipeline_job(pipeline_id) -> "ModelEvaluation":
        """Creates a ModelEvaluation SDK resource from an evaluation pipeline that has already run on a managed Vertex model.
        Args:
            pipeline_id (str):
                Required. A fully-qualified pipeline job run ID.
                Example: "projects/123/locations/us-central1/pipelineJobs/456" or
                "456" when project and location are initialized or passed.

        Returns:
            A Vertex AI ModelEvaluation resource.
        Raises:
            ValueError: if the provided `pipeline_id` is not a ModelEvaluation pipeline.
        """
        pipeline_job_resource = pipeline_jobs.PipelineJob.get(resource_name=pipeline_id)

        if model_evaluation_utils._validate_model_evaluation_pipeline(
            pipeline_job_resource
        ):
            print("woohooo its an eval pipeline")

    def __init__(
        self,
        evaluation_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves the ModelEvaluation resource and instantiates its representation.

        Args:
            evaluation_name (str):
                Required. A fully-qualified model evaluation resource name or evaluation ID.
                Example: "projects/123/locations/us-central1/models/456/evaluations/789" or
                "789" when project and location are initialized or passed.
            project (str):
                Optional project to retrieve model evaluation from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional location to retrieve model evaluation from. If not set, location
                set in aiplatform.init will be used.
            credentials: Optional[auth_credentials.Credentials]=None,
                Custom credentials to use to retrieve this model evaluation. If not set,
                credentials set in aiplatform.init will be used.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=evaluation_name,
        )

        self._gca_resource = self._get_gca_resource(resource_name=evaluation_name)
