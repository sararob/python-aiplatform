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
from google.protobuf import field_mask_pb2

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform import jobs

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
        """Creates a ModelEvaluation SDK resource from an evaluation pipeline that has already run on a managed Vertex model."""

    def __init__(
        self,
        evaluation_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=evaluation_name,
        )

        self._gca_resource = self._get_gca_resource(resource_name=evaluation_name)

    @classmethod
    def get(
        cls,
        resource_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "ModelEvaluation":
        """Get a Vertex AI Model Evaluation given the resource name.

        Args:
            resource_name (str):
                Required. A fully-qualified resource name or ID.
            project (str):
                Optional. Project to retrieve evaluation from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve evaluation from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to get this evaluation.
                Overrides credentials set in aiplatform.init.

        Returns:
            A Vertex AI Model Evaluation.
        """
        self = cls._empty_constructor(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=resource_name,
        )

        self._gca_resource = self._get_gca_resource(resource_name=resource_name)

        return self