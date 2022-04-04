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

from re import template
from google.auth import credentials as auth_credentials
from google.protobuf import field_mask_pb2

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import models
from google.cloud.aiplatform import utils
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform import pipeline_service

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

from google.cloud.aiplatform_v1.types import model_evaluation

_LOGGER = base.Logger(__name__)

_MODEL_EVAL_PIPELINE_TEMPLATE = "/Users/sararob/Dev/sara-fork/python-aiplatform/google/cloud/aiplatform/model_evaluation/sdk_pipeline_experimental.json"

class ModelEvaluationJob(pipeline_service.VertexAiPipelineBasedService):

    @property
    def _template_ref(self) -> str:
        return _MODEL_EVAL_PIPELINE_TEMPLATE

    @property
    def backing_pipeline_job(self) -> pipeline_jobs.PipelineJob:
        return pipeline_jobs.PipelineJob.get(
            resource_name=self.resource_name
        )

    @property
    def pipeline_console_uri(self) -> str:
        return super().pipeline_console_uri

    @property
    def metadata_output_artifact(self) -> Optional[str]:
        return super().metadata_output_artifact
        # set if pipeline is complete

    def __init__(
        self,
        evaluation_pipeline_run: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves a ModelEvaluationJob and instantiates its representation."""

        super().__init__(
            pipeline_job_id=evaluation_pipeline_run,
        )

    @classmethod
    def submit(
        cls,
        model: models.Model,
        prediction_type: str,
        pipeline_root,
        gcs_source_uris: List[str],
        class_names: List[str],
        target_column_name: str,
        instances_format: Optional[str] = "jsonl",

    ) -> "ModelEvaluationJob":

        # TODO: format template params to macth with pipeline template
        template_params = {
            "batch_predict_gcs_source_uris": gcs_source_uris,
            "batch_predict_instances_format": instances_format,
            "class_names": class_names,
            "model_name": model.resource_name,
            "prediction_type": prediction_type,
            "project": initializer.global_config.project,
            "location": initializer.global_config.location,
            "root_dir": pipeline_root,
            "target_column_name": target_column_name,
        }
        
        eval_pipeline_run = cls._create_and_submit_pipeline_job(
            cls,
            template_ref=_MODEL_EVAL_PIPELINE_TEMPLATE,
            template_params=template_params,
            pipeline_root=pipeline_root,
            project = initializer.global_config.project,
            location = initializer.global_config.location,
            credentials = initializer.global_config.credentials,
        )

        model_eval_job_resource = cls.__new__(cls)
        
        model_eval_job_resource.backing_pipeline_job = eval_pipeline_run
        model_eval_job_resource.pipeline_console_uri = eval_pipeline_run._dashboard_uri

        return model_eval_job_resource

    def get_model_evaluation(
        self,

    ) -> Optional[model_evaluation.ModelEvaluation]:
        """Creates a ModelEvaluation resource and instantiates its representation."""
        print('hello')

        # check if backing job has completed
        # if not, return None
        # if yes, return the instantiated ModelEvaluation resource
            # get the pipeline output artifact (evaluation)
            # pass the resource name to the 
