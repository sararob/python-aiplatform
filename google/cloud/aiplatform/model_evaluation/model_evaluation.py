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
from google.cloud.aiplatform import jobs
from google.cloud.aiplatform import base
from google.cloud.aiplatform import utils
from google.cloud.aiplatform import models

from typing import Dict, Optional


class ModelEvaluation(base.VertexAiResourceNounWithFutureManager):

    client_class = utils.ModelClientWithOverride
    _resource_noun = "evaluations"
    _delete_method = None
    _getter_method = "get_model_evaluation"
    _list_method = "list_model_evaluations"
    _parse_resource_name_method = "parse_model_evaluation_path"
    _format_resource_name_method = "model_evaluation_path"

    @property
    def metrics(self) -> Optional[Dict]:
        """Gets the evaluation metrics from the Model Evaluation.
        Returns:
            A dict with model metrics created from the Model Evaluation or
            None if the metrics for this evaluation are empty.
        """
        if self._gca_resource.metrics:
            return self.to_dict()["metrics"]

    @property
    def _backing_pipeline_job(self) -> Optional[str]:
        """The id of the managed pipeline for this model evaluation job.

        Returns:
            The pipeline job ID if this evaluation ran from a managed pipeline or None.
        """
        if self._gca_resource.metadata["pipeline_job_id"]:
            return self._gca_resource.metadata["pipeline_job_id"]

    @property
    def _batch_prediction_job(self) -> Optional[jobs.BatchPredictionJob]:
        """The batch prediction job used for this evaluation if this ran as a
        Model Evaluation pipeline.
        Returns:
            An instantiated representation of the Batch Prediction Job if it exists.
        """
        # TODO: implement when self.backing_pipeline_job is available
        raise NotImplementedError

    @property
    def _metadata_output_artifact(self) -> Optional[str]:
        """The MLMD metadata artifact uri created by the Model Evaluation pipeline.

        Returns:
            The MLMD uri string if this Model Evaluation was created from a pipeline run.
        """
        # TODO: implement when self.backing_pipeline_job is available
        raise NotImplementedError

    def __init__(
        self,
        evaluation_name: str,
        model_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves the ModelEvaluation resource and instantiates its representation.

        Args:
            evaluation_name (str):
                Required. A fully-qualified model evaluation resource name or evaluation ID.
                Example: "projects/123/locations/us-central1/models/456/evaluations/789" or
                "789". If passing only the evaluation ID, model_id must be provided.
            model_id (str):
                Optional. The ID of the model to retrieve this evaluation from. If passing
                only the evaluation ID as evaluation_name, model_id must be provided.
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

        self._gca_resource = self._get_gca_resource(
            resource_name=evaluation_name,
            parent_resource_name_fields={models.Model._resource_noun: model_id}
            if model_id
            else model_id,
        )

    def delete(self):
        raise NotImplementedError(
            "Deleting a model evaluation has not been implemented yet."
        )
