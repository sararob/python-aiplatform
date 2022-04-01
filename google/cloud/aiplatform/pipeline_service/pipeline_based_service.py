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

import abc
from re import template

from google.auth import credentials as auth_credentials
from google.protobuf import field_mask_pb2

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform import pipeline_jobs

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

class VertexAiPipelineBasedService(base.VertexAiResourceNounWithFutureManager):
    """Base class for Vertex AI Pipeline based services."""

    @property
    @abc.abstractmethod
    def backing_pipeline_job(self) -> pipeline_jobs.PipelineJob:
        """The PipelineJob associated with the resource."""
        pass

    @property
    @abc.abstractmethod
    def pipeline_console_uri(self) -> str:
        """The console URI of the PipelineJob created by the service."""
        pass

    @property
    @abc.abstractmethod
    def metadata_output_artifact(self) -> str:
        """The ML Metadata output artifact resource URI from the completed pipeline run."""
        pass

    @property
    def state(self) -> Optional[str]:
        """The state of the Pipeline run associated with the service."""
        if self.backing_pipeline_job:
            return self.backing_pipeline_job.state
        return None

    @classmethod
    @abc.abstractmethod
    def _template_ref(self) -> str:
        """The pipeline template URL for this service."""
        pass

    def __init__(
        self,
        pipeline_job_id: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing Pipeline Based Service given the ID of the pipeline execution.

        Example Usage:

            pipeline_service = aiplatform.VertexAiPipelineBasedService(
                pipeline_job_id = "projects/123/locations/us-central1/pipelines/runs/456"
            )

            pipeline_service = aiplatform.VertexAiPipelinebasedService(
                pipeline_job_id = "456"
            )

        Args:
            pipeline_job_id(str):
                Required. A fully-qualified pipeline job run ID.
                Example: "projects/123/locations/us-central1/pipelines/runs/456" or
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
            project=project,
            location=location,
            credentials=credentials,
            resource_name=pipeline_job_id,
        )

        self._gca_resource = self._get_gca_resource(resource_name=pipeline_job_id)

    def _create_pipeline_job(
        self,
        template_ref: str,
        template_params: Dict[str, Any],
        project: str,
        location: str,
        credentials: auth_credentials.Credentials,
    ) -> pipeline_jobs.PipelineJob:
        """Create a new PipelineJob using the provided template and parameters.

        Args:
            template_ref (str):
                Required. The path of the compiled Pipeline JSON template file in the template artifact registry.
            template_artifacts (Dict[str, Any]) TODO
                Required. The MLMD artifact resources to pass to the given pipeline template.
            template_params (Dict[str, Any]):
                Required. The parameters to pass to the given pipeline template.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create the PipelineJob.
                Overrides credentials set in aiplatform.init.
            project (str):
                Optional. The project to run this PipelineJob in. If not set,
                the project set in aiplatform.init will be used.
            location (str):
                Optional. Location to create PipelineJob. If not set,
                location set in aiplatform.init will be used.

        Returns:
            (VertexAiPipelineBasedService):
                Instantiated representation of a Vertex AI Pipeline based service.
        """

        # api_client = cls._instantiate_client(location=location, credentials=credentials)

        project = project or initializer.global_config.project
        location = location or initializer.global_config.location

        service_pipeline_job = pipeline_jobs.PipelineJob(
            display_name="service-test-pipeline-job",
            template_path=template_ref,
            parameter_values=template_params,
            project=project,
            location=location,
            credentials=credentials,
        )

        service_pipeline_job.submit()

        self.backing_pipeline_job = service_pipeline_job
        self.pipeline_console_uri = service_pipeline_job._dashboard_uri

        return service_pipeline_job