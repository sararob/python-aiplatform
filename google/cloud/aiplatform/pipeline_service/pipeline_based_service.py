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
from datetime import datetime

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

from google.cloud.aiplatform.compat.types import (
    pipeline_job_v1 as gca_pipeline_job_v1,
    pipeline_state_v1 as gca_pipeline_state_v1,
)

_LOGGER = base.Logger(__name__)

_PIPELINE_COMPLETE_STATES = set(
    [
        gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED,
        gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_FAILED,
        gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_CANCELLED,
        gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_PAUSED,
    ]
)


class VertexAiPipelineBasedService(base.VertexAiStatefulResource):
    """Base class for Vertex AI Pipeline based services."""

    client_class = utils.PipelineJobClientWithOverride
    _resource_noun = "pipelineJob"
    _delete_method = "delete_pipeline_job"
    _getter_method = "get_pipeline_job"
    _list_method = "list_pipeline_jobs"
    _parse_resource_name_method = "parse_pipeline_job_path"
    _format_resource_name_method = "pipeline_job_path"

    _valid_done_states = _PIPELINE_COMPLETE_STATES

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
    def metadata_output_artifact(self) -> Optional[str]:
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
                pipeline_job_id = "projects/123/locations/us-central1/pipelinesJobs/456"
            )

            pipeline_service = aiplatform.VertexAiPipelinebasedService(
                pipeline_job_id = "456"
            )

        Args:
            pipeline_job_id(str):
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
            project=project,
            location=location,
            credentials=credentials,
            resource_name=pipeline_job_id,
        )
        self._gca_resource = self._get_gca_resource(resource_name=pipeline_job_id)

    @classmethod
    def _create_and_submit_pipeline_job(
        cls,
        template_params: Dict[str, Any],
        pipeline_root: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "VertexAiPipelineBasedService":
        """Create a new PipelineJob using the provided template and parameters.

        Args:
            template_ref (str):
                Required. The path of the compiled Pipeline JSON template file in the template artifact registry.
            template_artifacts (Dict[str, Any]) TODO
                Required. The MLMD artifact resources to pass to the given pipeline template.
            template_params (Dict[str, Any]):
                Required. The parameters to pass to the given pipeline template.
            pipeline_root (str)
                Required. The GCS directory to store the pipeline run output.
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

        self = cls._empty_constructor(
            project=project, location=location, credentials=credentials,
        )

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # TODO: add the name of the service to display_name
        service_pipeline_job = pipeline_jobs.PipelineJob(
            display_name=f"pipeline-service-job-{timestamp}",
            template_path=self._template_ref,
            parameter_values=template_params,
            pipeline_root=pipeline_root,
            project=self.project,
            location=location,
            credentials=credentials,
        )

        service_pipeline_job.submit()

        self._gca_resource = self._get_gca_resource(service_pipeline_job.resource_name)

        return self