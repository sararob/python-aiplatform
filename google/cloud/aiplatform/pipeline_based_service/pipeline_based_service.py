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
import json
from google.auth import credentials as auth_credentials

from google.cloud.aiplatform import base
from google.cloud.aiplatform import utils
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform.utils import yaml_utils

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

_LOGGER = base.Logger(__name__)


class _VertexAiPipelineBasedService(base.VertexAiStatefulResource):
    """Base class for Vertex AI Pipeline based services."""

    client_class = utils.PipelineJobClientWithOverride
    _resource_noun = "pipelineJob"
    _delete_method = "delete_pipeline_job"
    _getter_method = "get_pipeline_job"
    _list_method = "list_pipeline_jobs"
    _parse_resource_name_method = "parse_pipeline_job_path"
    _format_resource_name_method = "pipeline_job_path"

    _valid_done_states = pipeline_jobs._PIPELINE_COMPLETE_STATES

    @property
    @classmethod
    @abc.abstractmethod
    def _template_ref(self) -> str:
        """The pipeline template URL for this service."""
        pass

    @property
    @abc.abstractmethod
    def metadata_output_artifact(self) -> Optional[str]:
        """The ML Metadata output artifact resource URI from the completed pipeline run."""
        pass

    @property
    def backing_pipeline_job(self) -> pipeline_jobs.PipelineJob:
        """The PipelineJob associated with the resource."""
        return pipeline_jobs.PipelineJob.get(resource_name=self.resource_name)

    @property
    def pipeline_console_uri(self) -> str:
        """The console URI of the PipelineJob created by the service."""
        if self.backing_pipeline_job:
            return self.backing_pipeline_job._dashboard_uri()

    @property
    def state(self) -> Optional[str]:
        """The state of the Pipeline run associated with the service."""
        if self.backing_pipeline_job:
            return self.backing_pipeline_job.state
        return None

    def _validate_pipeline_template_matches_service(
        self, pipeline_job: pipeline_jobs.PipelineJob
    ):
        """Utility function to validate that the passed in pipeline ID matches
        the template of the Pipeline Based Service.

        Raises:
            ValueError: if the provided pipeline ID doesn't match the pipeline service.

        """

        # TODO: figure out a better way to do this
        # pipeline_job_resource = pipeline_jobs.PipelineJob.get(
        #     resource_name=pipeline_job_id,
        # )
        service_pipeline_json = yaml_utils.load_yaml(self._template_ref)

        print(service_pipeline_json)

        current_pipeline_components = []
        template_ref_components = []

        # Adding this for TestModelEvaluationJob::test_init_model_evaluation_job
        # Not sure if it's needed
        if not pipeline_job.pipeline_spec:
            raise ValueError(
                f"The provided pipeline template is not compatible with {self.__class__.__name__}"
            )

        for order, component_name in enumerate(
            pipeline_job.pipeline_spec.get("components")
        ):
            current_pipeline_components.append(component_name)

        for comp in service_pipeline_json["pipelineSpec"]["components"]:
            template_ref_components.append(comp)

        if current_pipeline_components != template_ref_components:
            raise ValueError(
                f"The provided pipeline template is not compatible with {self.__class__.__name__}"
            )

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
        Raises:
            ValueError: if the pipeline template used in this PipelineJob is not consistent with the _template_ref defined on the subclass.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=pipeline_job_id,
        )

        job_resource = pipeline_jobs.PipelineJob.get(resource_name=pipeline_job_id)

        self._validate_pipeline_template_matches_service(job_resource)

        self._gca_resource = self._get_gca_resource(resource_name=pipeline_job_id)

    @classmethod
    def _create_and_submit_pipeline_job(
        cls,
        template_params: Dict[str, Any],
        pipeline_root: str,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        job_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "_VertexAiPipelineBasedService":
        """Create a new PipelineJob using the provided template and parameters.

        Args:
            template_ref (str):
                Required. The path of the compiled Pipeline JSON template file in the template artifact registry.
            template_artifacts (Dict[str, Any]) TODO: dependent on pipelines backend work
                Required. The MLMD artifact resources to pass to the given pipeline template.
            template_params (Dict[str, Any]):
                Required. The parameters to pass to the given pipeline template.
            pipeline_root (str)
                Required. The GCS directory to store the pipeline run output.
            service_account (str):
                Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
                If left unspecified, the job is not peered with any network.
            job_id (str):
                Optional. The unique ID of the job run.
                If not specified, pipeline name + timestamp will be used.
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

        # TODO: use cls._generate_display_name()
        service_name = cls.__name__.lower()

        self = cls._empty_constructor(
            project=project, location=location, credentials=credentials,
        )

        service_pipeline_job = pipeline_jobs.PipelineJob(
            display_name=service_name,
            template_path=self._template_ref,
            parameter_values=template_params,
            pipeline_root=pipeline_root,
            job_id=job_id,
            project=self.project,
            location=location,
            credentials=credentials,
        )

        service_pipeline_job.submit(
            service_account=service_account, network=network,
        )

        self._gca_resource = self._get_gca_resource(service_pipeline_job.resource_name)

        return self
