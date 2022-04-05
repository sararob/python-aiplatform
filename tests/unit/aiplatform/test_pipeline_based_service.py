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

from datetime import datetime
import os
import pytest
import json

from unittest import mock
from unittest.mock import patch
from importlib import reload

from google.api_core import operation
from google.auth.exceptions import GoogleAuthError
from google.auth import credentials as auth_credentials

from google.cloud import storage

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer

# from google.protobuf import field_mask_pb2

from google.cloud.aiplatform_v1.services.pipeline_service import (
    client as pipeline_service_client_v1,
)
from google.cloud.aiplatform_v1.types import (
    pipeline_job as gca_pipeline_job_v1,
    pipeline_state as gca_pipeline_state_v1,
)

# pipeline job
_TEST_PROJECT = "test-project"
_TEST_LOCATION = "us-central1"
_TEST_PIPELINE_JOB_DISPLAY_NAME = "sample-pipeline-job-display-name"
_TEST_PIPELINE_JOB_ID = "sample-test-pipeline-202111111"
_TEST_GCS_BUCKET_NAME = "my-bucket"
_TEST_CREDENTIALS = auth_credentials.AnonymousCredentials()
_TEST_SERVICE_ACCOUNT = "abcde@my-project.iam.gserviceaccount.com"

_TEST_TEMPLATE_PATH = f"gs://{_TEST_GCS_BUCKET_NAME}/job_spec.json"
_TEST_PARENT = f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}"
_TEST_NETWORK = f"projects/{_TEST_PROJECT}/global/networks/{_TEST_PIPELINE_JOB_ID}"

_TEST_PIPELINE_JOB_NAME = f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/pipelineJobs/{_TEST_PIPELINE_JOB_ID}"

_TEST_PIPELINE_PARAMETER_VALUES_LEGACY = {"string_param": "hello"}
_TEST_PIPELINE_PARAMETER_VALUES = {
    "string_param": "hello world",
    "bool_param": True,
    "double_param": 12.34,
    "int_param": 5678,
    "list_int_param": [123, 456, 789],
    "list_string_param": ["lorem", "ipsum"],
    "struct_param": {"key1": 12345, "key2": 67890},
}

_TEST_PIPELINE_SPEC_LEGACY = {
    "pipelineInfo": {"name": "my-pipeline"},
    "root": {
        "dag": {"tasks": {}},
        "inputDefinitions": {"parameters": {"string_param": {"type": "STRING"}}},
    },
    "schemaVersion": "2.0.0",
    "components": {},
}
_TEST_PIPELINE_SPEC = {
    "pipelineInfo": {"name": "my-pipeline"},
    "root": {
        "dag": {"tasks": {}},
        "inputDefinitions": {
            "parameters": {
                "string_param": {"parameterType": "STRING"},
                "bool_param": {"parameterType": "BOOLEAN"},
                "double_param": {"parameterType": "NUMBER_DOUBLE"},
                "int_param": {"parameterType": "NUMBER_INTEGER"},
                "list_int_param": {"parameterType": "LIST"},
                "list_string_param": {"parameterType": "LIST"},
                "struct_param": {"parameterType": "STRUCT"},
            }
        },
    },
    "schemaVersion": "2.1.0",
    "components": {},
}

_TEST_PIPELINE_GET_METHOD_NAME = "get_fake_pipeline_job"
_TEST_PIPELINE_LIST_METHOD_NAME = "list_fake_pipeline_jobs"
_TEST_PIPELINE_CANCEL_METHOD_NAME = "cancel_fake_pipeline_job"
_TEST_PIPELINE_DELETE_METHOD_NAME = "delete_fake_pipeline_job"
_TEST_PIPELINE_RESOURCE_NAME = (
    f"{_TEST_PARENT}/fakePipelineJobs/{_TEST_PIPELINE_JOB_ID}"
)
_TEST_PIPELINE_CREATE_TIME = datetime.now()

# pipeline based service
# _TEST_TEMPLATE_REF

# TODO mocks

@pytest.fixture
def mock_pipeline_service_create():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "create_pipeline_job"
    ) as mock_create_pipeline_job:
        mock_create_pipeline_job.return_value = gca_pipeline_job_v1.PipelineJob(
            name=_TEST_PIPELINE_JOB_NAME,
            state=gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED,
            create_time=_TEST_PIPELINE_CREATE_TIME,
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
        )
        yield mock_create_pipeline_job


def make_pipeline_job(state):
    return gca_pipeline_job_v1.PipelineJob(
        name=_TEST_PIPELINE_JOB_NAME,
        state=state,
        create_time=_TEST_PIPELINE_CREATE_TIME,
        service_account=_TEST_SERVICE_ACCOUNT,
        network=_TEST_NETWORK,
    )

@pytest.fixture
def mock_pipeline_service_get():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "get_pipeline_job"
    ) as mock_get_pipeline_job:
        mock_get_pipeline_job.side_effect = [
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_RUNNING
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
            ),
        ]

        yield mock_get_pipeline_job


@pytest.fixture
def mock_pipeline_service_get_with_fail():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "get_pipeline_job"
    ) as mock_get_pipeline_job:
        mock_get_pipeline_job.side_effect = [
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_RUNNING
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_RUNNING
            ),
            make_pipeline_job(
                gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_FAILED
            ),
        ]

        yield mock_get_pipeline_job

@pytest.fixture
def mock_load_json(job_spec_json):
    with patch.object(storage.Blob, "download_as_bytes") as mock_load_json:
        mock_load_json.return_value = json.dumps(job_spec_json).encode()
        yield mock_load_json

class TestPipelineBasedService:

    @pytest.mark.parametrize(
        "pipeline_name", [_TEST_PIPELINE_JOB_ID, _TEST_PIPELINE_JOB_NAME]
    )
    def test_init_pipeline_based_service(self, pipeline_name, mock_pipeline_service_get):
        aiplatform.init(project=_TEST_PROJECT)

        my_pipeline_based_service = aiplatform.VertexAiPipelineBasedService(
            pipeline_job_id=pipeline_name,
            _template_ref=_TEST_TEMPLATE_PATH,
        )

        # my_featurestore = aiplatform.Featurestore(featurestore_name=featurestore_name)

        mock_pipeline_service_get.assert_called_once_with(
            name=my_pipeline_based_service.resource_name,
        )


    # TODO: test_init_with_invalid_pipeline_run_id

    # TODO: test_create_and_submit_pipeline_job

    # TODO: test_create_and_submit_pipeline_job_with_invalid_params_raises