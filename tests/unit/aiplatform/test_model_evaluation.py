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

import pytest
import yaml
import json
from google.protobuf import json_format

from unittest import mock
from unittest.mock import patch
from datetime import datetime
from google.auth import credentials as auth_credentials

from google.cloud import storage

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import models
from google.cloud.aiplatform import model_evaluation

from google.cloud.aiplatform_v1.services.model_service import (
    client as model_service_client,
)

from google.cloud.aiplatform_v1.services.pipeline_service import (
    client as pipeline_service_client_v1,
)
from google.cloud.aiplatform_v1.types import (
    pipeline_job as gca_pipeline_job_v1,
    pipeline_state as gca_pipeline_state_v1,
)

from google.cloud.aiplatform.compat.types import model as gca_model

from google.cloud.aiplatform_v1.types import model_evaluation as gca_model_evaluation

_TEST_PROJECT = "test-project"
_TEST_LOCATION = "us-central1"
_TEST_MODEL_NAME = "test-model"
_TEST_MODEL_ID = "1028944691210842416"
_TEST_EVAL_ID = "1028944691210842622"

_TEST_MODEL_RESOURCE_NAME = model_service_client.ModelServiceClient.model_path(
    _TEST_PROJECT, _TEST_LOCATION, _TEST_MODEL_ID
)

_TEST_MODEL_EVAL_RESOURCE_NAME = model_service_client.ModelServiceClient.model_evaluation_path(
    _TEST_PROJECT, _TEST_LOCATION, _TEST_MODEL_ID, _TEST_EVAL_ID,
)

_TEST_MODEL_EVAL_METRICS = {
    "auPrc": 0.80592036,
    "auRoc": 0.8100363,
    "logLoss": 0.53061414,
    "confidenceMetrics": [
        {
            "confidenceThreshold": -0.01,
            "recall": 1.0,
            "precision": 0.5,
            "falsePositiveRate": 1.0,
            "f1Score": 0.6666667,
            "recallAt1": 1.0,
            "precisionAt1": 0.5,
            "falsePositiveRateAt1": 1.0,
            "f1ScoreAt1": 0.6666667,
            "truePositiveCount": "415",
            "falsePositiveCount": "415",
        },
        {
            "recall": 1.0,
            "precision": 0.5,
            "falsePositiveRate": 1.0,
            "f1Score": 0.6666667,
            "recallAt1": 0.74216866,
            "precisionAt1": 0.74216866,
            "falsePositiveRateAt1": 0.25783134,
            "f1ScoreAt1": 0.74216866,
            "truePositiveCount": "415",
            "falsePositiveCount": "415",
        },
    ],
}

# pipeline job
_TEST_ID = "1028944691210842416"
_TEST_PIPELINE_JOB_DISPLAY_NAME = "sample-pipeline-job-display-name"
_TEST_PIPELINE_JOB_ID = "sample-test-pipeline-202111111"
_TEST_GCS_BUCKET_NAME = "my-bucket"
_TEST_CREDENTIALS = auth_credentials.AnonymousCredentials()
_TEST_SERVICE_ACCOUNT = "abcde@my-project.iam.gserviceaccount.com"
_TEST_PIPELINE_ROOT = f"gs://{_TEST_GCS_BUCKET_NAME}/pipeline_root"
_TEST_PIPELINE_CREATE_TIME = datetime.now()

_TEST_TEMPLATE_PATH = f"gs://{_TEST_GCS_BUCKET_NAME}/job_spec.json"
_TEST_PARENT = f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}"
_TEST_NETWORK = f"projects/{_TEST_PROJECT}/global/networks/{_TEST_PIPELINE_JOB_ID}"

_TEST_PIPELINE_JOB_NAME = f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/pipelineJobs/{_TEST_PIPELINE_JOB_ID}"
_TEST_INVALID_PIPELINE_JOB_NAME = (
    f"prj/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/{_TEST_PIPELINE_JOB_ID}"
)
_TEST_MODEL_EVAL_JOB_DISPLAY_NAME = "test-eval-job"

_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES = {
    "batch_predict_gcs_source_uris": ["gs://my-bucket/my-prediction-data.csv"],
    "batch_predict_instances_format": "csv",
    "class_names": ["0", "1"],
    "model_name": _TEST_MODEL_RESOURCE_NAME,
    "prediction_type": "classification",
    "project": _TEST_PROJECT,
    "location": _TEST_LOCATION,
    "root_dir": _TEST_PIPELINE_ROOT,
    "target_column_name": "predict_class",
}


_TEST_MODEL_EVAL_PIPELINE_SPEC = {
    "pipelineInfo": {"name": "evaluation-sdk-pipeline"},
    "root": {
        "dag": {"tasks": {}},
        "inputDefinitions": {
            "parameters": {
                "batch_predict_gcs_source_uris": {"type": "STRING"},
                "batch_predict_instances_format": {"type": "STRING"},
                "class_names": {"type": "STRING"},
                "model_name": {"type": "STRING"},
                "prediction_type": {"type": "STRING"},
                "project": {"type": "STRING"},
                "location": {"type": "STRING"},
                "root_dir": {"type": "STRING"},
                "target_column_name": {"type": "STRING"},
            }
        },
    },
    "schemaVersion": "2.1.0",
    "sdkVersion": "kfp-1.8.12",
    "components": {}
}

_TEST_INVALID_MODEL_EVAL_PIPELINE_SPEC = {
    "pipelineInfo": {"name": "my-pipeline"},
    "root": {
        "dag": {"tasks": {}},
        "inputDefinitions": {
            "parameters": {
                "batch_predict_gcs_source_uris": {"type": "STRING"},
                "batch_predict_instances_format": {"type": "STRING"},
                "class_names": {"type": "STRING"},
                "model_name": {"type": "STRING"},
                "prediction_type": {"type": "STRING"},
                "project": {"type": "STRING"},
                "location": {"type": "STRING"},
                "root_dir": {"type": "STRING"},
                "target_column_name": {"type": "STRING"},
            }
        },
    },
    "schemaVersion": "2.0.0",
    "sdkVersion": "kfp-1.8.12",
    "components": {},
}

_TEST_MODEL_EVAL_PIPELINE_SPEC_JSON = json.dumps(
    {
        "pipelineInfo": {"name": "evaluation-sdk-pipeline"},
        "root": {
            "dag": {"tasks": {}},
            "inputDefinitions": {
                "parameters": {
                    "batch_predict_gcs_source_uris": {"type": "STRING"},
                    "batch_predict_instances_format": {"type": "STRING"},
                    "class_names": {"type": "STRING"},
                    "model_name": {"type": "STRING"},
                    "prediction_type": {"type": "STRING"},
                    "project": {"type": "STRING"},
                    "location": {"type": "STRING"},
                    "root_dir": {"type": "STRING"},
                    "target_column_name": {"type": "STRING"},
                }
            },
        },
        "schemaVersion": "2.1.0",
        "sdkVersion": "kfp-1.8.12",
        "components": {}
    }
)


_TEST_MODEL_EVAL_PIPELINE_JOB = json.dumps(
    {
        "runtimeConfig": {
            "parameterValues": _TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES
        },
        "pipelineSpec": json.loads(_TEST_MODEL_EVAL_PIPELINE_SPEC_JSON),
    }
)

_TEST_INVALID_MODEL_EVAL_PIPELINE_SPEC_JSON = json.dumps({
    "pipelineInfo": {"name": "my-pipeline"},
    "root": {
        "dag": {"tasks": {}},
        "inputDefinitions": {
            "parameters": {
                "batch_predict_gcs_source_uris": {"type": "STRING"},
                "batch_predict_instances_format": {"type": "STRING"},
                "class_names": {"type": "STRING"},
                "model_name": {"type": "STRING"},
                "prediction_type": {"type": "STRING"},
                "project": {"type": "STRING"},
                "location": {"type": "STRING"},
                "root_dir": {"type": "STRING"},
                "target_column_name": {"type": "STRING"},
            }
        },
    },
    "schemaVersion": "2.0.0",
    "sdkVersion": "kfp-1.8.12",
    "components": {},
})

_TEST_INVALID_MODEL_EVAL_PIPELINE_JOB = json.dumps(
    {
        "runtimeConfig": {
            "parameterValues": _TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES
        },
        "pipelineSpec": json.loads(_TEST_INVALID_MODEL_EVAL_PIPELINE_SPEC_JSON),
    }
)

_TEST_EVAL_RESOURCE_NAME = f"projects/{_TEST_ID}/locations/{_TEST_LOCATION}/models/{_TEST_ID}/evaluations/{_TEST_ID}"


@pytest.fixture
def get_model_mock():
    with mock.patch.object(
        model_service_client.ModelServiceClient, "get_model"
    ) as get_model_mock:
        get_model_mock.return_value = gca_model.Model(
            display_name=_TEST_MODEL_NAME, name=_TEST_MODEL_RESOURCE_NAME,
        )

        yield get_model_mock


@pytest.fixture
def mock_model():
    model = mock.MagicMock(models.Model)
    model.name = _TEST_MODEL_ID
    model._latest_future = None
    model._exception = None
    model._gca_resource = gca_model.Model(
        display_name="test-eval-model",
        description="This is the mock Model's description",
        name=_TEST_MODEL_NAME,
    )
    yield model


# ModelEvaluation mocks
@pytest.fixture
def mock_model_eval_get():
    with mock.patch.object(
        model_service_client.ModelServiceClient, "get_model_evaluation"
    ) as mock_get_model_eval:
        mock_get_model_eval.return_value = gca_model_evaluation.ModelEvaluation(
            name=_TEST_MODEL_EVAL_RESOURCE_NAME, metrics=_TEST_MODEL_EVAL_METRICS,
        )
        yield mock_get_model_eval


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
def mock_pipeline_service_get_pending():
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
        ]

        yield mock_get_pipeline_job


@pytest.fixture
def mock_load_json(job_spec_json):
    with patch.object(storage.Blob, "download_as_bytes") as mock_load_json:
        mock_load_json.return_value = json.dumps(job_spec_json).encode()
        yield mock_load_json


@pytest.fixture
def mock_load_yaml_and_json(job_spec):
    with patch.object(storage.Blob, "download_as_bytes") as mock_load_yaml_and_json:
        mock_load_yaml_and_json.return_value = job_spec.encode()
        yield mock_load_yaml_and_json


@pytest.fixture
def mock_model_eval_get():
    with mock.patch.object(
        model_service_client.ModelServiceClient, "get_model_evaluation"
    ) as mock_get_model_eval:
        mock_get_model_eval.return_value = gca_model_evaluation.ModelEvaluation(
            name=_TEST_EVAL_RESOURCE_NAME, metrics=_TEST_MODEL_EVAL_METRICS,
        )
        yield mock_get_model_eval

@pytest.fixture
def mock_model_eval_job_get():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "get_pipeline_job"
    ) as mock_get_model_eval_job:
        mock_get_model_eval_job.return_value = gca_pipeline_job_v1.PipelineJob(
            name=_TEST_PIPELINE_JOB_NAME,
            state=gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED,
            create_time=_TEST_PIPELINE_CREATE_TIME,
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
            pipeline_spec=_TEST_MODEL_EVAL_PIPELINE_SPEC,
        )
        yield mock_get_model_eval_job

@pytest.fixture
def mock_invalid_model_eval_job_get():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "get_pipeline_job"
    ) as mock_get_model_eval_job:
        mock_get_model_eval_job.return_value = gca_pipeline_job_v1.PipelineJob(
            name=_TEST_PIPELINE_JOB_NAME,
            state=gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED,
            create_time=_TEST_PIPELINE_CREATE_TIME,
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
            pipeline_spec=_TEST_INVALID_MODEL_EVAL_PIPELINE_SPEC,
        )
        yield mock_get_model_eval_job

@pytest.fixture
def mock_model_eval_job_create():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "create_pipeline_job"
    ) as mock_create_model_eval_job:
        mock_create_model_eval_job.return_value = gca_pipeline_job_v1.PipelineJob(
            name=_TEST_PIPELINE_JOB_NAME,
            state=gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED,
            create_time=_TEST_PIPELINE_CREATE_TIME,
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
            pipeline_spec=_TEST_MODEL_EVAL_PIPELINE_SPEC, # this should be a protobuf, use json_utils parsing method, compare both dicts
        )
        yield mock_create_model_eval_job


@pytest.fixture
def mock_model_eval_job_get():
    with mock.patch.object(
        pipeline_service_client_v1.PipelineServiceClient, "get_pipeline_job"
    ) as mock_get_model_eval_job:
        mock_get_model_eval_job.return_value = gca_pipeline_job_v1.PipelineJob(
            name=_TEST_PIPELINE_JOB_NAME,
            state=gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED,
            create_time=_TEST_PIPELINE_CREATE_TIME,
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
            pipeline_spec=_TEST_MODEL_EVAL_PIPELINE_SPEC,
        )
        yield mock_get_model_eval_job


class TestModelEvaluation:
    def test_init_model_evaluation_with_only_resource_name(self, mock_model_eval_get):
        aiplatform.init(project=_TEST_PROJECT)

        aiplatform.ModelEvaluation(evaluation_name=_TEST_MODEL_EVAL_RESOURCE_NAME)

        mock_model_eval_get.assert_called_once_with(
            name=_TEST_MODEL_EVAL_RESOURCE_NAME, retry=base._DEFAULT_RETRY
        )

    def test_init_model_evaluation_with_eval_id_and_model_id(self, mock_model_eval_get):
        aiplatform.init(project=_TEST_PROJECT)

        aiplatform.ModelEvaluation(
            evaluation_name=_TEST_EVAL_ID, model_id=_TEST_MODEL_ID
        )

        mock_model_eval_get.assert_called_once_with(
            name=_TEST_MODEL_EVAL_RESOURCE_NAME, retry=base._DEFAULT_RETRY
        )

    def test_init_model_evaluatin_with_id_project_and_location(
        self, mock_model_eval_get
    ):
        aiplatform.init(project=_TEST_PROJECT)

        aiplatform.ModelEvaluation(
            evaluation_name=_TEST_MODEL_EVAL_RESOURCE_NAME,
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
        )
        mock_model_eval_get.assert_called_once_with(
            name=_TEST_MODEL_EVAL_RESOURCE_NAME, retry=base._DEFAULT_RETRY
        )

    def test_init_model_evaluation_with_invalid_evaluation_resource_raises(
        self, mock_model_eval_get
    ):
        aiplatform.init(project=_TEST_PROJECT)

        with pytest.raises(ValueError):
            aiplatform.ModelEvaluation(evaluation_name=_TEST_MODEL_RESOURCE_NAME)

    def test_get_model_evaluation_metrics(self, mock_model_eval_get):
        aiplatform.init(project=_TEST_PROJECT)

        eval_metrics = aiplatform.ModelEvaluation(
            evaluation_name=_TEST_MODEL_EVAL_RESOURCE_NAME
        ).metrics
        assert eval_metrics == _TEST_MODEL_EVAL_METRICS

    def test_no_delete_model_evaluation_method(self, mock_model_eval_get):

        my_eval = aiplatform.ModelEvaluation(
            evaluation_name=_TEST_MODEL_EVAL_RESOURCE_NAME
        )

        with pytest.raises(NotImplementedError):
            my_eval.delete()


class TestModelEvaluationJob:
    class FakeModelEvaluationJob(model_evaluation.ModelEvaluationJob):
        _template_ref = _TEST_TEMPLATE_PATH
        metadat_output_artifact = "TODO"

    @pytest.mark.parametrize(
        "job_spec", [_TEST_MODEL_EVAL_PIPELINE_JOB],
    )
    def test_init_model_evaluation_job(
        self,
        mock_pipeline_service_create,
        job_spec,
        mock_load_yaml_and_json,
        mock_model,
        get_model_mock,
        mock_model_eval_job_get,
        mock_model_eval_job_create,
    ):
        aiplatform.init(project=_TEST_PROJECT)

        self.FakeModelEvaluationJob(
            evaluation_pipeline_run=_TEST_PIPELINE_JOB_NAME
        )

        mock_model_eval_job_get.assert_called_once_with(
            name=_TEST_PIPELINE_JOB_NAME, retry=base._DEFAULT_RETRY
        )

    @pytest.mark.parametrize(
        "job_spec", [_TEST_INVALID_MODEL_EVAL_PIPELINE_JOB],
    )
    def test_init_model_evaluation_job_with_invalid_eval_template_raises(
        self,
        mock_pipeline_service_create,
        job_spec,
        mock_load_yaml_and_json,
        mock_model,
        get_model_mock,
        mock_model_eval_job_get,
        mock_model_eval_job_create,
    ):
        aiplatform.init(project=_TEST_PROJECT)
        with pytest.raises(ValueError):
            self.FakeModelEvaluationJob(
                evaluation_pipeline_run=_TEST_PIPELINE_JOB_NAME
            )

    def test_init_model_evaluation_job_with_invalid_pipeline_job_name_raises(
        self, mock_pipeline_service_get,
    ):
        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            credentials=_TEST_CREDENTIALS,
        )

        with pytest.raises(ValueError):
            model_evaluation.ModelEvaluationJob(
                evaluation_pipeline_run=_TEST_INVALID_PIPELINE_JOB_NAME,
            )

    @pytest.mark.parametrize(
        "job_spec", [_TEST_MODEL_EVAL_PIPELINE_SPEC_JSON],
    )
    def test_model_evaluation_job_submit(
        self,
        mock_pipeline_service_create,
        job_spec,
        mock_load_yaml_and_json,
        mock_model,
        get_model_mock,
        mock_model_eval_job_get,
        mock_model_eval_job_create,
    ):
        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            credentials=_TEST_CREDENTIALS,
            staging_bucket=_TEST_GCS_BUCKET_NAME,
        )

        test_model_eval_job = self.FakeModelEvaluationJob.submit(
            model_name=_TEST_MODEL_RESOURCE_NAME,
            prediction_type=_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES[
                "prediction_type"
            ],
            pipeline_root=_TEST_GCS_BUCKET_NAME,
            target_column_name=_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES[
                "target_column_name"
            ],
            display_name=_TEST_MODEL_EVAL_JOB_DISPLAY_NAME,
            gcs_source_uris=_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES[
                "batch_predict_gcs_source_uris"
            ],
            class_names=_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES["class_names"],
            instances_format=_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES[
                "batch_predict_instances_format"
            ],
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
        )

        test_model_eval_job.wait()


        expected_runtime_config_dict = {
            "gcsOutputDirectory": _TEST_GCS_BUCKET_NAME,
            "parameterValues": _TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES,
        }

        runtime_config = gca_pipeline_job_v1.PipelineJob.RuntimeConfig()._pb
        json_format.ParseDict(expected_runtime_config_dict, runtime_config)

        job_spec = yaml.safe_load(job_spec)
        pipeline_spec = job_spec.get("pipelineSpec") or job_spec

        # Construct expected request
        expected_gapic_pipeline_job = gca_pipeline_job_v1.PipelineJob(
            display_name=_TEST_MODEL_EVAL_JOB_DISPLAY_NAME,
            pipeline_spec={
                "components": {},
                "pipelineInfo": pipeline_spec["pipelineInfo"],
                "root": pipeline_spec["root"],
                "schemaVersion": "2.1.0",
                "sdkVersion": "kfp-1.8.12",
            },
            runtime_config=runtime_config,
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
        )

        # print('hiii',expected_gapic_pipeline_job)

        test_job_create_time_str = _TEST_PIPELINE_CREATE_TIME.strftime("%Y%m%d%H%M%S")

        mock_model_eval_job_create.assert_called_with(
            parent=_TEST_PARENT,
            # display_name=_TEST_MODEL_EVAL_JOB_DISPLAY_NAME,
            pipeline_job=expected_gapic_pipeline_job,
            pipeline_job_id=f"evaluation-sdk-pipeline-{test_job_create_time_str}",
            # service_account=_TEST_SERVICE_ACCOUNT,
            # network=_TEST_NETWORK,
            timeout=None,
        )

    # TODO: test_model_evaluation_job_submit_with_invalid_*

    # TODO: test_model_evaluation_job_get_model_evaluation_with_failed_pipeline_run_raises

    # TODO: test_model_evaluation_job_get_model_evaluation_with_pending_pipeline_run

    # TODO: test_model_evaluation_job_get_model_evaluation_with_successful_pipeline_run
