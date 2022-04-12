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

import yaml
import datetime
import os
from typing import Type
from google.cloud.aiplatform.model_evaluation import model_evaluation_job
import pytest
import json

from unittest import mock
from unittest.mock import patch
from importlib import reload

from google.api_core import operation
from google.auth import credentials as auth_credentials
from google.protobuf import json_format
from google.cloud import storage

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import models
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import pipeline_based_service
from google.cloud.aiplatform import pipeline_jobs

from test_utils import json_file

from google.cloud.aiplatform_v1.services.pipeline_service import (
    client as pipeline_service_client_v1,
)
from google.cloud.aiplatform_v1.types import (
    pipeline_job as gca_pipeline_job_v1,
    pipeline_state as gca_pipeline_state_v1,
)

from google.cloud.aiplatform.pipeline_based_service.pipeline_based_service import (
    _VertexAiPipelineBasedService,
)

from google.cloud.aiplatform.model_evaluation import ModelEvaluation

from google.cloud.aiplatform.model_evaluation.model_evaluation_job import (
    ModelEvaluationJob,
)

from google.cloud.aiplatform_v1.services.model_service import (
    client as model_service_client,
)

from google.cloud.aiplatform.compat.types import model as gca_model

from google.cloud.aiplatform_v1.types import (
    encryption_spec as gca_encryption_spec,
    model_evaluation as gca_model_evaluation,
    model_evaluation_slice as gca_model_evaluation_slice,
    io as gca_io,
    types as gca_types,
)

_TEST_API_CLIENT = pipeline_service_client_v1.PipelineServiceClient

# pipeline job
_TEST_PROJECT = "test-project"
_TEST_LOCATION = "us-central1"
_TEST_PIPELINE_JOB_DISPLAY_NAME = "sample-pipeline-job-display-name"
_TEST_PIPELINE_JOB_ID = "sample-test-pipeline-202111111"
_TEST_GCS_BUCKET_NAME = "my-bucket"
_TEST_CREDENTIALS = auth_credentials.AnonymousCredentials()
_TEST_SERVICE_ACCOUNT = "abcde@my-project.iam.gserviceaccount.com"

_TEST_ID = "1028944691210842416"

_TEST_MODEL_RESOURCE_NAME = model_service_client.ModelServiceClient.model_path(
    _TEST_PROJECT, _TEST_LOCATION, _TEST_ID
)

# _TEST_MODEL = models.Model(
#     model_name=_TEST_MODEL_RESOURCE_NAME,
# )


_TEST_TEMPLATE_PATH = f"gs://{_TEST_GCS_BUCKET_NAME}/job_spec.json"
_TEST_PIPELINE_ROOT = f"gs://{_TEST_GCS_BUCKET_NAME}/pipeline_root"
_TEST_PARENT = f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}"
_TEST_NETWORK = f"projects/{_TEST_PROJECT}/global/networks/{_TEST_PIPELINE_JOB_ID}"

_TEST_MODEL_NAME = "test-model"

_TEST_PIPELINE_JOB_NAME = f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/pipelineJobs/{_TEST_PIPELINE_JOB_ID}"
_TEST_INVALID_PIPELINE_JOB_NAME = (
    f"prj/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/{_TEST_PIPELINE_JOB_ID}"
)
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
    "schemaVersion": "2.0.0",
    "sdkVersion": "kfp-1.8.12",
    "components": {
        "comp-importer": {
            "executorLabel": "exec-importer",
            "inputDefinitions": {"parameters": {"uri": {"type": "STRING"}}},
            "outputDefinitions": {
                "artifacts": {
                    "artifact": {
                        "artifactType": {
                            "schemaTitle": "google.VertexModel",
                            "schemaVersion": "0.0.1",
                        }
                    }
                }
            },
        },
        "comp-model-batch-predict": {
            "executorLabel": "exec-model-batch-predict",
            "inputDefinitions": {
                "artifacts": {
                    "model": {
                        "artifactType": {
                            "schemaTitle": "google.VertexModel",
                            "schemaVersion": "0.0.1",
                        }
                    }
                },
                "parameters": {
                    "accelerator_count": {"type": "INT"},
                    "accelerator_type": {"type": "STRING"},
                    "bigquery_destination_output_uri": {"type": "STRING"},
                    "encryption_spec_key_name": {"type": "STRING"},
                    "explanation_metadata": {"type": "STRING"},
                    "explanation_parameters": {"type": "STRING"},
                    "gcs_destination_output_uri_prefix": {"type": "STRING"},
                    "gcs_source_uris": {"type": "STRING"},
                    "generate_explanation": {"type": "STRING"},
                    "instances_format": {"type": "STRING"},
                    "job_display_name": {"type": "STRING"},
                    "labels": {"type": "STRING"},
                    "location": {"type": "STRING"},
                    "machine_type": {"type": "STRING"},
                    "manual_batch_tuning_parameters_batch_size": {"type": "INT"},
                    "max_replica_count": {"type": "INT"},
                    "model_parameters": {"type": "STRING"},
                    "predictions_format": {"type": "STRING"},
                    "project": {"type": "STRING"},
                    "starting_replica_count": {"type": "INT"},
                },
            },
            "outputDefinitions": {
                "artifacts": {
                    "batchpredictionjob": {
                        "artifactType": {
                            "schemaTitle": "google.VertexBatchPredictionJob",
                            "schemaVersion": "0.0.1",
                        }
                    }
                },
                "parameters": {"gcp_resources": {"type": "STRING"}},
            },
        },
        "comp-model-evaluation": {
            "executorLabel": "exec-model-evaluation",
            "inputDefinitions": {
                "artifacts": {
                    "batch_prediction_job": {
                        "artifactType": {
                            "schemaTitle": "google.VertexBatchPredictionJob",
                            "schemaVersion": "0.0.1",
                        }
                    }
                },
                "parameters": {
                    "class_names": {"type": "STRING"},
                    "classification_type": {"type": "STRING"},
                    "dataflow_disk_size": {"type": "INT"},
                    "dataflow_machine_type": {"type": "STRING"},
                    "dataflow_max_workers_num": {"type": "INT"},
                    "dataflow_workers_num": {"type": "INT"},
                    "example_weight_column": {"type": "STRING"},
                    "generate_feature_attribution": {"type": "STRING"},
                    "ground_truth_column": {"type": "STRING"},
                    "location": {"type": "STRING"},
                    "positive_classes": {"type": "STRING"},
                    "prediction_id_column": {"type": "STRING"},
                    "prediction_label_column": {"type": "STRING"},
                    "prediction_score_column": {"type": "STRING"},
                    "predictions_format": {"type": "STRING"},
                    "problem_type": {"type": "STRING"},
                    "project": {"type": "STRING"},
                    "root_dir": {"type": "STRING"},
                },
            },
            "outputDefinitions": {
                "artifacts": {
                    "evaluation_metrics": {
                        "artifactType": {
                            "schemaTitle": "system.Metrics",
                            "schemaVersion": "0.0.1",
                        }
                    }
                }
            },
        },
    },
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
        "schemaVersion": "2.0.0",
        "sdkVersion": "kfp-1.8.12",
        "components": {
            "comp-importer": {
                "executorLabel": "exec-importer",
                "inputDefinitions": {"parameters": {"uri": {"type": "STRING"}}},
                "outputDefinitions": {
                    "artifacts": {
                        "artifact": {
                            "artifactType": {
                                "schemaTitle": "google.VertexModel",
                                "schemaVersion": "0.0.1",
                            }
                        }
                    }
                },
            },
            "comp-model-batch-predict": {
                "executorLabel": "exec-model-batch-predict",
                "inputDefinitions": {
                    "artifacts": {
                        "model": {
                            "artifactType": {
                                "schemaTitle": "google.VertexModel",
                                "schemaVersion": "0.0.1",
                            }
                        }
                    },
                    "parameters": {
                        "accelerator_count": {"type": "INT"},
                        "accelerator_type": {"type": "STRING"},
                        "bigquery_destination_output_uri": {"type": "STRING"},
                        "encryption_spec_key_name": {"type": "STRING"},
                        "explanation_metadata": {"type": "STRING"},
                        "explanation_parameters": {"type": "STRING"},
                        "gcs_destination_output_uri_prefix": {"type": "STRING"},
                        "gcs_source_uris": {"type": "STRING"},
                        "generate_explanation": {"type": "STRING"},
                        "instances_format": {"type": "STRING"},
                        "job_display_name": {"type": "STRING"},
                        "labels": {"type": "STRING"},
                        "location": {"type": "STRING"},
                        "machine_type": {"type": "STRING"},
                        "manual_batch_tuning_parameters_batch_size": {"type": "INT"},
                        "max_replica_count": {"type": "INT"},
                        "model_parameters": {"type": "STRING"},
                        "predictions_format": {"type": "STRING"},
                        "project": {"type": "STRING"},
                        "starting_replica_count": {"type": "INT"},
                    },
                },
                "outputDefinitions": {
                    "artifacts": {
                        "batchpredictionjob": {
                            "artifactType": {
                                "schemaTitle": "google.VertexBatchPredictionJob",
                                "schemaVersion": "0.0.1",
                            }
                        }
                    },
                    "parameters": {"gcp_resources": {"type": "STRING"}},
                },
            },
            "comp-model-evaluation": {
                "executorLabel": "exec-model-evaluation",
                "inputDefinitions": {
                    "artifacts": {
                        "batch_prediction_job": {
                            "artifactType": {
                                "schemaTitle": "google.VertexBatchPredictionJob",
                                "schemaVersion": "0.0.1",
                            }
                        }
                    },
                    "parameters": {
                        "class_names": {"type": "STRING"},
                        "classification_type": {"type": "STRING"},
                        "dataflow_disk_size": {"type": "INT"},
                        "dataflow_machine_type": {"type": "STRING"},
                        "dataflow_max_workers_num": {"type": "INT"},
                        "dataflow_workers_num": {"type": "INT"},
                        "example_weight_column": {"type": "STRING"},
                        "generate_feature_attribution": {"type": "STRING"},
                        "ground_truth_column": {"type": "STRING"},
                        "location": {"type": "STRING"},
                        "positive_classes": {"type": "STRING"},
                        "prediction_id_column": {"type": "STRING"},
                        "prediction_label_column": {"type": "STRING"},
                        "prediction_score_column": {"type": "STRING"},
                        "predictions_format": {"type": "STRING"},
                        "problem_type": {"type": "STRING"},
                        "project": {"type": "STRING"},
                        "root_dir": {"type": "STRING"},
                    },
                },
                "outputDefinitions": {
                    "artifacts": {
                        "evaluation_metrics": {
                            "artifactType": {
                                "schemaTitle": "system.Metrics",
                                "schemaVersion": "0.0.1",
                            }
                        }
                    }
                },
            },
        },
    }
)

_TEST_INVALID_MODEL_EVAL_PIPELINE_SPEC = {
    "pipelineInfo": {"name": "evaluation-sdk-pipeline"},
    "root": {
        "dag": {"tasks": {}},
        "inputDefinitions": {
            "parameters": {
                "batch_predict_gcs_source_uris": {"type": "LIST"},
                "batch_predict_instances_format": {"type": "STRING"},
                "class_names": {"type": "LIST"},
                "model_name": {"type": "STRING"},
                "prediction_type": {"type": "STRING"},
                "project": {"type": "STRING"},
                "location": {"parameterType": "STRING"},
                "root_dir": {"type": "STRING"},
                "target_column_name": {"type": "STRING"},
            }
        },
    },
    "schemaVersion": "2.0.0",
    "sdkVersion": "kfp-1.8.12",
    "components": {
        "comp-importer": {
            "executorLabel": "exec-importer",
            "inputDefinitions": {"parameters": {"uri": {"type": "STRING"}}},
            "outputDefinitions": {
                "artifacts": {
                    "artifact": {
                        "artifactType": {
                            "schemaTitle": "google.VertexModel",
                            "schemaVersion": "0.0.1",
                        }
                    }
                }
            },
        },
        "comp-model-batch-predict": {
            "executorLabel": "exec-model-batch-predict",
            "inputDefinitions": {
                "artifacts": {
                    "model": {
                        "artifactType": {
                            "schemaTitle": "google.VertexModel",
                            "schemaVersion": "0.0.1",
                        }
                    }
                },
                "parameters": {
                    "accelerator_count": {"type": "INT"},
                    "accelerator_type": {"type": "STRING"},
                    "bigquery_destination_output_uri": {"type": "STRING"},
                    "encryption_spec_key_name": {"type": "STRING"},
                    "explanation_metadata": {"type": "STRING"},
                    "explanation_parameters": {"type": "STRING"},
                    "gcs_destination_output_uri_prefix": {"type": "STRING"},
                    "gcs_source_uris": {"type": "STRING"},
                    "generate_explanation": {"type": "STRING"},
                    "instances_format": {"type": "STRING"},
                    "job_display_name": {"type": "STRING"},
                    "labels": {"type": "STRING"},
                    "location": {"type": "STRING"},
                    "machine_type": {"type": "STRING"},
                    "manual_batch_tuning_parameters_batch_size": {"type": "INT"},
                    "max_replica_count": {"type": "INT"},
                    "model_parameters": {"type": "STRING"},
                    "predictions_format": {"type": "STRING"},
                    "project": {"type": "STRING"},
                    "starting_replica_count": {"type": "INT"},
                },
            },
            "outputDefinitions": {
                "artifacts": {
                    "batchpredictionjob": {
                        "artifactType": {
                            "schemaTitle": "google.VertexBatchPredictionJob",
                            "schemaVersion": "0.0.1",
                        }
                    }
                },
                "parameters": {"gcp_resources": {"type": "STRING"}},
            },
        },
    },
}

_TEST_MODEL_EVAL_PIPELINE_JOB = json.dumps(
    {
        "runtimeConfig": {
            "parameterValues": _TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES
        },
        "pipelineSpec": json.loads(_TEST_MODEL_EVAL_PIPELINE_SPEC_JSON),
    }
)

_TEST_PIPELINE_GET_METHOD_NAME = "get_fake_pipeline_job"
_TEST_PIPELINE_LIST_METHOD_NAME = "list_fake_pipeline_jobs"
_TEST_PIPELINE_CANCEL_METHOD_NAME = "cancel_fake_pipeline_job"
_TEST_PIPELINE_DELETE_METHOD_NAME = "delete_fake_pipeline_job"
_TEST_PIPELINE_RESOURCE_NAME = (
    f"{_TEST_PARENT}/fakePipelineJobs/{_TEST_PIPELINE_JOB_ID}"
)
_TEST_PIPELINE_CREATE_TIME = datetime.datetime.now()

_TEST_EVAL_PIPELINE_DISPLAY_NAME = "modelevaluationjob"

_TEST_EVAL_RESOURCE_NAME = f"projects/{_TEST_ID}/locations/{_TEST_LOCATION}/models/{_TEST_ID}/evaluations/{_TEST_ID}"


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
    model.name = _TEST_ID
    # model.resource_name = _TEST_MODEL_RESOURCE_NAME,
    model._latest_future = None
    model._exception = None
    model._gca_resource = gca_model.Model(
        display_name="test-eval-model",
        description="This is the mock Model's description",
        name=_TEST_MODEL_NAME,
    )
    yield model


# Mocks specific to ModelEvaluation


@pytest.fixture
def mock_model_eval_get():
    with mock.patch.object(
        model_service_client.ModelServiceClient, "get_model_evaluation"
    ) as mock_get_model_eval:
        mock_get_model_eval.return_value = gca_model_evaluation.ModelEvaluation(
            name=_TEST_EVAL_RESOURCE_NAME
        )
        yield mock_get_model_eval


@pytest.fixture
def mock_model_eval_create():
    with mock.patch.object(
        model_service_client.ModelServiceClient, "get_model_evaluation"
    ) as mock_get_model_eval:
        mock_get_model_eval.return_value = gca_model_evaluation.ModelEvaluation(
            name=_TEST_EVAL_RESOURCE_NAME
        )
        yield mock_get_model_eval


class TestModelEvaluationJob:
    class FakeModelEvaluationJob(model_evaluation_job.ModelEvaluationJob):
        _template_ref = _TEST_TEMPLATE_PATH
        metadat_output_artifact = "TODO"

    @pytest.mark.parametrize(
        "job_spec", [_TEST_MODEL_EVAL_PIPELINE_JOB],
    )
    @pytest.mark.usefixtures(
        "mock_pipeline_service_get", "mock_pipeline_service_create",
    )
    def test_init_model_evaluation_job(
        self,
        mock_pipeline_service_get,
        mock_pipeline_service_create,
        job_spec,
        mock_load_yaml_and_json,
        json_file,
    ):
        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            credentials=_TEST_CREDENTIALS,
            staging_bucket=_TEST_GCS_BUCKET_NAME,
        )

        self.FakeModelEvaluationJob._template_ref = os.path.join(
            os.getcwd(), "tests/unit/aiplatform/test_model_eval_template.json"
        )

        job_spec = yaml.safe_load(job_spec)
        pipeline_spec = job_spec.get("pipelineSpec") or job_spec

        # job_spec = yaml.safe_load(job_spec)
        job = pipeline_jobs.PipelineJob(
            display_name=_TEST_PIPELINE_JOB_DISPLAY_NAME,
            template_path=self.FakeModelEvaluationJob._template_ref,
            parameter_values=_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES,
            job_id=_TEST_PIPELINE_JOB_ID,
        )

        job.run(
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
            create_request_timeout=None,
        )

        job.wait()
        # job.pipeline_spec = _TEST_MODEL_EVAL_PIPELINE_SPEC

        self.FakeModelEvaluationJob(evaluation_pipeline_run=job.resource_name,)

        # expected_runtime_config_dict = {
        #     "gcsOutputDirectory": _TEST_GCS_BUCKET_NAME,
        #     "parameterValues": _TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES,
        # }
        # runtime_config = gca_pipeline_job_v1.PipelineJob.RuntimeConfig()._pb
        # json_format.ParseDict(expected_runtime_config_dict, runtime_config)

        # job_spec = yaml.safe_load(job_spec)
        # pipeline_spec = job_spec.get("pipelineSpec") or job_spec

        # expected_gapic_pipeline_job = gca_pipeline_job_v1.PipelineJob(
        #     display_name=_TEST_PIPELINE_JOB_DISPLAY_NAME,
        #     pipeline_spec={
        #         "components": pipeline_spec["components"],
        #         "pipelineInfo": pipeline_spec["pipelineInfo"],
        #         "root": pipeline_spec["root"],
        #         "schemaVersion": "2.0.0",
        #         "sdkVersion": "kfp-1.8.12",
        #     },
        #     runtime_config=runtime_config,
        #     service_account=_TEST_SERVICE_ACCOUNT,
        #     network=_TEST_NETWORK,
        # )

    # def test_init_model_evaluation_job_with_invalid_eval_template_raises(
    #     self,
    #     mock_pipeline_service_get,
    # ):

    #     aiplatform.init(
    #         project=_TEST_PROJECT,
    #         location=_TEST_LOCATION,
    #         credentials=_TEST_CREDENTIALS,
    #     )

    #     self.FakeModelEvaluationJob._template_ref = "TODO invalid template ref"

    #     with pytest.raises(ValueError):
    #         self.FakeModelEvaluationJob(
    #             evaluation_pipeline_run=_TEST_PIPELINE_JOB_NAME,
    #         )

    def test_init_model_evaluation_job_with_invalid_pipeline_job_name_raises(
        self, mock_pipeline_service_get,
    ):
        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            credentials=_TEST_CREDENTIALS,
        )

        with pytest.raises(ValueError):
            ModelEvaluationJob(evaluation_pipeline_run=_TEST_INVALID_PIPELINE_JOB_NAME,)

    @pytest.mark.parametrize(
        "job_spec",
        [_TEST_MODEL_EVAL_PIPELINE_SPEC_JSON, _TEST_MODEL_EVAL_PIPELINE_JOB],
    )
    def test_model_evaluation_job_submit(
        self,
        mock_pipeline_service_create,
        mock_pipeline_service_get,
        job_spec,
        mock_load_yaml_and_json,
        mock_model,
        get_model_mock,
    ):
        self.FakeModelEvaluationJob._template_ref = _TEST_TEMPLATE_PATH

        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            credentials=_TEST_CREDENTIALS,
            staging_bucket=_TEST_GCS_BUCKET_NAME,
        )

        managed_model = models.Model(model_name=_TEST_MODEL_RESOURCE_NAME,)

        test_eval_job = self.FakeModelEvaluationJob.submit(
            model=managed_model,
            prediction_type=_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES[
                "prediction_type"
            ],
            pipeline_root=_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES["root_dir"],
            target_column_name=_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES[
                "target_column_name"
            ],
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
        expected_runtime_config_dict = {
            "gcsOutputDirectory": _TEST_PIPELINE_ROOT,
            "parameterValues": _TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES,
        }

        runtime_config = gca_pipeline_job_v1.PipelineJob.RuntimeConfig()._pb
        json_format.ParseDict(expected_runtime_config_dict, runtime_config)

        job_spec = yaml.safe_load(job_spec)
        pipeline_spec = job_spec.get("pipelineSpec") or job_spec

        fake_service_display_name = self.FakeModelEvaluationJob.__name__.lower()

        # Construct expected request
        expected_gapic_pipeline_job = gca_pipeline_job_v1.PipelineJob(
            display_name=fake_service_display_name,
            pipeline_spec={
                "components": pipeline_spec["components"],
                "pipelineInfo": pipeline_spec["pipelineInfo"],
                "root": pipeline_spec["root"],
                "schemaVersion": pipeline_spec["schemaVersion"],
                "sdkVersion": pipeline_spec["sdkVersion"],
            },
            # runtime_config=runtime_config,
            service_account=_TEST_SERVICE_ACCOUNT,
            network=_TEST_NETWORK,
        )

        # print(expected_gapic_pipeline_job)

        test_job_create_time_str = _TEST_PIPELINE_CREATE_TIME.strftime("%Y%m%d%H%M%S")

        mock_pipeline_service_create.assert_called_with(
            parent=_TEST_PARENT,
            pipeline_job=expected_gapic_pipeline_job,
            pipeline_job_id=f"evaluation-sdk-pipeline-{test_job_create_time_str}",
            # service_account=_TEST_SERVICE_ACCOUNT,
            # network=_TEST_NETWORK,
            timeout=None,
        )

        # mock_pipeline_service_get.assert_called_with(
        #     name=_TEST_PIPELINE_JOB_NAME, retry=base._DEFAULT_RETRY
        # )

        # assert job._gca_resource == make_pipeline_job(
        #     gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED
        # )

    # TODO: test_model_evaluation_job_submit

    # TODO: test_model_evaluation_job_submit_with_invalid_*

    # TODO: test_model_evaluation_job_get_model_evaluation_with_failed_pipeline_run_raises

    # TODO: test_model_evaluation_job_get_model_evaluation_with_pending_pipeline_run

    # TODO: test_model_evaluation_job_get_model_evaluation_with_successful_pipeline_run


class TestModelEvaluation:
    def test_init_model_evaluation(self, mock_model_eval_get):
        aiplatform.init(project=_TEST_PROJECT)

        my_model_eval = aiplatform.ModelEvaluation(
            evaluation_name=_TEST_EVAL_RESOURCE_NAME
        )

        mock_model_eval_get.assert_called_once_with(
            name=_TEST_EVAL_RESOURCE_NAME, retry=base._DEFAULT_RETRY
        )

    # TODO: test_init_model_evaluation_with_invalid_evaluation_resource_raises

    def test_get_model_evaluation_from_pipeline_job(
        self,
        mock_model_eval_get,
        mock_pipeline_service_get,
        mock_pipeline_service_create,
    ):
        aiplatform.init(project=_TEST_PROJECT)

        job = pipeline_jobs.PipelineJob(
            display_name=_TEST_PIPELINE_JOB_DISPLAY_NAME,
            template_path=_TEST_TEMPLATE_PATH,
            parameter_values=_TEST_MODEL_EVAL_PIPELINE_PARAMETER_VALUES,
            job_id=_TEST_PIPELINE_JOB_ID,
        )

        ModelEvaluation.get_from_pipeline_job(pipeline_job=job.resource_name,)

        mock_model_eval_get.assert_called_once_with(
            name=_TEST_EVAL_RESOURCE_NAME, retry=base._DEFAULT_RETRY
        )

    def test_get_model_evaluation_from_pipeline_job_id(
        self,
        mock_model_eval_get,
        mock_pipeline_service_get,
        mock_pipeline_service_create,
    ):
        aiplatform.init(project=_TEST_PROJECT)

        model_eval_test = ModelEvaluation.get_from_pipeline_job(
            pipeline_job_id=_TEST_PIPELINE_JOB_ID,
        )

        print(model_eval_test)

        mock_model_eval_get.assert_called_once_with(
            name=_TEST_EVAL_RESOURCE_NAME, retry=base._DEFAULT_RETRY
        )

    # TODO: test_get_model_evaluation_with_invalid_pipeline_job_raises

    # TODO: test_get_model_evaluation_with_invalid_template_raises

    # print('add tests here')

    # @pytest.mark.parametrize(
    #     "featurestore_name", [_TEST_FEATURESTORE_ID, _TEST_FEATURESTORE_NAME]
    # )
    # def test_init_featurestore(self, featurestore_name, get_featurestore_mock):
    #     aiplatform.init(project=_TEST_PROJECT)

    #     my_featurestore = aiplatform.Featurestore(featurestore_name=featurestore_name)

    #     get_featurestore_mock.assert_called_once_with(
    #         name=my_featurestore.resource_name, retry=base._DEFAULT_RETRY
    #     )
