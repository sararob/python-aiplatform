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

import importlib
import os
import uuid

import pytest

from google.cloud import storage

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from tests.system.aiplatform import e2e_base

from google.cloud.aiplatform.compat.types import (
    pipeline_state as gca_pipeline_state,
)

_BLOB_PATH = "california-housing-data.csv"
_DATASET_SRC = "https://dl.google.com/mlcc/mledu-datasets/california_housing_train.csv"
_DIR_NAME = os.path.dirname(os.path.abspath(__file__))
_LOCAL_TRAINING_SCRIPT_PATH = os.path.join(
    _DIR_NAME, "test_resources/california_housing_training_script.py"
)
_INSTANCE = {
    "longitude": -124.35,
    "latitude": 40.54,
    "housing_median_age": 52.0,
    "total_rooms": 1820.0,
    "total_bedrooms": 300.0,
    "population": 806,
    "households": 270.0,
    "median_income": 3.014700,
}

_TEST_PROJECT = e2e_base._PROJECT
_TEST_LOCATION = e2e_base._LOCATION
_EVAL_METRICS_KEYS_CLASSIFICATION = [
    "auPrc",
    "auRoc",
    "logLoss",
    "confidenceMetrics",
    "confusionMatrix",
]


# TODO: swap this when we run it on sample test project
# _TEST_IRIS_MODEL_ID = "1603035176739274752"
# _TEST_AUTOML_EVAL_DATA_URI = "gs://ucaip-sample-resources/iris_1000.csv"
_TEST_IRIS_MODEL_ID = "6091413165134839808"
_TEST_AUTOML_EVAL_DATA_URI = "gs://sdk-model-eval/iris_training.csv"
_EVAL_DATA_NO_LABEL_COLUMN = "gs://sdk-model-eval/feature_columns.csv"
_TEST_TF_EVAL_DATA_JSONL = "gs://sdk-model-eval/iris.jsonl"
_TEST_XGB_CLASSIFICATION_MODEL_ID = "2094468495843524608"
_TEST_XGB_REGRESSION_MODEL_ID = "6448604910580662272"
_TEST_CUSTOM_TF_MODEL_ID = "5047141001538306048"
_TEST_SCIKIT_MODEL_ID = "2372002822880231424"

_TEST_PERMANENT_CUSTOM_MODEL_CLASSIFICATION_RESOURCE_NAME = f"projects/{_TEST_PROJECT}/locations/us-central1/models/{_TEST_XGB_CLASSIFICATION_MODEL_ID}"
_TEST_PERMANENT_TF_MODEL_CLASSIFICATION_RESOURCE_NAME = (
    f"projects/{_TEST_PROJECT}/locations/us-central1/models/{_TEST_CUSTOM_TF_MODEL_ID}"
)
_TEST_PERMANENT_CUSTOM_MODEL_REGRESSION_RESOURCE_NAME = f"projects/{_TEST_PROJECT}/locations/us-central1/models/{_TEST_XGB_REGRESSION_MODEL_ID}"
_TEST_PERMANENT_SCIKIT_RESOURCE_NAME = (
    f"projects/{_TEST_PROJECT}/locations/us-central1/models/{_TEST_SCIKIT_MODEL_ID}"
)
_TEST_PERMANENT_AUTOML_MODEL_RESOURCE_NAME = (
    f"projects/{_TEST_PROJECT}/locations/us-central1/models/{_TEST_IRIS_MODEL_ID}"
)

_LOGGER = base.Logger(__name__)


class TestModelEvaluationJob(e2e_base.TestEndToEnd):

    _temp_prefix = "temp_vertex_sdk_model_evaluation_test"

    def setup_method(self):
        importlib.reload(initializer)
        importlib.reload(aiplatform)

        aiplatform.init(project=_TEST_PROJECT, location=_TEST_LOCATION)

    @pytest.fixture()
    def storage_client(self):
        yield storage.Client(project=_TEST_PROJECT)

    @pytest.fixture()
    def staging_bucket(self, storage_client):
        new_staging_bucket = f"temp-sdk-integration-{uuid.uuid4()}"
        bucket = storage_client.create_bucket(
            new_staging_bucket, location="us-central1"
        )

        yield bucket

        # bucket.delete(force=True)

    # TODO: get this test passing with custom models
    def test_model_evaluate_custom_tabular_model(self, staging_bucket):

        custom_model = aiplatform.Model(model_name=_TEST_PERMANENT_SCIKIT_RESOURCE_NAME)

        eval_job = custom_model.evaluate(
            data_type="tabular",
            gcs_source_uris=[_TEST_AUTOML_EVAL_DATA_URI],
            prediction_type="classification",
            target_column_name="species",
            evaluation_staging_path=f"gs://{staging_bucket.name}",
            instances_format="jsonl",
        )

        _LOGGER.info("%s, state before completion", eval_job.backing_pipeline_job.state)

        eval_job.wait()

        _LOGGER.info("%s, state after completion", eval_job.backing_pipeline_job.state)

        assert (
            eval_job.state == gca_pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED
        )
        assert eval_job.state == eval_job.backing_pipeline_job.state

        assert eval_job.resource_name == eval_job.backing_pipeline_job.resource_name

        eval_metrics_artifact = aiplatform.Artifact(
            artifact_name=eval_job._metadata_output_artifact
        )

        assert isinstance(eval_metrics_artifact, aiplatform.Artifact)

        _LOGGER.info("%s metadata output artifact", eval_job._metadata_output_artifact)

        model_eval = eval_job.get_model_evaluation()

        eval_metrics_dict = model_eval.metrics

        for metric_name in _EVAL_METRICS_KEYS_CLASSIFICATION:
            assert metric_name in eval_metrics_dict

    # def test_model_evaluate_automl_tabular_model(self, staging_bucket):

    #     automl_model = aiplatform.Model(
    #         model_name=_TEST_PERMANENT_AUTOML_MODEL_RESOURCE_NAME
    #     )

    #     eval_job = automl_model.evaluate(
    #         data_type="tabular",
    #         gcs_source_uris=[_TEST_AUTOML_EVAL_DATA_URI],
    #         prediction_type="classification",
    #         target_column_name="species",
    #         evaluation_staging_path=f"gs://{staging_bucket.name}",
    #         instances_format="csv",
    #     )

    #     eval_job.wait()

    #     assert (
    #         eval_job.state == gca_pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED
    #     )
    #     assert eval_job.state == eval_job.backing_pipeline_job.state

    #     assert eval_job.resource_name == eval_job.backing_pipeline_job.resource_name

    #     eval_metrics_artifact = aiplatform.Artifact(
    #         artifact_name=eval_job._metadata_output_artifact
    #     )

    #     assert isinstance(eval_metrics_artifact, aiplatform.Artifact)

    #     _LOGGER.info("%s metadata output artifact", eval_job._metadata_output_artifact)

    #     model_eval = eval_job.get_model_evaluation()

    #     eval_metrics_dict = model_eval.metrics

    #     for metric_name in _EVAL_METRICS_KEYS_CLASSIFICATION:
    #         assert metric_name in eval_metrics_dict
