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
from urllib import request

import pytest

from google.cloud import storage

from google.cloud import aiplatform
from google.cloud.aiplatform import initializer
from tests.system.aiplatform import e2e_base

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

        bucket.delete(force=True)

    def test_model_evaluate_custom_model(self, staging_bucket):
        # assert shared_state["bucket"]
        # bucket = shared_state["bucket"]

        blob = staging_bucket.blob(_BLOB_PATH)

        # Download the CSV file into memory and save it directory to staging bucket
        with request.urlopen(_DATASET_SRC) as response:
            data = response.read()
            blob.upload_from_string(data)

        dataset_gcs_source = f"gs://{staging_bucket.name}/{_BLOB_PATH}"

        ds = aiplatform.TabularDataset.create(
            gcs_source=[dataset_gcs_source],
            sync=False,
            create_request_timeout=180.0,
        )

        custom_job = aiplatform.CustomTrainingJob(
            display_name=self._make_display_name("train-housing-custom-model-eval"),
            script_path=_LOCAL_TRAINING_SCRIPT_PATH,
            container_uri="gcr.io/cloud-aiplatform/training/tf-cpu.2-2:latest",
            requirements=["gcsfs==0.7.1"],
            model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-2:latest",
            staging_bucket=staging_bucket.name,
        )

        # model to use for eval testsing: projects/462141068491/locations/us-central1/models/720026184565391360

        custom_model = custom_job.run(
            ds,
            replica_count=1,
            model_display_name=self._make_display_name("custom-housing-model"),
            timeout=1234,
            restart_job_on_worker_restart=True,
            enable_web_access=True,
            sync=False,
            create_request_timeout=None,
        )

        custom_model.wait()

        eval_job = custom_model.evaluate(
            data_type="tabular",
            gcs_source_uris=[dataset_gcs_source],
            prediction_type="regression",
            target_column_name="median_house_value",
            evaluation_staging_path=f"gs://{staging_bucket.name}",
            instances_format="csv",
            evaluation_job_display_name="test-pipeline-display-name",
        )

        print(eval_job.backing_pipeline_job.state, "state before completion")

        eval_job.wait()

        print(eval_job.backing_pipeline_job, "pipeline job backing eval")
        print(eval_job.backing_pipeline_job.resource_name, "pipeline resource name")
        print(eval_job.backing_pipeline_job.state, "pipeline job completed state")

        model_eval = eval_job.get_model_evaluation()
        print(model_eval.metrics, "eval metrics")
        print(model_eval.metadata_output_artifact, "mlmd uri of eval")
