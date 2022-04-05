# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
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

import re
from typing import Dict, NamedTuple, Optional

from google.cloud.aiplatform import utils
from google.cloud.aiplatform import pipeline_jobs

def _validate_model_evaluation_pipeline(pipeline_run: pipeline_jobs.PipelineJob):
    """Helper function to validate whether the provided pipeline run 
    was a Model Evaluation pipeline run."""   
    print(pipeline_run._gca_resource.job_detail.task_details)