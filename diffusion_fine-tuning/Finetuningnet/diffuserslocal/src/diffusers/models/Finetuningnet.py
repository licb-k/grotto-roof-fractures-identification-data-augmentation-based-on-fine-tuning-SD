# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from ..utils import deprecate
from .Finetuningnet.Finetuningnet import (  # noqa
    BaseOutput,
    #ControlNetConditioningEmbedding,
    FinetuningnetModel,
    FinetuningnetOutput,
    zero_module,
)


class FinetuningnetOutput(FinetuningnetOutput):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FinetuningnetOutput` from `diffusers.models.Finetuningnet` is deprecated and this will be removed in a future version."
        deprecate("FinetuningnetOutput", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)


class FinetuningnetModel(FinetuningnetModel):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FinetuningnetModel` from `diffusers.models.Finetuningnet` is deprecated and this will be removed in a future version."
        deprecate("FinetuningnetModel", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)

