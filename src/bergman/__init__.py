# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_bergman": ["BERGMAN_PRETRAINED_CONFIG_ARCHIVE_MAP", "BergmanConfig", "BergmanOnnxConfig"],
    "tokenization_bergman": ["BergmanTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_bergman_fast"] = ["BergmanTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_bergman"] = [
        "BERGMAN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BergmanForCausalLM",
        "BergmanForMaskedLM",
        "BergmanForMultipleChoice",
        "BergmanForQuestionAnswering",
        "BergmanForSequenceClassification",
        "BergmanForTokenClassification",
        "BergmanModel",
        "BergmanPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_bergman"] = [
        "TF_BERGMAN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFBergmanForCausalLM",
        "TFBergmanForMaskedLM",
        "TFBergmanForMultipleChoice",
        "TFBergmanForQuestionAnswering",
        "TFBergmanForSequenceClassification",
        "TFBergmanForTokenClassification",
        "TFBergmanMainLayer",
        "TFBergmanModel",
        "TFBergmanPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_bergman"] = [
        "FlaxBergmanForCausalLM",
        "FlaxBergmanForMaskedLM",
        "FlaxBergmanForMultipleChoice",
        "FlaxBergmanForQuestionAnswering",
        "FlaxBergmanForSequenceClassification",
        "FlaxBergmanForTokenClassification",
        "FlaxBergmanModel",
        "FlaxBergmanPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_bergman import BERGMAN_PRETRAINED_CONFIG_ARCHIVE_MAP, BergmanConfig, BergmanOnnxConfig
    from .tokenization_bergman import BergmanTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_bergman_fast import BergmanTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bergman import (
            BERGMAN_PRETRAINED_MODEL_ARCHIVE_LIST,
            BergmanForCausalLM,
            BergmanForMaskedLM,
            BergmanForMultipleChoice,
            BergmanForQuestionAnswering,
            BergmanForSequenceClassification,
            BergmanForTokenClassification,
            BergmanModel,
            BergmanPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
