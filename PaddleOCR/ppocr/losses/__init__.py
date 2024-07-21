# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy

import paddle
import paddle.nn as nn

# basic loss function
# basic_loss
from .basic_loss import DistanceLoss, LossFromOutput

# cls loss
from .cls_loss import ClsLoss

# combined loss function
from .combined_loss import CombinedLoss
from .det_ct_loss import CTLoss

# det loss
from .det_db_loss import DBLoss
from .det_drrg_loss import DRRGLoss
from .det_east_loss import EASTLoss
from .det_fce_loss import FCELoss
from .det_pse_loss import PSELoss
from .det_sast_loss import SASTLoss

# e2e loss
from .e2e_pg_loss import PGLoss
from .kie_sdmgr_loss import SDMGRLoss
from .rec_aster_loss import AsterLoss
from .rec_att_loss import AttentionLoss
from .rec_can_loss import CANLoss
from .rec_ce_loss import CELoss

# rec loss
from .rec_ctc_loss import CTCLoss
from .rec_multi_loss import MultiLoss
from .rec_nrtr_loss import NRTRLoss
from .rec_pren_loss import PRENLoss
from .rec_rfl_loss import RFLLoss
from .rec_sar_loss import SARLoss
from .rec_satrn_loss import SATRNLoss
from .rec_spin_att_loss import SPINAttentionLoss
from .rec_srn_loss import SRNLoss
from .rec_vl_loss import VLLoss

# sr loss
from .stroke_focus_loss import StrokeFocusLoss

# table loss
from .table_att_loss import SLALoss, TableAttentionLoss
from .table_master_loss import TableMasterLoss
from .text_focus_loss import TelescopeLoss

# vqa token loss
from .vqa_token_layoutlm_loss import VQASerTokenLayoutLMLoss


def build_loss(config):
    support_dict = [
        "DBLoss",
        "PSELoss",
        "EASTLoss",
        "SASTLoss",
        "FCELoss",
        "CTCLoss",
        "ClsLoss",
        "AttentionLoss",
        "SRNLoss",
        "PGLoss",
        "CombinedLoss",
        "CELoss",
        "TableAttentionLoss",
        "SARLoss",
        "AsterLoss",
        "SDMGRLoss",
        "VQASerTokenLayoutLMLoss",
        "LossFromOutput",
        "PRENLoss",
        "MultiLoss",
        "TableMasterLoss",
        "SPINAttentionLoss",
        "VLLoss",
        "StrokeFocusLoss",
        "SLALoss",
        "CTLoss",
        "RFLLoss",
        "DRRGLoss",
        "CANLoss",
        "TelescopeLoss",
        "SATRNLoss",
        "NRTRLoss",
    ]
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception("loss only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
