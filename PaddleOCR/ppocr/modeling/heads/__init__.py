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

__all__ = ["build_head"]


def build_head(config):
    # det head
    # cls head
    from .cls_head import ClsHead
    from .det_ct_head import CT_Head
    from .det_db_head import DBHead, PFHeadLocal
    from .det_east_head import EASTHead
    from .det_fce_head import FCEHead
    from .det_pse_head import PSEHead
    from .det_sast_head import SASTHead
    from .e2e_pg_head import PGHead

    # kie head
    from .kie_sdmgr_head import SDMGRHead
    from .rec_abinet_head import ABINetHead
    from .rec_aster_head import AsterHead
    from .rec_att_head import AttentionHead
    from .rec_can_head import CANHead

    # rec head
    from .rec_ctc_head import CTCHead
    from .rec_multi_head import MultiHead
    from .rec_nrtr_head import Transformer
    from .rec_pren_head import PRENHead
    from .rec_rfl_head import RFLHead
    from .rec_robustscanner_head import RobustScannerHead
    from .rec_sar_head import SARHead
    from .rec_satrn_head import SATRNHead
    from .rec_spin_att_head import SPINAttentionHead
    from .rec_srn_head import SRNHead
    from .rec_visionlan_head import VLHead
    from .table_att_head import SLAHead, TableAttentionHead
    from .table_master_head import TableMasterHead

    support_dict = [
        "DBHead",
        "PSEHead",
        "FCEHead",
        "EASTHead",
        "SASTHead",
        "CTCHead",
        "ClsHead",
        "AttentionHead",
        "SRNHead",
        "PGHead",
        "Transformer",
        "TableAttentionHead",
        "SARHead",
        "AsterHead",
        "SDMGRHead",
        "PRENHead",
        "MultiHead",
        "ABINetHead",
        "TableMasterHead",
        "SPINAttentionHead",
        "VLHead",
        "SLAHead",
        "RobustScannerHead",
        "CT_Head",
        "RFLHead",
        "DRRGHead",
        "CANHead",
        "SATRNHead",
        "PFHeadLocal",
    ]

    if config["name"] == "DRRGHead":
        from .det_drrg_head import DRRGHead

        support_dict.append("DRRGHead")

    # table head

    module_name = config.pop("name")
    assert module_name in support_dict, Exception("head only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
