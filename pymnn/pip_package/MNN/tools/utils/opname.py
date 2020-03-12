# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python OP API """
import MNN.tools.mnn_fb.OpType as OpType
import MNN.tools.utils.getkey as GetKey
def optype_to_name(op_type):
    """convert from op type to op name """
    return GetKey.get_key(OpType.OpType.__dict__, op_type)
