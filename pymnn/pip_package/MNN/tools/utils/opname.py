# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python OP API """
from . import getkey as GetKey
from ..mnn_fb import OpType as OpType
def optype_to_name(op_type):
    """convert from op type to op name """
    return GetKey.get_key(OpType.OpType.__dict__, op_type)
