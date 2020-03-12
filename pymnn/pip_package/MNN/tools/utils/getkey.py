# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python get_key API """
def get_key(d, value):
    """get the key in the dict d if value match"""
    for k, v in d.items():
        if v == value:
            return k
    return "?"
