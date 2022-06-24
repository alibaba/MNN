import urllib.request
import json
import time
import datetime
import os
import base64

try:
    import aliyun.log
except ImportError:
    print("try 'pip install -U aliyun-log-python-sdk'")
    os.system("pip install -U aliyun-log-python-sdk")
from aliyun.log import LogClient
from aliyun.log import PutLogsRequest
from aliyun.log.logitem import LogItem

import platform
import os
import uuid


class MNNLogger(object):
    def __init__(self):
        self._url = base64.urlsafe_b64decode(b'aHR0cHM6Ly8xMDMyMjc3OTQ5NDA5MTkzLmNuLWhhbmd6aG91LmZjLmFsaXl1bmNzLmNvbS8yMDE2LTA4LTE1L3Byb3h5L21ubi1zZXJ2aWNlL3dvcmtzdGF0aW9uLXN0cy8=').decode()
        self._endpoint = base64.urlsafe_b64decode(b'aHR0cHM6Ly9jbi1oYW5nemhvdS5sb2cuYWxpeXVuY3MuY29t').decode()
        self._log_project = base64.urlsafe_b64decode(b'bW5uLW1vbml0b3I=').decode()
        self._log_store = base64.urlsafe_b64decode(b'bW5uLWNvbXByZXNz').decode()
        self._network_available = True
        self._activate()

    def _activate(self):
        try:
            req = urllib.request.Request(self._url)
            res = urllib.request.urlopen(req)
            data = res.read()
            temp_credentials = json.loads(data)

            access_key_id = temp_credentials['Credentials']['AccessKeyId']
            access_key = temp_credentials['Credentials']['AccessKeySecret']
            security_token = temp_credentials['Credentials']['SecurityToken']
            self._expire_time = temp_credentials['Credentials']['Expiration']

            self._client = LogClient(self._endpoint, access_key_id, access_key, security_token)
        except:
            self._network_available = False

    def _is_token_valid(self):
        utc_date = datetime.datetime.strptime(self._expire_time, "%Y-%m-%dT%H:%M:%SZ")
        local_date = utc_date + datetime.timedelta(hours=8)
        now_time = int(time.time())
        if local_date.timestamp() - now_time < 60:
            return False
        else:
            return True

    def _get_machine_id(self, os_type):
        machine_id = ""

        if os_type == "Linux":
            if os.path.exists("/etc/machine-id"):
                machine_id = os.popen("cat /etc/machine-id").readline().strip().lower()
        elif os_type == "Darwin":
            res = os.popen("ioreg -rd1 -c IOPlatformExpertDevice | grep UUID").readline().strip().split('"')
            if len(res) > 1:
                machine_id = res[-2].lower()
        elif os_type == "Windows":
            res = os.popen("reg query HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Cryptography\ /v MachineGuid").read().strip().split(" ")[-1].lower()
        else:
            pass

        if machine_id == "":
            machine_id = uuid.uuid1().hex[20:]

        return machine_id

    def _collect_basic_logs(self):
        basic_logs = {}
        from mnncompress.version import __version__
        basic_logs["version"] = __version__

        os_type = platform.system()
        basic_logs["os"] = os_type
        basic_logs["machine_id"] = self._get_machine_id(os_type)
        
        return basic_logs

    def _collect_contents(self, log_dict, contents):
        for key, value in log_dict.items():
            key = str(key)
            if isinstance(value, dict):
                value = json.dumps(value)
            else:
                value = str(value)
            contents.append((key, value))
    
    def put_log(self, log_dict, topic=None):
        if not self._network_available:
            print("network not available...")
            return False
        
        try:
            if not self._is_token_valid():
                self._activate()
            
            contents = []
            self._collect_contents(self._collect_basic_logs(), contents)
            self._collect_contents(log_dict, contents)
            
            log_item = LogItem()
            log_item.set_time(int(time.time()))
            log_item.set_contents(contents)
            
            req = PutLogsRequest(self._log_project, self._log_store, topic, '', [log_item,])
            res = self._client.put_logs(req)
            return True
        except:
            return False

    def on_done(self, framework, model_guid, detail):
        if framework not in ["tensorflow", "pytorch"]:
            raise ValueError("framework should be tensorflow or pytorch")
        log_dict = {}
        log_dict["framework"] = framework
        log_dict["model_guid"] = model_guid
        log_dict["detail"] = detail
        res = self.put_log(log_dict)

        return res

mnn_logger = MNNLogger()
