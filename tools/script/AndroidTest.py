import os
import sys
import time
import base64
import hashlib
import irmasdk
from irmasdk.api import RequestApi
import modelTestForAndroid as modelTest

def run_cmd(args):
    from subprocess import Popen, PIPE, STDOUT
    stdout, _ = Popen(args, stdout=PIPE, stderr=STDOUT).communicate()
    return stdout.decode('utf-8')

def push_adbkey(auth):
    # get fingerprint and title
    adbkey_file = os.path.expanduser('~') + '/.android/adbkey.pub'
    if not os.path.exists(adbkey_file):
        print('No adbkey.pub file!')
        return
    adbkey = open(adbkey_file).read().split(' ')
    hexdigest = hashlib.md5(base64.b64decode(adbkey[0])).hexdigest()
    chunks = [ hexdigest[i:i+2] for i in range(0, len(hexdigest), 2) ]
    fingerprint = ':'.join(chunks)
    title = adbkey[1]
    print(fingerprint, title)
    # push them
    body = {}
    body['fingerprint'] = fingerprint
    body['title'] = title
    res = RequestApi().request_irma(auth=auth, body=body, path='/api/user/adbKey', method='POST')
    if res['code'] != 200:
        raise NameError('push adbkey failed: ', res)

class IRMASession:
    def __init__(self):
        try:
            print(run_cmd(['adb', 'disconnect']))
        except:
            pass
        self.domain = 'https://irma-backend.alibaba-inc.com'
        self.key = 'tb-tech-mnn'
        self.secret = '8RqrPcPbTUe0onP8w6sRFtaV5J6ifUxm'
        self.auth = irmasdk.Auth(domain=self.domain, key=self.key, secret=self.secret)
        self.avilable_devicelist = []
        self.acquired_devicelist = []
        self.device_symb = None
        self.device_rcid = None
        self.device_url  = None
        push_adbkey(self.auth)

    def list_device(self):
        self.avilable_devicelist = []
        devicelist = irmasdk.list_device(self.auth)
        if devicelist['code'] != 200:
            raise NameError('list_device: ', devicelist)
        for device in devicelist['data']:
            if device['status'] == 'idle':
                self.avilable_devicelist.append(device['symbol'])
        if len(self.avilable_devicelist) == 0:
            raise NameError('Error: no device !')

    def get_url(self, symbol):
        connect = irmasdk.get_device_remote_connect(self.auth, symbol=symbol)
        if connect['code'] != 200:
            raise NameError('get_device_remote_connect: ', connect)
        return connect['data']['remoteConnectUrl']

    def acquire_device(self):
        # have an avliable device
        if self.device_symb != None or self.device_rcid != None:
            return
        # get all my devices and select one
        self.get_all_devices()
        if len(self.acquired_devicelist) > 0:
            device = self.acquired_devicelist[0]
            self.device_symb = device[0]
            self.device_rcid = device[1]
            self.device_url  = device[2]
            return
        # update avilable_devicelist and acquire one
        self.list_device()
        for symbol in self.avilable_devicelist:
            my_device = irmasdk.acquire_device(self.auth, symbol=symbol, timeout=20)
            if my_device['code'] == 200:
                self.device_symb = symbol
                self.device_rcid = my_device['data']['remoteControlId']
                break
        if self.device_rcid == 'None' or self.device_symb == 'None':
            raise NameError('Error: not acquire !')
        else:
            print('acquire decvice: symbol is %s, rcid is %s'%(self.device_symb, self.device_rcid))
        self.device_url = self.get_url(self.device_symb)

    def release_device(self):
        if self.device_symb == None or self.device_rcid == None:
            return
        release = irmasdk.release_device(self.auth, symbol=self.device_symb, rcid=self.device_rcid)
        if release['code'] != 200:
            raise NameError('release_device: ', release)
        print('release decvice: symbol is %s, rcid is %s'%(self.device_symb, self.device_rcid))
        device_symb = None
        device_rcid = None
        device_url  = None

    def get_all_devices(self):
        my_devices = irmasdk.get_device_by_user(self.auth)
        if my_devices['code'] != 200:
            raise NameError('get_device_by_user: ', my_devices)
        for my_device in my_devices['data']:
            symbol = my_device['symbol']
            rcid = irmasdk.acquire_device(self.auth, symbol=symbol, timeout='20')['data']['remoteControlId']
            url = self.get_url(symbol)
            self.acquired_devicelist.append((symbol, rcid, url))
        print('all acquire devices is : ', self.acquired_devicelist)

    def release_all_devices(self):
        self.get_all_devices()
        for device in self.acquired_devicelist:
            symbol = device[0]
            rcid = device[1]
            release = irmasdk.release_device(self.auth, symbol=symbol, rcid=rcid)
            if release['code'] != 200:
                raise NameError('release_device: ', release)
            print('release decvice: symbol is %s, rcid is %s'%(symbol, rcid))
        try:
            print(run_cmd(['adb', 'disconnect']))
        except:
            pass

    def connect(self):
        self.acquire_device()
        res = run_cmd(['adb', 'connect', self.device_url])
        print(res)
        # sleep 0.5s for connecting
        time.sleep(0.5)
        res = run_cmd(['adb', 'devices'])
        print(res)

    def disconnect(self):
        res = run_cmd(['adb', 'disconnect', self.device_url])
        print(res)
        self.release_device()

    def __del__(self):
        self.release_all_devices()


def test_on_device(model_dir, bits):
    # update *.so and *.out
    res = run_cmd(['../updateTest.sh'])
    print(res)
    # run unit test
    message = run_cmd(['adb', 'shell', 'cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && ./run_test.out all 0 0.002 1 %d'%(bits)])
    if 'TEST_NAME_UNIT' in message:
        print(message)
    else:
        print("TEST_NAME_UNIT%d: 单元测试%d\nTEST_CASE_AMOUNT_UNIT%d: {\"blocked\":1,\"failed\":0,\"passed\":0,\"skipped\":0}\n"%(bits, bits, bits))
        # exit(1)
    # run model test
    modelTest.android_test(model_dir, ' 0 ', ' 0.002 ', bits)


if __name__ == '__main__':
    model_dir = sys.argv[1]
    bits = int(sys.argv[2])
    session = IRMASession()
    session.connect()
    test_on_device(model_dir, bits)
    session.disconnect()
