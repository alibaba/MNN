#!/usr/bin/env python3
###############################################################################
##
## Simple utility class to create a forwarded socket connection to an
## application's stetho domain socket.
##
## Usage:
##
##   sock = stetho_open(
##       device='<serial-no>',
##       process='com.facebook.stetho.sample')
##   doHttp(sock)
##
###############################################################################

import os
import socket
import struct
import re

def get_adb_server_port_from_server_socket():
  socket_spec = os.environ.get('ADB_SERVER_SOCKET')
  if not socket_spec:
      return None
  if not socket_spec.startswith('tcp:'):
    raise HumanReadableError(
      'Invalid or unsupported socket spec \'%s\' specified in ADB_SERVER_SOCKET.' % (
        socket_spec))
  return socket_spec.split(':')[-1]

def get_adb_server_port():
  defaultPort = 5037
  portStr = get_adb_server_port_from_server_socket() or os.environ.get('ANDROID_ADB_SERVER_PORT')
  if portStr is None:
    return defaultPort
  elif portStr.isdigit():
    return int(portStr)
  else:
    raise HumanReadableError(
      'Invalid integer \'%s\' specified in ANDROID_ADB_SERVER_PORT or ADB_SERVER_SOCKET.' % (
        portStr))

def stetho_open(device=None, process=None, port=None):
  if port is None:
    port = get_adb_server_port()

  adb = _connect_to_device(device, port)

  socket_name = None
  if process is None:
    socket_name = _find_only_stetho_socket(device, port)
  else:
    socket_name = _format_process_as_stetho_socket(process)

  try:
    adb.select_service('localabstract:%s' % (socket_name))
  except SelectServiceError as e:
    raise HumanReadableError(
        'Failure to target process %s: %s (is it running?)' % (
            process, e.reason))

  return adb.sock

def read_input(sock, n, tag):
  data = b'';
  while len(data) < n:
    incoming_data = sock.recv(n - len(data))
    if len(incoming_data) == 0:
      break
    data += incoming_data
  if len(data) != n:
    raise IOError('Unexpected end of stream while reading %s.' % tag)
  return data

def _find_only_stetho_socket(device, port):
  adb = _connect_to_device(device, port)
  try:
    adb.select_service('shell:cat /proc/net/unix')
    last_stetho_socket_name = None
    process_names = []
    for line in adb.sock.makefile():
      row = re.split(r'\s+', line.rstrip())
      if len(row) < 8:
        continue
      socket_name = row[7]
      if not socket_name.startswith('@stetho_'):
        continue
      # Filter out entries that are not server sockets
      if int(row[3], 16) != 0x10000 or int(row[5]) != 1:
        continue
      last_stetho_socket_name = socket_name[1:]
      process_names.append(
          _parse_process_from_stetho_socket(socket_name))
    if len(process_names) > 1:
      raise HumanReadableError(
          'Multiple stetho-enabled processes available:%s\n' % (
              '\n\t'.join([''] + list(set(process_names)))) +
          'Use -p <process> or the environment variable STETHO_PROCESS to ' +
          'select one')
    elif last_stetho_socket_name == None:
      raise HumanReadableError('No stetho-enabled processes running')
    else:
      return last_stetho_socket_name
  finally:
    adb.sock.close()

def _connect_to_device(device=None, port=None):
  if port is None:
    raise HumanReadableError('Must specify a port when calling _connect_to_device')

  adb = AdbSmartSocketClient()
  adb.connect(port)

  try:
    if device is None:
      adb.select_service('host:transport-any')
    else:
      adb.select_service('host:transport:%s' % (device))

    return adb
  except SelectServiceError as e:
    raise HumanReadableError(
        'Failure to target device %s: %s' % (device, e.reason))

def _parse_process_from_stetho_socket(socket_name):
  m = re.match(r"^@stetho_(.+)_devtools_remote$", socket_name)
  if m is None:
    raise Exception('Unexpected Stetho socket formatting: %s' % (socket_name))
  return m.group(1)

def _format_process_as_stetho_socket(process):
  return 'stetho_%s_devtools_remote' % (process)

class AdbSmartSocketClient(object):
  """Implements the smartsockets system defined by:
  https://android.googlesource.com/platform/system/core/+/master/adb/protocol.txt
  """

  def __init__(self):
    pass

  def connect(self, port=5037):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      sock.connect(('localhost', port))
    except ConnectionRefusedError:
      sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
      sock.connect(('localhost', port))
    self.sock = sock

  def select_service(self, service):
    message = '%04x%s' % (len(service), service)
    self.sock.send(message.encode('ascii'))
    status = read_input(self.sock, 4, "status")
    if status == b'OKAY':
      # All good...
      pass
    elif status == b'FAIL':
      reason_len = int(read_input(self.sock, 4, "fail reason"), 16)
      reason = read_input(self.sock, reason_len, "fail reason lean").decode('ascii')
      raise SelectServiceError(reason)
    else:
      raise Exception('Unrecognized status=%s' % (status))

class SelectServiceError(Exception):
  def __init__(self, reason):
    self.reason = reason

  def __str__(self):
    return repr(self.reason)

class HumanReadableError(Exception):
  def __init__(self, reason):
    self.reason = reason

  def __str__(self):
    return self.reason
