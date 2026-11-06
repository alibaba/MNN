#!/usr/bin/env python3
"""Small stateful stdio MCP fixture with pagination and a server-initiated ping."""
import json
import os
import sys


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


with open(os.environ['MNN_MCP_TEST_MARKER'], 'w') as marker:
    marker.write(str(os.getpid()))

initialized = False
calls = 0
for line in sys.stdin:
    request = json.loads(line)
    method = request.get('method')
    if method == 'notifications/initialized':
        initialized = True
        continue
    if method == 'initialize':
        emit({'jsonrpc':'2.0','id':request['id'],'result':{'protocolVersion':'2025-06-18',
             'capabilities':{'tools':{}},'serverInfo':{'name':'fixture','version':'1'}}})
    elif method == 'tools/list':
        assert initialized
        cursor = request.get('params', {}).get('cursor')
        if not cursor:
            emit({'jsonrpc':'2.0','id':'server-ping','method':'ping'})
            reply = json.loads(sys.stdin.readline())
            assert reply['id'] == 'server-ping' and reply['result'] == {}
        name = 'echo' if not cursor else 'second'
        result = {'tools':[{'name':name,'description':'Test echo tool','inputSchema':{'type':'object',
                  'properties':{'text':{'type':'string'}},'required':['text']}}]}
        if not cursor:
            result['nextCursor'] = 'page-2'
        emit({'jsonrpc':'2.0','id':request['id'],'result':result})
    elif method == 'tools/call':
        calls += 1
        result = {'content':[{'type':'text','text':request['params']['arguments'].get('text','')}],
                  'structuredContent':{'calls':calls,'secret_present':bool(os.environ.get('MNN_MCP_TEST_SECRET'))}}
        emit({'jsonrpc':'2.0','id':request['id'],'result':result})
    else:
        emit({'jsonrpc':'2.0','id':request['id'],'error':{'code':-32601,'message':'unsupported'}})
