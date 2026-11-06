#!/usr/bin/env python3
"""Minimal stateful MCP Streamable HTTP server for local protocol tests."""
import http.server
import json
import os
import time


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    calls = 0

    def log_message(self, *args):
        pass

    def respond(self, status, body=b'', headers=None):
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)
            self.wfile.flush()

    def do_POST(self):
        if self.headers.get('X-Test-Secret') != 'http-private-secret':
            self.respond(401)
            return
        message = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        method = message.get('method')
        if method != 'initialize' and (self.headers.get('Mcp-Session-Id') != 'fixture-session' or
                                       self.headers.get('MCP-Protocol-Version') != '2025-06-18'):
            self.respond(400)
            return
        if method == 'notifications/initialized':
            self.respond(202)
            return
        if method == 'initialize':
            with open(os.environ['MNN_MCP_HTTP_STARTED'], 'w') as marker:
                marker.write('started')
            result = {'protocolVersion':'2025-06-18','capabilities':{'tools':{}},'serverInfo':{'name':'http-fixture','version':'1'}}
        elif method == 'tools/list':
            result = {'tools':[{'name':'remote_echo','description':'Echo through HTTP','inputSchema':{'type':'object',
                       'properties':{'text':{'type':'string'}},'required':['text']}}]}
        elif method == 'tools/call':
            Handler.calls += 1
            result = {'content':[{'type':'text','text':message['params']['arguments']['text']}],
                      'structuredContent':{'calls':Handler.calls}}
        else:
            self.respond(400)
            return
        payload = json.dumps({'jsonrpc':'2.0','id':message['id'],'result':result}).encode()
        if method == 'tools/list':
            body = b'event: message\r\ndata: ' + payload + b'\r\n\r\n'
            self.respond(200, body, {'Content-Type':'text/event-stream'})
        else:
            headers = {'Content-Type':'application/json'}
            if method == 'initialize':
                headers['Mcp-Session-Id'] = 'fixture-session'
            self.respond(200, payload, headers)

    def do_DELETE(self):
        if self.headers.get('Mcp-Session-Id') != 'fixture-session':
            self.respond(400)
            return
        if os.environ.get('MNN_MCP_HTTP_DELETE_STARTED'):
            with open(os.environ['MNN_MCP_HTTP_DELETE_STARTED'], 'w') as marker:
                marker.write('started')
            time.sleep(2)
        with open(os.environ['MNN_MCP_HTTP_CLOSED'], 'w') as marker:
            marker.write('closed')
        self.respond(200)


server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
with open(os.environ['MNN_MCP_HTTP_PORT'], 'w') as port:
    port.write(str(server.server_port))
server.serve_forever()
