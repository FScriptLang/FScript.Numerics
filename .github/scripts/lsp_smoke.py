"""Exercise completion/hover/diagnostics in the published VS Code server, not source builds."""
from pathlib import Path
import json
import os
import queue
import subprocess
import threading
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = '0.82.0'
server = Path(os.environ.get('FSCRIPT_RELEASE_SERVER_DIR', ROOT / '.out' / 'vscode' / 'extension' / 'server'))
if not (server / 'FScript.LanguageServer.dll').exists():
    assert 'FSCRIPT_RELEASE_SERVER_DIR' not in os.environ, f'Missing server: {server}'
    archive = ROOT / '.out' / f'fscript-{VERSION}.vsix'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve(f'https://github.com/MagnusOpera/FScript/releases/download/{VERSION}/fscript-{VERSION}.vsix', archive)
    with zipfile.ZipFile(archive) as package:
        for name in package.namelist():
            if name.startswith('extension/server/'):
                package.extract(name, ROOT / '.out' / 'vscode')

class Client:
    def __enter__(self):
        self.log = (ROOT / '.out' / 'lsp-stderr.log').open('wb')
        self.process = subprocess.Popen(['dotnet', str(server / 'FScript.LanguageServer.dll')],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, cwd=ROOT)
        self.messages = queue.Queue()
        self.identifier = 0
        def read():
            try:
                while True:
                    headers = {}
                    while (line := self.process.stdout.readline()) not in (b'\r\n', b'\n'):
                        if not line:
                            raise EOFError('Language server closed its output; see .out/lsp-stderr.log')
                        key, value = line.decode().split(':', 1)
                        headers[key.lower()] = value.strip()
                    length = int(headers['content-length'])
                    data = bytearray()
                    while len(data) < length:
                        chunk = self.process.stdout.read(length-len(data))
                        if not chunk:
                            raise EOFError('Incomplete JSON-RPC response')
                        data.extend(chunk)
                    self.messages.put(json.loads(data))
            except Exception as error:
                self.messages.put(error)
        threading.Thread(target=read, daemon=True).start()
        return self

    def send(self, method, params, identifier=None):
        message = {'jsonrpc':'2.0','method':method,'params':params}
        if identifier is not None:
            message['id'] = identifier
        data = json.dumps(message).encode()
        self.process.stdin.write(f'Content-Length: {len(data)}\r\n\r\n'.encode()+data)
        self.process.stdin.flush()

    def until(self, predicate):
        while True:
            message = self.messages.get(timeout=20)
            if isinstance(message, Exception):
                raise message
            if predicate(message):
                assert 'error' not in message, message
                return message

    def request(self, method, params):
        self.identifier += 1
        self.send(method, params, self.identifier)
        return self.until(lambda m: m.get('id') == self.identifier)['result']

    def open(self, uri, source):
        self.send('textDocument/didOpen', {'textDocument':{'uri':uri,'languageId':'fscript','version':1,'text':source}})
        return self.until(lambda m: m.get('method') == 'textDocument/publishDiagnostics' and m['params']['uri'] == uri)['params']['diagnostics']

    def __exit__(self, *args):
        self.process.kill()
        self.process.wait(timeout=10)
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()

with Client() as client:
    folder = ROOT / 'examples'
    client.request('initialize', {'processId':os.getpid(),'rootUri':folder.as_uri(),'capabilities':{},
        'initializationOptions':{'workspaceFolderUri':folder.as_uri(),'configurationPath':str(folder / 'fscript.toml'),'workspaceTrusted':True}})
    client.send('initialized', {})
    info = client.request('fscript/environmentInfo', {})
    assert not info['failed'] and not info['blockedByTrust'], info
    for script in folder.glob('*.fss'):
        assert not (diagnostics := client.open(script.as_uri(),script.read_text())), (script,diagnostics)
    uri = (folder / 'editor-check.fss').as_uri()
    assert not client.open(uri, 'let a = NDArray.ones [2;3]\nlet shape = a.Shape\nNDArray.sum a\n')
    def at(method, line, character):
        return json.dumps(client.request(method,{'textDocument':{'uri':uri},'position':{'line':line,'character':character}}))
    functions = at('textDocument/completion',2,8)
    assert 'NDArray.matmul' in functions and 'NDArray.set' in functions, functions
    fields = at('textDocument/completion',1,14)
    assert 'Shape' in fields and 'Payload' not in fields, fields
    assert 'int list' in (hover := at('textDocument/hover',1,16)), hover
    assert 'NDArray' in (hover := at('textDocument/hover',0,17)), hover
    assert client.open((folder / 'invalid-editor-check.fss').as_uri(),'NDArray.ofList [1;2]')
    print('Published VS Code server: manifest loading, examples, completion, hover, and authoritative diagnostics passed')
