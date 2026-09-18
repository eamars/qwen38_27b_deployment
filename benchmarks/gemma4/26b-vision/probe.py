import base64, io, json, urllib.request
from pathlib import Path
import struct, zlib
out=Path(__file__).resolve().parent
def post(body):
    req=urllib.request.Request('http://127.0.0.1:8094/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as r: return json.load(r)
def chunk(tag,data):
    return struct.pack('>I',len(data))+tag+data+struct.pack('>I',zlib.crc32(tag+data))
rows=b''.join(b'\0'+b''.join(b'\xff\0\0' if 280<=x<1256 and 280<=y<1256 else b'\xff\xff\xff' for x in range(1536)) for y in range(1536))
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',1536,1536,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b'')
b=io.BytesIO(png)
uri='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
for name,content in [('text','Write a Python function to return the first n Fibonacci numbers. Explain briefly.'),('vision',[{'type':'text','text':'What color and shape is shown? Answer briefly.'},{'type':'image_url','image_url':{'url':uri}}])]:
    result=post({'messages':[{'role':'user','content':content}],'max_tokens':256,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
    (out/(name+'.json')).write_text(json.dumps(result,indent=2))
    assert result['timings']['draft_n_accepted'] > 0
    if name == 'vision':
        answer = result['choices'][0]['message']['content'].lower()
        assert 'red' in answer and 'square' in answer
    print(name, json.dumps(result),flush=True)
