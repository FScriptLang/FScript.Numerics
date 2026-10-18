// Actual published CLI adapter, including terminal-host launch and opaque numerical inspection.
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const root=process.cwd(),folder=process.env.NUMERICS_SMOKE_FOLDER??path.join(root,'examples');
const program=path.join(folder,'debug-check.fss');
fs.writeFileSync(program,'let data = NDArray.ones [1000000]\nlet x = NDArray.ofRows [[1.0];[2.0]]\nlet model = StandardScaler.fit x\nlet product = BLAS.matmul (NDArray.identity 2) x\nNDArray.sum product\n');
const adapter=spawn('dotnet',['tool','run','fscript','--','--debug-adapter'],{cwd:root,stdio:['pipe','pipe','pipe']});
let buffer=Buffer.alloc(0),sequence=0,errors='';const pending=new Map(),messages=[],waits=[],children=[];
function write(message){message.seq=++sequence;const body=Buffer.from(JSON.stringify(message));adapter.stdin.write(Buffer.concat([Buffer.from(`Content-Length: ${body.length}\r\n\r\n`),body]));return sequence;}
function request(command,args={}){return new Promise((resolve,reject)=>{const seq=write({type:'request',command,arguments:args});const timer=setTimeout(()=>reject(new Error(`${command} timed out: ${errors}`)),30000);pending.set(seq,{resolve,timer});});}
function wait(event){const index=messages.findIndex(m=>m.event===event);if(index>=0)return Promise.resolve(messages.splice(index,1)[0]);return new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error(`${event} timed out: ${errors}`)),30000);waits.push({event,resolve,timer});});}
adapter.stderr.on('data',b=>errors+=b);
adapter.stdout.on('data',data=>{buffer=Buffer.concat([buffer,data]);while(true){const end=buffer.indexOf('\r\n\r\n');if(end<0)return;const size=Number(/Content-Length:\s*(\d+)/i.exec(buffer.subarray(0,end).toString())[1]);if(buffer.length<end+4+size)return;const message=JSON.parse(buffer.subarray(end+4,end+4+size));buffer=buffer.subarray(end+4+size);
 if(message.type==='request'&&message.command==='runInTerminal'){const child=spawn(message.arguments.args[0],message.arguments.args.slice(1),{cwd:message.arguments.cwd,env:process.env,stdio:['pipe','pipe','pipe']});children.push(child);child.stderr.on('data',b=>errors+=b);child.stdout.on('data',()=>{});write({type:'response',request_seq:message.seq,command:message.command,success:true,body:{processId:child.pid}});}
 else if(message.type==='response'&&pending.has(message.request_seq)){const p=pending.get(message.request_seq);pending.delete(message.request_seq);clearTimeout(p.timer);p.resolve(message);}
 else {const index=waits.findIndex(w=>w.event===message.event);if(index>=0){const w=waits.splice(index,1)[0];clearTimeout(w.timer);w.resolve(message);}else messages.push(message);}
}});
async function checked(command,args){const response=await request(command,args);assert.equal(response.success,true,response.message);return response.body;}
try{
 await checked('initialize',{adapterID:'fscript',supportsRunInTerminalRequest:true,linesStartAt1:true,columnsStartAt1:true});
 await checked('launch',{program,cwd:folder,configurationPath:path.join(folder,'fscript.toml')});await wait('initialized');
 const points=await checked('setBreakpoints',{source:{path:program},breakpoints:[{line:5}]});assert.equal(points.breakpoints[0].verified,true);
 await checked('configurationDone');const stop=await wait('stopped');assert.equal(stop.body.reason,'breakpoint');const stack=await checked('stackTrace',{threadId:stop.body.threadId});const frameId=stack.stackFrames[0].id;assert.equal(stack.stackFrames[0].line,5);
 const array=await checked('evaluate',{frameId,expression:'data'});assert.ok(array.variablesReference>0);
 const fields=await checked('variables',{variablesReference:array.variablesReference});assert.equal(fields.variables.length,9);assert.ok(fields.variables.some(v=>v.name==='Size'&&v.value==='1000000'));assert.ok(!JSON.stringify(fields).includes('Payload'));assert.ok(JSON.stringify(fields).length<6000);
 const model=await checked('evaluate',{frameId,expression:'model'});const metadata=await checked('variables',{variablesReference:model.variablesReference});assert.deepEqual(metadata.variables.map(v=>v.name).sort(),['FeatureCount','Id']);
 assert.equal((await request('evaluate',{frameId,expression:'data.Payload'})).success,false);
 await checked('continue',{threadId:stop.body.threadId});await wait('terminated');console.log('Published CLI debugger: NuGet/native execution and bounded array/model inspection passed');
}finally{
 adapter.stdin.end();adapter.kill();for(const child of children)child.kill();for(const p of pending.values())clearTimeout(p.timer);for(const w of waits)clearTimeout(w.timer);fs.rmSync(program,{force:true});
}
