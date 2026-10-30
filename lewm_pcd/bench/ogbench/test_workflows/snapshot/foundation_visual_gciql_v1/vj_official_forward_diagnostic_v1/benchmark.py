"""Finite forward-only A/B/C diagnostic; never resumes a producer or writes its cache."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback

ATOL=RTOL=1e-5
SPEC_SHA='c8323e5c917549115e29d9374d1f453e5c1fce2b69982a2aadd7ca247a217a99'
ADAPTER_SHA='3f5cb292728c809737ff0dbe851095cec8825883b0f009d913d315677f462bfe'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2)
def validate_options(args):
    assert args.warmups==2 and args.trials==5,'This finite diagnostic is fixed at 2 warmups + 5 timed batches'
    if args.execute:
        assert args.input
        assert args.spec and args.adapter and args.output and args.paused_pids and args.protected_cache
        if args.official_autocast:assert args.official_evidence and args.official_evidence_sha256
def validate_pixels(value,layout):
    import numpy as np
    expected=(32,224,224,3) if layout=='BHWC' else (32,3,224,224)
    assert value.shape==expected and value.dtype==np.uint8,'Expected exactly 32 real uint8 RGB224 frames'

def proc_identity(pid):
    raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rindex(')')+2:].split()
    assert fields[0]=='T',f'Old process {pid} is not stopped: {fields[0]}'
    return dict(pid=pid,state=fields[0],start_ticks=int(fields[19]))
def cache_stats(root):
    root=Path(root)
    files={str(p.relative_to(root)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,inode=p.stat().st_ino)
           for p in root.rglob('*') if p.is_file()}
    return dict(files=files,episode_receipts=sum(name.endswith('/receipt.json') for name in files),
                latent_files=sum(name.endswith('/latent.npy') for name in files))
def safety_check(pids,identities):
    assert [proc_identity(pid) for pid in pids]==identities,'Old paused process identity/state changed'
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
    active={int(line.strip()) for line in query.stdout.splitlines() if line.strip().isdigit()}
    assert active<=set(pids)|{os.getpid()},f'External GPU process: {active-set(pids)-{os.getpid()}}'

def fingerprint(model):
    value=hashlib.sha256()
    for key,tensor in model.state_dict().items():
        array=tensor.detach().cpu().numpy()
        value.update(key.encode()+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
    return value.hexdigest()
def rng_hash(torch,np):
    value=(random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes(),
           [state.numpy().tobytes() for state in torch.cuda.get_rng_state_all()])
    return hashlib.sha256(pickle.dumps(value)).hexdigest()
def metrics(torch,actual,reference):
    a,b=actual.float(),reference.float();diff=a-b
    cosine=torch.nn.functional.cosine_similarity(a.reshape(len(a),-1),b.reshape(len(b),-1),dim=1)
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return dict(finite=finite,bitwise_equal=bool(torch.equal(actual,reference)),max_abs=float(diff.abs().max()),
        mean_abs=float(diff.abs().mean()),RMSE=float(diff.square().mean().sqrt()),cosine_mean=float(cosine.mean()),
        cosine_min=float(cosine.min()),atol=ATOL,rtol=RTOL,fixed_1e5_allclose=finite and bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL)),
        actual_dtype=str(actual.dtype),reference_dtype=str(reference.dtype))
def measure(torch,function,warmups,trials):
    for _ in range(warmups):function()
    torch.cuda.synchronize();walls=[];events=[]
    for _ in range(trials):
        torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter();start.record();value=function();end.record();torch.cuda.synchronize()
        walls.append((time.perf_counter()-wall)*1000);events.append(start.elapsed_time(end));del value
    return dict(wall_ms=walls,cuda_event_ms=events,wall_median_ms=statistics.median(walls),cuda_event_median_ms=statistics.median(events),
        warmups=warmups,trials=trials,CUDA_synchronized_each_trial=True)

def measure_arm(torch,encoder,function):
    for _ in range(2):function()
    timings={key:[] for key in ('wall_ms','cuda_event_ms','preprocess_H2D_ms','complete_forward_ms','pool_output_validation_ms')}
    current=[]
    pre=encoder.register_forward_pre_hook(lambda module,inputs:current[1].record())
    post=encoder.register_forward_hook(lambda module,inputs,value:current[2].record())
    try:
        for _ in range(5):
            torch.cuda.synchronize();current[:]=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
            wall=time.perf_counter();current[0].record();value=function();current[3].record();torch.cuda.synchronize()
            timings['wall_ms'].append((time.perf_counter()-wall)*1000)
            for key,start,end in [('cuda_event_ms',0,3),('preprocess_H2D_ms',0,1),('complete_forward_ms',1,2),('pool_output_validation_ms',2,3)]:
                timings[key].append(current[start].elapsed_time(current[end]))
            del value
    finally:pre.remove();post.remove()
    return dict(samples=timings,medians={key:statistics.median(values) for key,values in timings.items()},warmups=2,trials=5,
        components_from_same_forward=True,A_tail_includes_original_finite_dtype_checks=True,CUDA_synchronized_each_trial=True)

def backend_profile(torch,function):
    try:
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:function()
        torch.cuda.synchronize()
        names=sorted({item.key for item in profile.key_averages() if any(key in item.key.lower() for key in ('scaled_dot_product','flash_attention','efficient_attention','cudnn_attention'))})
        return dict(available=True,CPU_activity_only=True,extra_forward_calls=1,not_used_for_timing=True,operator_names=names)
    except Exception as error:return dict(available=False,error=repr(error),not_used_for_timing=True,backend_not_modified=True)

def execute(args):
    import numpy as np
    import torch
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    code=1;stop=threading.Event();violations=[];baseline=None;pids=[];identities=[]
    try:
        pids=[int(value) for value in args.paused_pids.split(',')];assert len(pids)==len(set(pids))==11
        identities=[proc_identity(pid) for pid in pids];safety_check(pids,identities)
        baseline=cache_stats(args.protected_cache)
        assert Path(args.protected_cache).is_dir() and baseline['episode_receipts']==59
        save(output/'paused_baseline.json',dict(processes=identities,cache=baseline,at=time.time(),not_old_54_episode_snapshot=True))
        assert sha(args.spec)==SPEC_SHA and sha(args.adapter)==ADAPTER_SHA
        spec=read(args.spec);assert spec['family']=='vjepa21_vitl1024' and spec['native_size']==384 and spec['output_dim']==1024
        assert spec['constructor_kwargs']['use_sdpa'] is True and spec['constructor_kwargs']['img_temporal_dim_size']==1
        official=None
        if args.official_autocast:
            assert sha(args.official_evidence)==args.official_evidence_sha256
            official=read(args.official_evidence)
            assert official['commit']==spec['source_commit']
            evidence={item['path']:item['fetched_UTF8_SHA256'] for item in official['sources']}
            assert evidence['evals/image_classification_frozen/eval.py']=='ff35b2729d45fc6b212275bec580704673b69058b00064f6e54b90e01e1a50e0'
            assert evidence['configs/eval_2_1/vitl-384/in1k.yaml']=='c9e378792ae3437ca77d3c9d6f7ff3f448128312cca34c25b4718a1365937129'
            assert any(item['verified'] is True and item['claim']=='Official frozen imageeval is differentprotocol' for item in official['findings'])
            assert 'FP16autocast' in official['allowed_diagnostic_candidate']
        if args.input:
            assert sha(args.input)==args.input_sha256,'Saved raw fixture SHA must be explicit'
            if Path(args.input).suffix=='.npz':
                with np.load(args.input,allow_pickle=False) as saved:pixels=saved[args.input_key].copy()
            else:
                assert Path(args.input).suffix=='.npy';pixels=np.load(args.input,allow_pickle=False)
            layout=args.layout;fixture=dict(path=str(args.input),sha256=sha(args.input),key=args.input_key)
        validate_pixels(pixels,layout)
        pixels=np.ascontiguousarray(pixels);raw=torch.from_numpy(pixels)
        if layout=='BHWC':raw=raw.permute(0,3,1,2).contiguous()
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(1)
        module_spec=importlib.util.spec_from_file_location('_diagnostic_frozen_adapter',args.adapter)
        adapter=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(adapter)
        model=adapter.load_encoder(args.spec,device='cuda')
        assert next(model.encoder.parameters()).dtype==torch.float32 and not model.training
        before=fingerprint(model);rng_before=rng_hash(torch,np)
        save(output/'config.json',dict(spec_sha256=SPEC_SHA,adapter_sha256=ADAPTER_SHA,fixture=fixture,
            raw_batch_shape=list(raw.shape),raw_dtype=str(raw.dtype),native_size=384,expected_tokens=[32,576,1024],
            temporal_input=1,feature='last_normalized_image_patch_mean',full_FOV=True,seed=0,torch=torch.__version__,CUDA=torch.version.cuda,
            device=torch.cuda.get_device_name(),Torch_threads=torch.get_num_threads(),warmups=2,trials=5,
            official_autocast_evidence=official,official_evidence_sha256=args.official_evidence_sha256,old_cache_not_modified=True,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),precision_flags_modified=False))
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            x=model.preprocess_uint8(raw);assert tuple(x.shape)==(32,3,1,384,384) and x.dtype==torch.float32
            captured={}
            hook=model.encoder.register_forward_hook(lambda module,inputs,result:captured.update(input=inputs[0],tokens=result))
            try:a_embedding=model.encode_uint8(raw)
            finally:hook.remove()
            a_tokens=captured['tokens'];assert torch.equal(captured['input'],x),'A and B preprocessing tensors differ'
            b_tokens=model.encoder(x);b_embedding=b_tokens.mean(dim=1)
            assert tuple(a_tokens.shape)==tuple(b_tokens.shape)==(32,576,1024)
            ab=dict(tokens=metrics(torch,b_tokens,a_tokens),embedding=metrics(torch,b_embedding,a_embedding))
            assert ab['tokens']['bitwise_equal'] and ab['embedding']['bitwise_equal'],'FP32 A/B strict equivalence failed'
            arms={}
            for label,half in [('A_wrapper_FP32',False),('B_complete_official_FP32',False)]+([('C_official_autocast_FP16',True)] if args.official_autocast else []):
                safety_check(pids,identities)
                def forward():
                    with torch.autocast('cuda',dtype=torch.float16,enabled=half):return model.encoder(x)
                def pool(value):return value.mean(dim=1)
                def direct_end_to_end():
                    prepared=model.preprocess_uint8(raw)
                    with torch.autocast('cuda',dtype=torch.float16,enabled=half):tokens=model.encoder(prepared);feature=pool(tokens)
                    return feature.float()
                tokens=forward();embedding=pool(tokens)
                full=model.encode_uint8 if label.startswith('A_') else None
                call=(lambda:full(raw)) if full else direct_end_to_end
                timing=measure_arm(torch,model.encoder,call)
                numeric=dict(tokens=metrics(torch,tokens,a_tokens),embedding=metrics(torch,embedding,a_embedding))
                profiler=backend_profile(torch,call) if label.startswith(('A_','C_')) else dict(skipped=True)
                safety_check(pids,identities)
                arms[label]=dict(timings=timing,SDPA_backend_CPU_operator_profile=profiler,numerics_vs_A=numeric,
                    forward_autocast='float16' if half else 'disabled_FP32',tokens_dtype=str(tokens.dtype),pool_result_dtype=str(embedding.dtype),
                    persisted_embedding_dtype='float32',storage_FP32_is_not_compute_FP32=half,
                    fixed_tolerance_equivalent=numeric['tokens']['fixed_1e5_allclose'] and numeric['embedding']['fixed_1e5_allclose'],
                    candidate_status='equivalent_at_original_tolerance' if numeric['tokens']['fixed_1e5_allclose'] and numeric['embedding']['fixed_1e5_allclose'] else 'not_equivalent_at_original_tolerance')
                del tokens,embedding
            assert fingerprint(model)==before and rng_hash(torch,np)==rng_before,'Frozen model or RNG changed'
        safety_check(pids,identities);assert cache_stats(args.protected_cache)==baseline and not violations
        save(output/'report.json',dict(actual_exit_code=0,diagnostic_execution_completed=True,AB_strict_equal=ab,arms=arms,
            model_state_SHA_before_after=before,RNG_before_after=rng_before,old_cache_stat_before_after_exact=True,
            paused_processes_stayed_T=True,full_cache_started=False,training_updates=0,SR_episodes=0,
            no_candidate_automatically_adopted=True,no_resolution_history_pool_or_SDPA_changes=True,at=time.time()))
        code=0
    except BaseException:
        save(output/'failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),monitor_violations=violations,at=time.time()));raise
    finally:
        stop.set()
        if baseline is not None:
            try:after=dict(cache_stat_equal=cache_stats(args.protected_cache)==baseline,paused_processes=[proc_identity(pid) for pid in pids],violations=violations)
            except BaseException as error:after=dict(readback_error=repr(error),violations=violations)
            save(output/'after_safety_readback.json',after)
        with (output/'outer_exit_code').open('x') as stream:stream.write(str(code)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--self-test',action='store_true')
    for name in ('spec','adapter','output','protected-cache','official-evidence'):p.add_argument('--'+name,type=Path)
    p.add_argument('--input','--raw-rgb',dest='input',type=Path)
    p.add_argument('--input-key',default='pixels');p.add_argument('--input-sha256');p.add_argument('--layout',choices=('BHWC','BCHW'),default='BHWC')
    p.add_argument('--paused-pids');p.add_argument('--official-evidence-sha256');p.add_argument('--official-autocast',action='store_true');p.add_argument('--warmups',type=int,default=2);p.add_argument('--trials',type=int,default=5)
    args=p.parse_args();validate_options(args)
    if args.self_test:
        import numpy as np
        validate_pixels(np.zeros((32,224,224,3),np.uint8),'BHWC')
        for bad in (np.zeros((1,224,224,3),np.uint8),np.zeros((32,224,224,3),np.float32)):
            try:validate_pixels(bad,'BHWC')
            except AssertionError:pass
            else:raise AssertionError('Bad input accepted')
        bad=argparse.Namespace(**vars(args));bad.trials=6
        try:validate_options(bad)
        except AssertionError:pass
        else:raise AssertionError('Unbounded diagnostic accepted')
        print(json.dumps(dict(actual_exit_code=0,CPU_options_and_input_shape_checks=True,NN_forward=False)));return
    if not args.execute:
        print(json.dumps(dict(prepared_only=True,batch=32,warmups=2,timed_batches=5,A_B='FP32 strict same full encoder/tensor/pool',
            C='optional confirmed official FP16 autocast; original tolerance unchanged',NN_forward=False)));return
    execute(args)

if __name__=='__main__':main()
