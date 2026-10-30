"""Fixed-input FP16/BF16 forward-only comparison. No producer or adoption."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import traceback

BENCHMARK_SHA='3cb9a822987a2ad69127de49c3f5d1a9642b78fd86ab33accb90169e8ec0489e'
FIXTURE_SHA='429d502c0c7dd5ed5f08152efd7c35fb031689acaa5606bcb98a493f71e164c3'

def utilities():
    path=Path(__file__).with_name('benchmark.py')
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest()==BENCHMARK_SHA
    spec=importlib.util.spec_from_file_location('_frozen_forward_diagnostic_utils',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def timed(torch,call):
    for _ in range(2):call()
    torch.cuda.synchronize();wall=[];events=[];last=None
    for _ in range(5):
        torch.cuda.synchronize()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        began=time.perf_counter();start.record();last=call();end.record();torch.cuda.synchronize()
        wall.append((time.perf_counter()-began)*1000);events.append(start.elapsed_time(end))
    import statistics
    return dict(warmups=2,trials=5,wall_ms=wall,cuda_event_ms=events,
        wall_median_ms=statistics.median(wall),cuda_event_median_ms=statistics.median(events)),last

def execute(args,u):
    import numpy as np
    import torch
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    code=1;baseline=None;pids=[];identities=[]
    try:
        pids=[int(v) for v in args.paused_pids.split(',')]
        assert len(pids)==len(set(pids))==11
        identities=[u.proc_identity(pid) for pid in pids];u.safety_check(pids,identities)
        baseline=u.cache_stats(args.protected_cache)
        assert baseline['episode_receipts']==baseline['latent_files']==59
        assert u.sha(args.spec)==u.SPEC_SHA and u.sha(args.adapter)==u.ADAPTER_SHA
        assert u.sha(args.input)==FIXTURE_SHA
        pixels=np.load(args.input,allow_pickle=False);u.validate_pixels(pixels,'BHWC')
        raw=torch.from_numpy(np.ascontiguousarray(pixels)).permute(0,3,1,2).contiguous()
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(1)
        supported=bool(torch.cuda.is_bf16_supported())
        u.save(out/'config.json',dict(fixture_SHA256=FIXTURE_SHA,spec_SHA256=u.SPEC_SHA,adapter_SHA256=u.ADAPTER_SHA,
            benchmark_utils_SHA256=BENCHMARK_SHA,torch=torch.__version__,CUDA=torch.version.cuda,
            capability=list(torch.cuda.get_device_capability()),bf16_supported=supported,
            batch=32,raw_shape=list(raw.shape),native_size=384,T=1,full_FOV=True,pool='normalized_patch_mean_then_float32',
            dtype_candidates=['float16','bfloat16'],warmups=2,trials=5,atol=1e-5,rtol=1e-5,
            no_gradient_optimizer_render_fullcache_or_signal=True,paused_processes=identities,
            float32_matmul_precision=torch.get_float32_matmul_precision(),TF32=torch.backends.cuda.matmul.allow_tf32))
        assert supported,'BF16 unsupported: do not silently substitute another precision'
        spec=importlib.util.spec_from_file_location('_original_encoder_adapter',args.adapter)
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        model=adapter.load_encoder(args.spec,device='cuda')
        before=u.fingerprint(model);rng_before=u.rng_hash(torch,np)
        u.save(out/'paused_baseline.json',dict(processes=identities,cache=baseline,model_state_SHA256=before,RNG_SHA256=rng_before))
        arms={}
        with torch.inference_mode():
            # Exactly one FP32 B32 reference, same native preprocess and model.
            with torch.autocast('cuda',enabled=False):
                reference_input=model.preprocess_uint8(raw)
                assert tuple(reference_input.shape)==(32,3,1,384,384)
                reference_tokens=model.encoder(reference_input)
                assert tuple(reference_tokens.shape)==(32,576,1024)
                reference_embedding=reference_tokens.mean(dim=1).float()
            for label,dtype in [('FP16',torch.float16),('BF16',torch.bfloat16)]:
                u.safety_check(pids,identities)
                def call(value=raw,compute_dtype=dtype):
                    with torch.autocast('cuda',enabled=False):prepared=model.preprocess_uint8(value)
                    with torch.autocast('cuda',dtype=compute_dtype):
                        tokens=model.encoder(prepared)
                        embedding=tokens.mean(dim=1).float()
                    return tokens,embedding
                timing,(tokens,embedding)=timed(torch,call)
                assert tuple(tokens.shape)==(32,576,1024) and tuple(embedding.shape)==(32,1024)
                single_tokens,single_embedding=call(raw[:1])
                b1=dict(tokens=u.metrics(torch,single_tokens,tokens[:1]),embedding=u.metrics(torch,single_embedding,embedding[:1]))
                error=dict(tokens=u.metrics(torch,tokens,reference_tokens),embedding=u.metrics(torch,embedding,reference_embedding))
                profile=u.backend_profile(torch,call)
                arms[label]=dict(timing=timing,vs_FP32=error,B1_vs_B32_original_tolerance=b1,
                    actual_tokens_dtype=str(tokens.dtype),persisted_embedding_dtype=str(embedding.dtype),
                    compute_autocast=str(dtype),SDPA_CPU_operator_profile=profile,
                    production_embedding_B1_B32_pass=b1['embedding']['fixed_1e5_allclose'],
                    extra_token_B1_B32_pass=b1['tokens']['fixed_1e5_allclose'],
                    FP32_equivalent=error['tokens']['fixed_1e5_allclose'] and error['embedding']['fixed_1e5_allclose'])
                del tokens,embedding,single_tokens,single_embedding
                u.safety_check(pids,identities)
            assert u.fingerprint(model)==before and u.rng_hash(torch,np)==rng_before
        u.safety_check(pids,identities);assert u.cache_stats(args.protected_cache)==baseline
        u.save(out/'report.json',dict(actual_exit_code=0,diagnostic_execution_completed=True,arms=arms,
            FP32_reference_batches=1,model_state_SHA_before_after=before,RNG_before_after=rng_before,
            old59_cache_stat_before_after_exact=True,paused11_processes_T_at_boundaries=True,
            no_candidate_adopted=True,fullcache_training_SR_started=False,
            speed_ratio_FP16_over_BF16=arms['FP16']['timing']['wall_median_ms']/arms['BF16']['timing']['wall_median_ms'],
            scope='single fixedB32 realRGB fixture; not modelSR or fullproducer throughput',at=time.time()))
        code=0
    except BaseException:
        u.save(out/'failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),at=time.time()));raise
    finally:
        if baseline is not None:
            try:after=dict(cache_stat_exact=u.cache_stats(args.protected_cache)==baseline,processes=[u.proc_identity(pid) for pid in pids])
            except BaseException as error:after=dict(readback_error=repr(error))
            u.save(out/'after_safety_readback.json',after)
        with (out/'outer_exit_code').open('x') as stream:stream.write(str(code)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    for name in ('spec','adapter','input','protected-cache','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--paused-pids');args=p.parse_args()
    u=utilities()
    if not args.execute:
        print(json.dumps(dict(actual_exit_code=0,prepared_only=True,NN_forward=False,batch=32,warmups=2,trials=5,FP32_reference_batches=1,
            candidates=['FP16','BF16'],fixture_SHA256=FIXTURE_SHA,original_tolerance=1e-5)));return
    assert all(getattr(args,name) is not None for name in ('spec','adapter','input','protected_cache','output','paused_pids'))
    assert args.output.name=='runtime_bf16_compare_v1','Use only the new exclusive diagnostic output'
    return execute(args,u)

if __name__=='__main__':main()
