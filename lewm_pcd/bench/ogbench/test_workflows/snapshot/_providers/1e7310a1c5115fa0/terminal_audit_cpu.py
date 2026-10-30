"""CPU audit of only1100 new vectors/ledger; original HDF/main cache never scanned."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from terminal_producer import sha,read,save,validate_payload,validate_identity
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--producer-outer-exit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:
        m=read(a.manifest);plan=read(a.plan);old=read(plan['cache_manifest'])
        if a.producer_outer_exit.read_text().strip()!='0' or m['status']!='passed' or m['exit_code']!=0 or m['plan_sha256']!=sha(a.plan):raise ValueError('Actual closed producer and exactplan required')
        if m['source_modality']!=plan['modality'] or m['modality']!=plan['modality'] or m['raw_id_mapping']!='episode*1001+1000' or m['cache_manifest_sha256']!=sha(plan['cache_manifest']) or m['cache_seal_sha256']!=sha(plan['cache_seal']):raise ValueError('Canonical modality/seal binding')
        for key in ['source_checkpoint_sha256','preprocess_source_sha256','encoder_hashes','precision']:
            if m[key]!=old[key]:raise ValueError('Original exact metadata differed '+key)
        if m['producer_source_sha256']!=sha(Path(__file__).with_name('terminal_producer.py')):raise ValueError('Actual producer source')
        for split,n in [('train',1000),('val',100)]:
            r=m['splits'][split];p=Path(r['path'])
            if sha(p)!=r['sha256']:raise ValueError('Real sidecar SHA')
            with np.load(p,allow_pickle=False) as z:
                if set(z.files)!={'raw_ids','latents'}:raise ValueError('Only declared arrays required')
                ids,lat=z['raw_ids'],z['latents'];validate_payload(ids,lat,n)
                if hashlib.sha256(ids.astype('<i8',copy=False).tobytes()).hexdigest()!=r['raw_ids_sha256'] or r['shape']!=[n,192] or r['dtype']!='float32' or r['raw_ids_shape']!=[n] or r['raw_ids_dtype']!='int64':raise ValueError('Sidecar row/header metadata')
        if sha(m['source_row_ledger'])!=m['source_row_ledger_sha256']:raise ValueError('Source ledger SHA')
        records=[json.loads(x) for x in Path(m['source_row_ledger']).read_text().splitlines()]
        if len(records)!=1100:raise ValueError('Exactly1100 source frames')
        for split,n in [('train',1000),('val',100)]:
            rows=[x for x in records if x['split']==split]
            if [x['episode'] for x in rows]!=list(range(n)):raise ValueError('Source IDs not exact ordered split')
            for r in rows:
                validate_identity(r['source_identity'],r['episode'])
                if r['raw_id']!=r['episode']*1001+1000 or r['source_stat_before']!=r['source_stat_after']:raise ValueError('Real terminal source metadata')
        report=dict(status='passed_terminal_next_sidecar_CPU_only',exit_code=0,real_terminal_frames=1100,finite_FP32_latent192=True,source_raw_ids_exact=True,manifest_sha256=sha(a.manifest),producer_outer_actual_exit=0,cache_precision_encoder_hash_metadata_exact=True,original_HDF_or_maincache_content_scanned=False,model_or_GPU_used=False,scope_limit='Input frame/model numeric production and boundary comparison evidenced by frozen producer ledger; this audit does not rerender/encode or certify GCIQL update semantics')
        save(a.output/'report.json',report);(a.output/'inner_exit_code').write_text('0\n');print(json.dumps(report,indent=2))
    except BaseException:
        import traceback
        save(a.output/'status.json',dict(status='failed',exit_code=1,error=traceback.format_exc()));(a.output/'inner_exit_code').write_text('1\n');raise
if __name__=='__main__':main()
