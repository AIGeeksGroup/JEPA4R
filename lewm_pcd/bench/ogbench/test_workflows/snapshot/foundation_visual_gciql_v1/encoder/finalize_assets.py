"""Wait for owned transport, then hash/CRC/finalize; never run a model."""
import json,pathlib,time,runpy,traceback,datetime
R=pathlib.Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/encoder')
A=pathlib.Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/assets/baselines/foundation/vjepa21_vitl1024_official_v1')
try:
 while not (R/'segments.actual_exit_code').exists():time.sleep(5)
 assert (R/'segments.actual_exit_code').read_text().strip()=='0'
 state=json.loads((R/'segments_status.json').read_text());assert state['exit_code']==0 and state['downloaded_bytes']==5151198524
 source=A/'vjepa2_1_vitl_dist_vitG_384.pt.part';target=A/'vjepa2_1_vitl_dist_vitG_384.pt'
 assert source.stat().st_size==5151198524 and not target.exists()
 source.rename(target)
 runpy.run_path(str(R/'prepare_assets.py'),run_name='__main__')
 (R/'finalize.actual_exit_code').write_text('0\n')
except BaseException as e:
 (R/'finalize_failure.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':1,'error':repr(e),'traceback':traceback.format_exc()},indent=2)+'\n');(R/'finalize.actual_exit_code').write_text('1\n');raise
