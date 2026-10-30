import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from terminal_producer import terminal_raw_ids, padding_for64, validate_payload, validate_identity, parse_args, classify

def test_terminal_raw999_to_next1000_preserves_episode_and_split():
    ids=terminal_raw_ids(1000)
    assert ids.dtype==np.dtype('int64') and ids.shape==(1000,)
    assert ids[0]==999+1==1000 and ids[1]==2001 and ids[-1]==1000999
    assert np.array_equal(ids//1001,np.arange(1000)) and np.all(ids%1001==1000)
    assert terminal_raw_ids(100)[-1]==100099
    # Compact row1000 is next episode's current0, not raw terminal1000.
    assert 1000//1000*1001+1000%1000==1001 and ids[0]!=1001

def test_source_terminal_identity_required_and_action_not_a_next_input():
    fields=dict(source_row=2001,source_episode=1,source_step=1000,source_terminal=True,valid_transition=False)
    validate_identity(fields,1)
    # An action may be NaN at terminal; it is neither read nor normalized into next input.
    validate_identity({**fields,'action':np.full(5,np.nan)},1)
    for field,value in [('source_row',2002),('source_episode',2),('source_step',999),('source_terminal',False),('valid_transition',True)]:
        with pytest.raises(ValueError):validate_identity({**fields,field:value},1)

def test_fixed64_tail_matches_real_split_and_refuses_underfilled_padding():
    assert padding_for64(64)==0 and padding_for64(40)==24 and padding_for64(36)==28
    with pytest.raises(ValueError):padding_for64(12)

def test_payload_real_ids_finite_FP32_no_zero_fill_or_cross_episode():
    ids=terminal_raw_ids(2);z=np.arange(384,dtype=np.float32).reshape(2,192)
    validate_payload(ids,z,2)
    for bad_ids,bad_z in [(np.array([1000,1001],np.int64),z),(ids.astype(np.float64),z),(ids,z.astype(np.float64)),(ids,np.zeros((2,191),np.float32)),(ids,np.full((2,192),np.nan,np.float32))]:
        with pytest.raises(ValueError):validate_payload(bad_ids,bad_z,2)

def test_default_no_execution_and_no_model_import():
    args=parse_args(['--plan','prepared.json'])
    assert not args.execute
    assert 'torch' not in sys.modules and 'jax' not in sys.modules

def test_historical_RGB_missing_modality_strictly_identified_not_rewritten():
    old=dict(source_rgb_index='real_RGB_index.json')
    assert classify(old)=='rgb' and 'source_modality' not in old
    assert classify(dict(source_modality='pcd',source_pcd_index='real_PCD_index.json'))=='pcd'
    for bad in [dict(),dict(source_modality='pixels',source_rgb_index='rgb'),dict(source_rgb_index='rgb',source_pcd_index='pcd')]:
        with pytest.raises(ValueError):classify(bad)
