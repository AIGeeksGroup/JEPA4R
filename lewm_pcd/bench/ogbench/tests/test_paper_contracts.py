"""Public model and Cube contracts; no model or simulator execution."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bench.ogbench.collection import build_plan
from bench.ogbench.workflow_downstream import group_contract


def test_current_model_widths_and_goals():
    expected={'lewm_rgb':192,'lewm_pcd':192,'pldm':512,'lejepa':512,'visreg':768,'vjepa':1024}
    for model,width in expected.items():
        for cube in (1,2,3,4):
            contract=group_contract(model,cube)['input_contract']
            assert contract['latent_dim']==width
            assert contract['current_dim']==width+19
            assert contract['goal_dim']==width+3*cube
            assert contract['zero_padding'] is False


def test_formal_collection_uses_full_horizons_and_independent_splits(tmp_path):
    for cube,horizon in enumerate((200,500,1000,600),1):
        plan=build_plan(cube,tmp_path/str(cube))
        assert plan['protocol']=='paper_cube_fixed_mix_v1'
        assert plan['raw_horizon']==plan['collection_protocol']['training_view_rows']==horizon
        assert plan['split_counts']=={'train':10000,'validation':1000}
        assert len(plan['parts'])==125


def test_smoke_is_an_explicit_small_budget(tmp_path):
    for cube in (1,2,3,4):
        plan=build_plan(cube,tmp_path/str(cube),mode='smoke')
        assert plan['split_counts']=={'train':2,'validation':1}
        assert len(plan['parts'])==3
        assert not (tmp_path/str(cube)).exists()
