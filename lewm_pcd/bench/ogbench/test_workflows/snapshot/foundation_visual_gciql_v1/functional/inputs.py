"""Width and Cube membership overlay; original sampling/reward AST stays exact."""
import ast
import hashlib
from pathlib import Path
import sys
import types

SOURCE_SHA = 'b9906dc2f4e7a38826f5ee05031713618a6ce408ee5555302801505452758f90'


def overlay_source(raw, dim):
    assert dim in (768, 1024) and hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    source = raw.decode()
    edits = [('192', str(dim), 9), ('211', str(dim+19), 1),
             ('self.cubes not in (2,3)', 'self.cubes not in (1,2,3,4)', 1)]
    result = source
    for before, after, count in edits:
        assert result.count(before) == count, (before, result.count(before))
        result = result.replace(before, after)
    def selected(text):
        return [(n.name, ast.dump(n, include_attributes=False)) for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name in ('sample','sample_goals','object_xyz','create_agent')]
    assert selected(source) == selected(result)
    return result, dict(source_sha256=SOURCE_SHA, edits=edits,
        effective_source_sha256=hashlib.sha256(result.encode()).hexdigest(), sampler_reward_factory_AST_exact=True)


def load_inputs(path, dim):
    source, proof = overlay_source(Path(path).read_bytes(), dim)
    module = types.ModuleType('_foundation_original_inputs')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    original = module.DistanceInputs
    class VisualInputs(original):
        pending = False
        def sample(self, batch_size, idxs=None, evaluation=False, *, prefetch_next=True):
            return super().sample(batch_size, idxs=idxs, evaluation=evaluation)
        def barrier(self):
            pass
    module.VisualInputs, module.width_proof = VisualInputs, proof
    return module
