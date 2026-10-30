import ast
from pathlib import Path
from accept_existing import select_own_run

SOURCE = '''
def accept(candidates):
    runs = {x for x in candidates}
    if len(runs) != 1:
        raise ValueError('one own native run required')
    return runs.pop()
'''


def run(source):
    module = {"Path": Path}
    exec(compile(source, "<test>", "exec"), module)
    return module["accept"]


original = run(ast.parse(SOURCE))
patched = run(select_own_run(ast.parse(SOURCE), "/own"))
try:
    original(["/parent", "/own"])
    raise AssertionError("original ambiguity was not rejected")
except ValueError:
    pass
assert patched(["/parent", "/own"]) == "/own"
for candidates in (["/parent"], []):
    try:
        patched(candidates)
        raise AssertionError("missing own run was not rejected")
    except ValueError:
        pass
print("run selection: positive and missing-own rejection passed")
