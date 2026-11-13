# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/gates/aggregation.py (verbatim)
"""A1 raw-event aggregation. No training, rollout, or paper-target assumptions."""
from numbers import Integral, Real
import math
import numpy as np

BC_EPOCHS = tuple(range(5, 100, 5)) + (99,)
# Optional raw-log labels: if supplied, one summary must not mix these protocols.
GROUP_FIELDS = ('environment', 'env', 'method', 'model', 'split', 'eval_split',
                'evaluation_split', 'initialization', 'route', 'role', 'evaluation_role')


def _integer(value, name, minimum=0):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return int(value)


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)): raise ValueError(name + ' must be boolean')
    return bool(value)


def _finite(value, name, nonnegative=False):
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(name + ' must be finite' + (' and nonnegative' if nonnegative else ''))
    return float(value)


def _identity(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)) or value == '':
        raise ValueError(name + ' must be a nonempty string or integer')
    return value


def _same_group(events, fields):
    for field in fields:
        present = [field in event for event in events]
        if any(present) and (not all(present) or any(event[field] != events[0][field] for event in events)):
            raise ValueError('mixed or incomplete group identity: ' + field)


def cube_summary(events, *, tasks=(1, 2, 3, 4, 5), episodes_per_task=50):
    events = list(events)
    budget = _integer(episodes_per_task, 'episodes_per_task', 1)
    if not tasks or len(set(tasks)) != len(tasks): raise ValueError('expected tasks must be unique')
    _same_group(events, GROUP_FIELDS + ('checkpoint', 'update', 'training_seed'))
    counts = {task: [0, 0] for task in tasks}; seen = set()
    for event in events:
        task = _integer(event['task'], 'task', 1)
        episode = _identity(event['episode_id'], 'episode_id')
        if task not in counts or (task, episode) in seen: raise ValueError('unexpected task or duplicate episode')
        seen.add((task, episode)); counts[task][0] += _boolean(event['success'], 'success'); counts[task][1] += 1
    if any(n != budget for _, n in counts.values()): raise ValueError('missing task or incorrect episode denominator')
    rows = [dict(task=t, successes=s, episodes=n, success_percent=100. * s / n) for t, (s, n) in counts.items()]
    successes, episodes = sum(x['successes'] for x in rows), sum(x['episodes'] for x in rows)
    return dict(tasks=rows, successes=successes, episodes=episodes, success_percent=100. * successes / episodes)


def _round(event):
    successes, episodes = _integer(event['successes'], 'successes'), _integer(event['episodes'], 'episodes', 1)
    if successes > episodes: raise ValueError('successes exceed denominator')
    return successes, episodes


def bc_summary(rounds, *, seeds=(100, 200, 300), epochs=BC_EPOCHS, episodes_per_round=25):
    rounds = list(rounds)
    _same_group(rounds, GROUP_FIELDS)
    if not seeds or len(set(seeds)) != len(seeds) or not epochs or len(set(epochs)) != len(epochs):
        raise ValueError('expected seeds and epochs must be nonempty and unique')
    budget = _integer(episodes_per_round, 'episodes_per_round', 1)
    rows = {seed: {} for seed in seeds}
    for event in rounds:
        seed, epoch = _integer(event['seed'], 'seed'), _integer(event['epoch'], 'epoch')
        if seed not in rows or epoch not in epochs or epoch in rows[seed]: raise ValueError('unexpected or duplicate seed/epoch')
        successes, episodes = _round(event)
        if episodes != budget: raise ValueError('incorrect BC denominator')
        rows[seed][epoch] = (successes, episodes)
    if any(set(row) != set(epochs) for row in rows.values()): raise ValueError('missing BC seed or evaluation round')
    result = []
    for seed in seeds:
        ordered = [dict(epoch=e, successes=rows[seed][e][0], episodes=rows[seed][e][1],
                        success_percent=100. * rows[seed][e][0] / rows[seed][e][1]) for e in epochs]
        best = max(ordered, key=lambda r: r['success_percent'])
        result.append(dict(seed=seed, best_percent=best['success_percent'], best_epoch=best['epoch'],
                           final_percent=ordered[-1]['success_percent'], final_epoch=ordered[-1]['epoch'], rounds=ordered))
    best_values, final_values = [r['best_percent'] for r in result], [r['final_percent'] for r in result]
    return dict(seeds=result, best_mean_percent=float(np.mean(best_values)),
                best_sample_std_pp=float(np.std(best_values, ddof=1)) if len(seeds) > 1 else None,
                final_mean_percent=float(np.mean(final_values)))


def select_saved_policy(rounds):
    """Return-based identity; equal highest returns retain the latest evaluated epoch."""
    rounds = list(rounds)
    if not rounds: raise ValueError('no policy evaluations')
    _same_group(rounds, GROUP_FIELDS + ('seed',))
    seen = set()
    for event in rounds:
        epoch = _integer(event['epoch'], 'epoch')
        if epoch in seen: raise ValueError('duplicate policy evaluation epoch')
        seen.add(epoch)
        _round(event); _finite(event['mean_return'], 'mean_return'); _identity(event['policy_id'], 'policy_id')
    best = max(rounds, key=lambda event: (event['mean_return'], event['epoch']))
    successes, episodes = _round(best)
    return dict(epoch=best['epoch'], policy_id=best['policy_id'], mean_return=best['mean_return'],
                success_percent=100. * successes / episodes)


def audit_saved_policies(rounds, save_events):
    """Check immediate highest-return (including equal-return) best-policy writes.

    Events describe best_policy writes, not periodic checkpoint snapshots.
    The frozen EAI patch uses >=, so a later tie must also be saved.
    The caller must bind these observed events to actual saved files/reloads;
    this function cannot establish that a filesystem write happened.
    """
    rounds = list(rounds)
    selected = select_saved_policy(rounds)
    expected, best_return, last_epoch = [], -math.inf, -1
    for event in rounds:
        if event['epoch'] <= last_epoch: raise ValueError('evaluations must be chronological')
        last_epoch = event['epoch']
        if event['mean_return'] >= best_return:
            best_return = event['mean_return']
            expected.append(dict(epoch=event['epoch'], policy_id=event['policy_id']))
    observed = [dict(epoch=_integer(event['epoch'], 'save epoch'),
                     policy_id=_identity(event['policy_id'], 'saved policy_id')) for event in save_events]
    return dict(passed=observed == expected, expected_saves=expected,
                observed_saves=observed, selected_policy=selected)


def bc_success(goal_steps, *, environment='native', overload=False):
    count = _integer(goal_steps, 'cumulative goal_steps')
    overload = _boolean(overload, 'overload')
    if environment not in ('native', 'rh56'): raise ValueError('unknown BC environment')
    return count > 25 and (environment != 'rh56' or not overload)


def cem_success(distance, *, native_failure=False):
    value = _finite(distance, 'distance', True)
    failed = _boolean(native_failure, 'native_failure')
    return value < 0.1 and not failed


def cem_summary(events, *, expected_count=None):
    events = list(events)
    if expected_count is not None and len(events) != _integer(expected_count, 'expected_count', 1):
        raise ValueError('incorrect fragment denominator')
    if not events: raise ValueError('no CEM fragments')
    _same_group(events, GROUP_FIELDS)
    seen, accepted, outside = set(), [], []
    for event in events:
        identity = _identity(event['fragment_id'], 'fragment_id')
        if identity in seen: raise ValueError('duplicate fragment')
        seen.add(identity)
        _identity(event['source_episode_id'], 'source_episode_id')
        initial = _finite(event['initial_distance'], 'initial_distance', True)
        success = cem_success(event['final_distance'], native_failure=event.get('native_failure', False))
        if 'success' in event and _boolean(event['success'], 'success') != success:
            raise ValueError('recorded fragment success differs from events')
        row = dict(fragment_id=identity, source_episode_id=event['source_episode_id'], success=success)
        accepted.append(row)
        if initial >= 0.1: outside.append(row)
    def summarize(rows):
        successes = sum(row['success'] for row in rows)
        return dict(successes=successes, fragments=len(rows),
                    success_percent=100. * successes / len(rows) if rows else None,
                    fragment_ids=[row['fragment_id'] for row in rows],
                    source_episode_ids=[row['source_episode_id'] for row in rows])
    return dict(all=summarize(accepted), initially_outside=summarize(outside))
