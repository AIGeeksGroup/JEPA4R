import copy
import unittest
try:
    from lewm_pcd.bench.gates import aggregation as gates
except ImportError:
    gates = None


def cube_events():
    return [{'task': task, 'episode_id': i, 'success': i < successes, 'checkpoint': '1m'}
            for task, successes in enumerate([1, 2, 0, 3, 4], 1) for i in range(50)]


def bc_rounds():
    result = []
    epochs = list(range(5, 100, 5)) + [99]
    for seed, best, final in [(100, 21, 3), (200, 24, 5), (300, 21, 1)]:
        for epoch in epochs:
            successes = best if epoch == 95 else (final if epoch == 99 else 0)
            result.append({'seed': seed, 'epoch': epoch, 'successes': successes, 'episodes': 25,
                           'mean_return': 10. if epoch == 10 else 1., 'policy_id': f'{seed}-{epoch}'})
    return result


class AggregationTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(gates, "Shared A1 aggregation is not implemented")

    def test_cube_a11_and_event_rejections(self):
        result = gates.cube_summary(cube_events())
        self.assertEqual(result['successes'], 10); self.assertEqual(result['episodes'], 250)
        self.assertEqual(result['success_percent'], 4.)
        self.assertEqual([x['success_percent'] for x in result['tasks']], [2., 4., 0., 6., 8.])
        for bad in [cube_events()[:-1], cube_events()+[cube_events()[0]], [e for e in cube_events() if e['task'] != 5]]:
            with self.assertRaises(ValueError):gates.cube_summary(bad)
        bad = cube_events(); bad[0]['success'] = 'false'
        with self.assertRaises(ValueError):gates.cube_summary(bad)
        bad = cube_events(); bad[0]['checkpoint'] = 'other'
        with self.assertRaises(ValueError):gates.cube_summary(bad)

    def test_bc_a12_best_final_ddof_and_saved_policy(self):
        result = gates.bc_summary(bc_rounds())
        self.assertEqual(result['best_mean_percent'], 88.)
        self.assertAlmostEqual(result['best_sample_std_pp'], 6.928203230275509)
        self.assertEqual(result['final_mean_percent'], 12.)
        self.assertEqual([r['best_percent'] for r in result['seeds']], [84., 96., 84.])
        events = [{'seed': 100, 'epoch': 10, 'successes': 10, 'episodes': 25, 'mean_return': 100., 'policy_id': 'return-best'},
                  {'seed': 100, 'epoch': 95, 'successes': 24, 'episodes': 25, 'mean_return': 90., 'policy_id': 'success-best'}]
        saved = gates.select_saved_policy(events)
        self.assertEqual(saved['epoch'], 10); self.assertEqual(saved['policy_id'], 'return-best')
        self.assertEqual(saved['success_percent'], 40.)

    def test_immediate_return_save_events_and_ties(self):
        self.assertTrue(hasattr(gates, 'audit_saved_policies'), 'Immediate-save event audit is not implemented')
        rounds = [{'seed': 100, 'epoch': e, 'successes': n, 'episodes': 25, 'mean_return': reward, 'policy_id': policy}
                  for e, n, reward, policy in [(5, 0, 80., 'p5'), (10, 10, 100., 'p10'), (95, 24, 90., 'p95'), (99, 3, 100., 'p99')]]
        saves = [{'epoch': 5, 'policy_id': 'p5'}, {'epoch': 10, 'policy_id': 'p10'}, {'epoch': 99, 'policy_id': 'p99'}]
        self.assertTrue(gates.audit_saved_policies(rounds, saves)['passed'])
        self.assertFalse(gates.audit_saved_policies(rounds, [{'epoch': 95, 'policy_id': 'p95'}])['passed'])
        self.assertFalse(gates.audit_saved_policies(rounds, saves[:-1])['passed'])
        self.assertEqual(gates.select_saved_policy(rounds)['policy_id'], 'p99')
        self.assertEqual(gates.select_saved_policy(list(reversed(rounds)))['epoch'], 99)
        self.assertFalse(gates.audit_saved_policies(rounds, [{'epoch': 5, 'policy_id': 'p5'}, {'epoch': 10, 'policy_id': 'wrong'}])['passed'])

    def test_explicit_semantic_groups_cannot_be_mixed_or_incomplete(self):
        for field, first, other in [('environment', 'native', 'rh56'), ('method', 'jepa4r', 'lewm'), ('split', 'held_out', 'train_init')]:
            rounds = bc_rounds()
            for event in rounds: event[field] = first
            self.assertEqual(gates.bc_summary(rounds)['best_mean_percent'], 88.)
            rounds[0][field] = other
            with self.assertRaises(ValueError): gates.bc_summary(rounds)
            del rounds[0][field]
            with self.assertRaises(ValueError): gates.bc_summary(rounds)
        events = [{'fragment_id': i, 'source_episode_id': i, 'initial_distance': .2, 'final_distance': .05} for i in range(2)]
        for field, first, other in [('route', 'pcd', 'pix'), ('method', 'jepa4r', 'lewm'), ('environment', 'native', 'rh56'), ('split', 'held_out', 'train_init')]:
            supplied = [dict(event, **{field: first}) for event in events]
            self.assertEqual(gates.cem_summary(supplied)['all']['success_percent'], 100.)
            supplied[0][field] = other
            with self.assertRaises(ValueError): gates.cem_summary(supplied)
            del supplied[0][field]
            with self.assertRaises(ValueError): gates.cem_summary(supplied)

    def test_real_rh56_episode_role_fields_cannot_mix(self):
        for field in ('role', 'evaluation_role'):
            rounds = bc_rounds()
            for event in rounds: event[field] = 'heldout_eval'
            self.assertEqual(gates.bc_summary(rounds)['best_mean_percent'], 88.)
            rounds[0][field] = 'train_initial'
            with self.assertRaises(ValueError): gates.bc_summary(rounds)
            del rounds[0][field]
            with self.assertRaises(ValueError): gates.bc_summary(rounds)
            cem = [{'fragment_id': i, 'source_episode_id': i, 'initial_distance': .2,
                    'final_distance': .05, field: 'heldout_eval'} for i in range(2)]
            cem[0][field] = 'train_initial'
            with self.assertRaises(ValueError): gates.cem_summary(cem)

    def test_bc_reject_missing_round_duplicate_seed_and_denominator(self):
        for bad in [bc_rounds()[:-1], bc_rounds()+[bc_rounds()[0]], [r for r in bc_rounds() if r['seed'] != 300]]:
            with self.assertRaises(ValueError):gates.bc_summary(bad)
        bad = bc_rounds(); bad[0]['episodes'] = 24
        with self.assertRaises(ValueError):gates.bc_summary(bad)
        bad = bc_rounds(); bad[0]['successes'] = 26
        with self.assertRaises(ValueError):gates.bc_summary(bad)
        bad = bc_rounds(); bad[0]['mean_return'] = float('nan')
        with self.assertRaises(ValueError):gates.select_saved_policy(bad)

    def test_a13_cumulative_and_strict_distance_boundaries(self):
        for count in [24, 25]:
            self.assertFalse(gates.bc_success(count)); self.assertFalse(gates.bc_success(count, environment='rh56'))
        self.assertTrue(gates.bc_success(26)); self.assertTrue(gates.bc_success(sum([True, False]*26)))
        self.assertTrue(gates.bc_success(26, environment='rh56', overload=False))
        self.assertFalse(gates.bc_success(26, environment='rh56', overload=True))
        self.assertTrue(gates.cem_success(.0999)); self.assertFalse(gates.cem_success(.1000))
        self.assertFalse(gates.cem_success(.01, native_failure=True))
        with self.assertRaises(ValueError):gates.cem_success(float('nan'))
        with self.assertRaises(ValueError):gates.cem_success(.2, native_failure='false')
        with self.assertRaises(ValueError):gates.bc_success(26, environment='unknown')

    def test_a14_cem_both_denominators_and_identity(self):
        events = [{'fragment_id': i, 'source_episode_id': f'episode-{i//2}', 'initial_distance': .09 if i < 12 else .2,
                   'final_distance': .05 if i < 15 else .2, 'native_failure': False} for i in range(20)]
        result = gates.cem_summary(events, expected_count=20)
        self.assertEqual(result['all']['success_percent'], 75.)
        self.assertEqual(result['initially_outside']['success_percent'], 37.5)
        self.assertEqual(result['initially_outside']['fragments'], 8)
        with self.assertRaises(ValueError):gates.cem_summary(events+[events[0]])
        with self.assertRaises(ValueError):gates.cem_summary(events[:-1], expected_count=20)
        result = gates.cem_summary(events[:12]); self.assertIsNone(result['initially_outside']['success_percent'])
        bad = copy.deepcopy(events); bad[0]['success'] = False
        with self.assertRaises(ValueError):gates.cem_summary(bad)

if __name__ == '__main__': unittest.main()
