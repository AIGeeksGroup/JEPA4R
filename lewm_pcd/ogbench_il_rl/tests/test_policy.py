import hashlib
import math
from pathlib import Path
import sys
import unittest
import tempfile
import json

import torch
from torch import nn

from ogbench_il_rl.models.policy import create_policy, unit_gaussian_nll, make_optimizer


class Trunk(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Flatten(),nn.Linear(12,192),nn.BatchNorm1d(192))
        self.calls=0
    def forward(self,x):
        self.calls+=1; return super().forward(x)


def world():
    model=nn.Module(); model.encoder=Trunk(); model.projector=nn.Sequential(nn.Linear(192,192),nn.BatchNorm1d(192))
    return model.eval()


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class PolicyTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(4); torch.manual_seed(21)

    def test_nll_raw_units_value_and_gradient(self):
        means=torch.tensor([[.1,.2,.3,.4,.5],[-.5,-.4,-.3,-.2,-.1]],dtype=torch.float64,requires_grad=True)
        target=torch.zeros_like(means)
        loss=unit_gaussian_nll(means,target)
        reference=-torch.distributions.Independent(torch.distributions.Normal(means,torch.ones_like(means)),1).log_prob(target).mean()
        torch.testing.assert_close(loss,reference,rtol=0,atol=1e-12)
        torch.testing.assert_close(loss,2.5*nn.functional.mse_loss(means,target)+2.5*math.log(2*math.pi),rtol=0,atol=1e-12)
        loss.backward(); torch.testing.assert_close(means.grad,(means.detach()-target)/2,rtol=0,atol=0)
        with self.assertRaises(ValueError): unit_gaussian_nll(means,torch.ones_like(means)*2)

    def test_original_trunk_numerics_bn_frozen_real_forwards_and_head_update(self):
        source=world(); current=torch.randn(8,3,2,2); goal=torch.randn(8,3,2,2)
        with torch.no_grad(): expected=source.projector(source.encoder(current))
        policy=create_policy(world_model=source,seed=0)
        torch.testing.assert_close(policy.encoder(current).logits,expected,rtol=0,atol=0)
        frozen=state_hash(policy.encoder); head=state_hash(policy.action_head); calls=source.encoder.calls
        policy.train(); self.assertFalse(policy.encoder.training)
        self.assertFalse(any(m.training for m in policy.encoder.modules() if isinstance(m,nn.BatchNorm1d)))
        optimizer=make_optimizer(policy)
        self.assertIsInstance(optimizer,torch.optim.Adam)
        self.assertEqual(optimizer.defaults['lr'],3e-4); self.assertEqual(optimizer.defaults['betas'],(.9,.999))
        self.assertEqual(optimizer.defaults['eps'],1e-8); self.assertEqual(optimizer.defaults['weight_decay'],0)
        unit_gaussian_nll(policy(current,goal),torch.zeros(8,5)).backward(); optimizer.step()
        self.assertEqual(source.encoder.calls,calls+2)
        self.assertEqual(state_hash(policy.encoder),frozen); self.assertNotEqual(state_hash(policy.action_head),head)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in policy.encoder.parameters()))
        self.assertEqual({id(p) for p in policy.parameters() if p.requires_grad},{id(p) for p in policy.action_head.parameters()})

    def test_strict_factory_reload_and_dimensions_and_unclipped_mean(self):
        policy=create_policy(world_model=world(),seed=0); other=create_policy(world_model=world(),seed=99)
        other.load_state_dict(policy.state_dict(),strict=True)
        current=torch.randn(4,3,2,2); goal=torch.randn_like(current)
        torch.testing.assert_close(policy(current,goal),other(current,goal),rtol=0,atol=0)
        for kwargs in [dict(history_size=3),dict(action_dim=25),dict(latent_dim=384)]:
            with self.assertRaises(ValueError): create_policy(world_model=world(),**kwargs)
        with self.assertRaises(ValueError): policy(current[:,None],goal[:,None])
        with self.assertRaises(ValueError): policy(current,goal[:2])
        with torch.no_grad(): policy.action_head[-1].weight.zero_(); policy.action_head[-1].bias.fill_(2)
        self.assertTrue(torch.equal(policy.get_action({'pixels':current,'goal':goal}),torch.full((4,5),2.)))
        self.assertTrue(all(not isinstance(layer,nn.LayerNorm) for layer in policy.action_head.modules()))
        for index in [1,3,5]: self.assertEqual(policy.action_head[index].approximate,'tanh')
        self.assertTrue(torch.equal(create_policy(world_model=world(),seed=0).action_head[0].weight,create_policy(world_model=world(),seed=0).action_head[0].weight))

    def test_effective1024_and_tail576_micro_gradient_and_adam_step(self):
        for size in [1024,576]:
            whole=create_policy(world_model=world(),seed=0); micro=create_policy(world_model=world(),seed=0)
            micro.load_state_dict(whole.state_dict(),strict=True)
            current=torch.randn(size,3,2,2); goal=torch.randn_like(current); target=torch.rand(size,5)*2-1
            one_opt=make_optimizer(whole); micro_opt=make_optimizer(micro)
            loss=unit_gaussian_nll(whole(current,goal),target); loss.backward()
            total=0.
            for start in range(0,size,64):
                stop=min(size,start+64); value=unit_gaussian_nll(micro(current[start:stop],goal[start:stop]),target[start:stop])
                weight=(stop-start)/size; (value*weight).backward(); total+=float(value.detach())*weight
            self.assertAlmostEqual(float(loss.detach()),total,places=5)
            for x,y in zip(whole.action_head.parameters(),micro.action_head.parameters()): torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=1e-6)
            one_opt.step(); micro_opt.step()
            for x,y in zip(whole.action_head.parameters(),micro.action_head.parameters()): torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)

    def test_generic_hydra_loader_factory_strict_full_policy_reload(self):
        from stable_worldmodel.wm.utils import load_pretrained
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; source.mkdir(); exported=root/'export'; exported.mkdir()
            original_world=world(); torch.save(original_world.state_dict(),source/'weights.pt')
            (source/'config.json').write_text(json.dumps({'_target_':'ogbench_il_rl.tests.test_policy.world'}))
            original=create_policy(source_checkpoint=str(source/'weights.pt'),seed=0)
            torch.save(original.state_dict(),exported/'weights.pt')
            (exported/'config.json').write_text(json.dumps({'_target_':'ogbench_il_rl.models.policy.create_policy',
                 'source_checkpoint':str(source/'weights.pt'),'modality':'pixels','history_size':1,'latent_dim':192,'action_dim':5,'seed':0}))
            loaded=load_pretrained(str(exported/'weights.pt'),cache_dir=str(root))
            current=torch.randn(4,3,2,2); goal=torch.randn_like(current)
            self.assertEqual(state_hash(original),state_hash(loaded))
            torch.testing.assert_close(original(current,goal),loaded(current,goal),rtol=0,atol=0)

    def test_pinned_flax_head_and_optax_adam_copied_weight_parity(self):
        from ogbench_il_rl.evaluation.official_state_sr import configure_cpu
        configure_cpu()
        import os
        os.environ['JAX_PLATFORMS']='cpu'
        import jax,jax.numpy as jnp,optax,numpy as np
        from utils.networks import GCActor
        policy=create_policy(world_model=world(),seed=0)
        inputs=torch.randn(8,384); target=torch.rand(8,5)*2-1
        model=GCActor(hidden_dims=(512,512,512),action_dim=5,const_std=True)
        params=model.init(jax.random.PRNGKey(42),jnp.asarray(inputs[:,:192].numpy()),jnp.asarray(inputs[:,192:].numpy()))['params']
        for i,index in enumerate([0,2,4]):
            params['actor_net'][f'Dense_{i}']['kernel']=jnp.asarray(policy.action_head[index].weight.detach().numpy().T)
            params['actor_net'][f'Dense_{i}']['bias']=jnp.asarray(policy.action_head[index].bias.detach().numpy())
        params['mean_net']['kernel']=jnp.asarray(policy.action_head[6].weight.detach().numpy().T)
        params['mean_net']['bias']=jnp.asarray(policy.action_head[6].bias.detach().numpy())
        x,g=jnp.asarray(inputs[:,:192].numpy()),jnp.asarray(inputs[:,192:].numpy())
        def objective(p): return -model.apply({'params':p},x,g).log_prob(jnp.asarray(target.numpy())).mean()
        means=policy.action_head(inputs)
        torch.testing.assert_close(means.detach(),torch.from_numpy(np.array(model.apply({'params':params},x,g).mode())),atol=1e-6,rtol=1e-5)
        loss=unit_gaussian_nll(means,target); loss.backward()
        self.assertAlmostEqual(float(loss.detach()),float(objective(params)),places=5)
        grads=jax.grad(objective)(params)
        tx=optax.adam(3e-4); updates,_=tx.update(grads,tx.init(params),params); next_params=optax.apply_updates(params,updates)
        # Framework GEMM/autodiff rounding may amplify near-zero gradients in
        # Adam. First check gradients, then isolate optimizer parameter parity
        # by supplying exactly the same computed gradient to both optimizers.
        for i,index in enumerate([0,2,4]):
            kernel=torch.from_numpy(np.array(grads['actor_net'][f'Dense_{i}']['kernel']).T)
            bias=torch.from_numpy(np.array(grads['actor_net'][f'Dense_{i}']['bias']))
            torch.testing.assert_close(policy.action_head[index].weight.grad,kernel,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(policy.action_head[index].bias.grad,bias,atol=1e-6,rtol=1e-5)
            policy.action_head[index].weight.grad=kernel.clone(); policy.action_head[index].bias.grad=bias.clone()
        kernel=torch.from_numpy(np.array(grads['mean_net']['kernel']).T); bias=torch.from_numpy(np.array(grads['mean_net']['bias']))
        torch.testing.assert_close(policy.action_head[6].weight.grad,kernel,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(policy.action_head[6].bias.grad,bias,atol=1e-6,rtol=1e-5)
        policy.action_head[6].weight.grad=kernel.clone(); policy.action_head[6].bias.grad=bias.clone()
        make_optimizer(policy).step()
        for i,index in enumerate([0,2,4]):
            torch.testing.assert_close(policy.action_head[index].weight,torch.from_numpy(np.array(next_params['actor_net'][f'Dense_{i}']['kernel']).T),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(policy.action_head[6].weight,torch.from_numpy(np.array(next_params['mean_net']['kernel']).T),atol=1e-6,rtol=1e-5)


if __name__=='__main__': unittest.main()
