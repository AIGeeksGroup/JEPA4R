"""Aggregated JEPA world model (work item 3).

    encoder -> projector -> latent1 (per frame, as in jepa.JEPA.encode, unchanged)
    aggregator(latent1[t-2], latent1[t-1], latent1[t]) -> latent2        (window = history_size frames)
    predictor(latent2[t], action[t]) -> latent1[t+1]                      (ARPredictor with a single token)

The world model's state for planning is latent2: the rollout (the framework's LeWM rollout, see
``rollout``) keeps a window of latent1 (observed frames, then predicted ones), aggregates it into
latent2 at every step, and the CEM cost is the MSE between the final latent2 and the goal's latent2
(goal = the frames leading up to the goal frame). ``cost_space='latent1'`` compares the last predicted latent1 with the goal frame's latent1.

Training (``agg_forward``): a clip of ``history_size + num_preds`` frames (upstream default 3 + 1 = 4);
window (frames 1-3) -> latent2 -> with action block 3 -> prediction of frame 4's latent1.
Loss = MSE(pred, next latent1) + lambda * SIGReg([latent2, next latent1]) -- the SIGReg call averages over
the stacked positions exactly like upstream averages over its 4 frame positions, so lambda is unchanged.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from jepa import JEPA, detach_clone


class AggJEPA(JEPA):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        aggregator,
        projector=None,
        pred_proj=None,
        obs_key="pixels",
        window=3,
        cost_space="latent2",
    ):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj, obs_key)
        assert cost_space in ("latent2", "latent1")
        self.aggregator = aggregator
        self.window = window
        self.cost_space = cost_space

    # encode(): inherited -- encoder + projector -> info['emb'] = latent1 (B, T, D), act_emb (B, T, D)

    def aggregate(self, emb):
        """latent1 window (B, K, D), K <= window -> latent2 (B, D)"""
        return self.aggregator(emb[:, -self.window:])

    def predict(self, latent2, act_emb):
        """latent2 (B, D) + action embedding (B, A_emb) -> next latent1 (B, D). Single-token ARPredictor."""
        preds = self.predictor(latent2.unsqueeze(1), act_emb.unsqueeze(1))
        return self.pred_proj(preds[:, 0])

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size=None):
        """Verbatim copy of stable_worldmodel.wm.lewm.LeWM.rollout (the framework's rollout for
        ``history_len > 1``: H context frames, the H-1 executed blocks in ``info['action_history']``,
        T strictly-future candidates) with two edits marked ``# 4D``: the observation key, and the
        per-step prediction (frame window -> aggregator -> latent2 -> predictor -> next latent1).
        pixels: (B, S, H, ...) -- H context frames (block timesteps)
        action_sequence: (B, S, T, action_dim) -- strictly-future candidates
        info['action_history']: (B, S, H - 1, action_dim) -- executed action
            blocks between the context frames (required when H > 1)
         - S is the number of action plan samples
         - T is the planning horizon
        Returns ``info`` with ``predicted_emb`` of shape (B, S, H + T, D);
        the first H entries are the encoded context frames.
        """
        if history_size is None:
            history_size = self.window  # 4D: the aggregator's window (the single-token predictor has num_frames 1)

        assert self.obs_key in info, f"{self.obs_key} not in info_dict"  # 4D: obs_key
        H = info[self.obs_key].size(2)
        B, S, T = action_sequence.shape[:3]
        act_past = info.get("action_history")
        if act_past is None:
            act_past = action_sequence.new_zeros(B, S, 0, action_sequence.size(-1))
        assert act_past.size(2) == H - 1, f"action_history must hold H-1={H - 1} executed blocks, got {act_past.size(2)}"
        # action paired with context frame k is the block leaving it; the
        # current frame (k = H-1) pairs with the first candidate
        info["action"] = torch.cat([act_past, action_sequence[:, :, :1]], dim=2)

        # encode initial state, or reuse cached embedding from a prior rollout.
        # detach: to avoid backprop in encoder
        if "emb" not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info["emb"] = _init["emb"].detach().unsqueeze(1).expand(B, S, -1, -1)

        # flatten batch and sample dimensions for rollout
        emb_init = rearrange(info["emb"], "b s ... -> (b s) ...")
        act_past_flat = rearrange(act_past, "b s ... -> (b s) ...")
        act_cand_flat = rearrange(action_sequence, "b s ... -> (b s) ...")
        all_act_emb = self.action_encoder(
            torch.cat([act_past_flat, act_cand_flat], dim=1)
        )  # (BS, H - 1 + T, A_emb); index k = block leaving frame k

        # rollout predictor autoregressively, one step per candidate
        # emb_list holds individual (BS, D) frames, each with its own grad_fn
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))  # H tensors of shape (BS, D)
        for t in range(T):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, D)
            act_trunc = all_act_emb[:, lo : H + t]  # (BS, HS, A_emb)
            emb_list.append(self.predict(self.aggregate(emb_trunc), act_trunc[:, -1]))  # 4D: window -> latent2, newest action -> next latent1

        emb = torch.stack(emb_list, dim=1)  # (BS, H + T, D)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """jepa.JEPA.get_cost with the goal / final state aggregated into latent2 (``cost_space='latent2'``).
        info_dict['goal'] holds the K frames leading up to the goal frame (eval.py goal window)."""
        assert "goal" in info_dict, "goal not in info_dict"
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal[self.obs_key] = goal["goal"]
        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)
        goal.pop("action")
        goal = self.encode(goal)  # emb: (B, K, D) latent1 of the goal window

        if self.cost_space == "latent2":
            info_dict["goal_emb"] = self.aggregate(goal["emb"]).unsqueeze(1)  # (B, 1, D)
        else:
            info_dict["goal_emb"] = goal["emb"]  # criterion uses the last frame
        info_dict = self.rollout(info_dict, action_candidates)
        if self.cost_space == "latent2":  # state after the last prediction = latent2 of the last `window` frames
            pe = info_dict["predicted_emb"]  # (B, S, H + T, D)
            B, S = pe.shape[:2]
            last = self.aggregate(rearrange(pe[:, :, -self.window :], "b s k d -> (b s) k d"))
            info_dict["predicted_emb"] = rearrange(last, "(b s) d -> b s 1 d", b=B, s=S)
        return self.criterion(info_dict)  # inherited: MSE of the last step vs goal


def agg_forward(self, batch, stage, cfg):
    """Training forward for AggJEPA (stable_pretraining Module forward). Mirrors train.lejepa_forward."""
    W = cfg.history_size
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb, act_emb = output["emb"], output["act_emb"]                      # (B, T, D)
    B, T, D = emb.shape
    n = T - W                                                            # number of targets per clip
    assert n >= 1, f"clip of {T} frames too short for a {W}-frame window"

    latent2 = torch.stack([self.model.aggregate(emb[:, i : i + W]) for i in range(n)], dim=1)   # (B, n, D)
    act_last = act_emb[:, W - 1 : W - 1 + n]                                                     # action at each window's newest frame
    pred = self.model.predict(latent2.reshape(B * n, D), act_last.reshape(B * n, -1)).reshape(B, n, D)
    target = emb[:, W:]                                                                          # next-frame latent1 (B, n, D)

    output["pred_loss"] = (pred - target).pow(2).mean()
    # one SIGReg call over 2n positions (n latent2 + n next-latent1), averaged like upstream's 4 positions
    output["sigreg_loss"] = self.sigreg(torch.cat([latent2, target], dim=1).transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output
