"""Experimental conditional u -> v -> w TensorGame policy and decoder."""

from __future__ import annotations

import copy
import math
import time

from .tensor_game import RankOneAction, validate_action
from .tensor_game_learning import COEFFICIENTS
from .tensor_game_prefix_coverage import sign_forms
from .tensor_game_search import canonicalize_action


def _one_hot(values, *, device, dtype):
    import torch

    indices = torch.tensor([COEFFICIENTS.index(int(x)) for x in values], device=device)
    return torch.nn.functional.one_hot(indices, 3).to(dtype).flatten()


def create_conditional_policy(independent_model, config):
    """Create an initially equivalent policy with zero conditional adapters."""
    import torch

    class ConditionalPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = copy.deepcopy(independent_model.features)
            self.value_head = copy.deepcopy(independent_model.value_head)
            self.u_head = torch.nn.Linear(config.hidden_dim, 12)
            self.v_head = torch.nn.Linear(config.hidden_dim, 12)
            self.w_head = torch.nn.Linear(config.hidden_dim, 12)
            self.uv_condition = torch.nn.Linear(12, 12)
            self.uvw_condition = torch.nn.Linear(24, 12)
            with torch.no_grad():
                weight = independent_model.policy_head.weight.reshape(12, 3, config.hidden_dim)
                bias = independent_model.policy_head.bias.reshape(12, 3)
                for head, start in ((self.u_head, 0), (self.v_head, 4), (self.w_head, 8)):
                    head.weight.copy_(weight[start : start + 4].reshape(12, config.hidden_dim))
                    head.bias.copy_(bias[start : start + 4].reshape(12))
                self.uv_condition.weight.zero_(); self.uv_condition.bias.zero_()
                self.uvw_condition.weight.zero_(); self.uvw_condition.bias.zero_()

        def encode(self, residual):
            flat = residual.reshape(residual.shape[0], -1).to(torch.float32)
            return self.features(flat / config.input_scale)

        def component_logits(self, features, u_values=None, v_values=None):
            u = self.u_head(features).reshape(-1, 4, 3)
            if u_values is None:
                return u, None, None
            u_context = torch.stack([_one_hot(row, device=features.device, dtype=features.dtype) for row in u_values])
            v = (self.v_head(features) + self.uv_condition(u_context)).reshape(-1, 4, 3)
            if v_values is None:
                return u, v, None
            v_context = torch.stack([_one_hot(row, device=features.device, dtype=features.dtype) for row in v_values])
            w = (self.w_head(features) + self.uvw_condition(torch.cat((u_context, v_context), 1))).reshape(-1, 4, 3)
            return u, v, w

        def teacher_forced_logits(self, residual, actions):
            if actions.ndim != 3 or actions.shape[2] != 12:
                raise ValueError("actions must have shape [batch, actions, 12]")
            features = self.encode(residual); batch, count, _ = actions.shape
            expanded = features[:, None, :].expand(batch, count, features.shape[1]).reshape(batch * count, -1)
            flat = actions.reshape(batch * count, 12)
            u, v, w = self.component_logits(expanded, flat[:, :4], flat[:, 4:8])
            return torch.cat((u, v, w), 1).reshape(batch, count, 12, 3)

        def forward(self, residual):
            features = self.encode(residual)
            return self.u_head(features).reshape(-1, 4, 3), self.value_head(features).squeeze(-1)

    return ConditionalPolicy()


def conditional_sequence_scores(model, residual, actions):
    """Teacher-forced log probability for explicit 12-token sequences."""
    import torch

    logits = model.teacher_forced_logits(residual, actions)
    logp = torch.log_softmax(logits, -1)
    classes = torch.zeros_like(actions)
    for index, coefficient in enumerate(COEFFICIENTS):
        classes = torch.where(actions == coefficient, index, classes)
    if torch.any((actions < -1) | (actions > 1)):
        raise ValueError("coefficient outside {-1,0,1}")
    return logp.gather(3, classes[..., None]).squeeze(3).sum(2)


def conditional_canonical_scores(model, residual, actions, mask):
    """Log-sum sign-equivalent sequence probability for canonical actions."""
    import torch

    rows = []
    for batch_index in range(actions.shape[0]):
        forms = []
        for action_values in actions[batch_index]:
            values = tuple(int(x) for x in action_values)
            action = RankOneAction(values[:4], values[4:8], values[8:])
            u, v, w = action.u, action.v, action.w
            forms.extend((u+v+w, tuple(-x for x in u)+tuple(-x for x in v)+w, tuple(-x for x in u)+v+tuple(-x for x in w), u+tuple(-x for x in v)+tuple(-x for x in w)))
        rows.append(torch.tensor(forms, device=actions.device))
    expanded = torch.stack(rows)
    scores = conditional_sequence_scores(model, residual, expanded)
    scores = torch.logsumexp(scores.reshape(actions.shape[0], actions.shape[1], 4), 2)
    return scores.masked_fill(~mask, -torch.inf)


def conditional_policy_loss(model, residual, actions, mask, weights=None):
    """Positive-set NLL or weighted sampled-support cross-entropy."""
    import torch

    if torch.any(mask.sum(1) == 0):
        raise ValueError("every state requires at least one action")
    scores = conditional_canonical_scores(model, residual, actions, mask)
    if weights is None:
        return torch.stack([-torch.logsumexp(row[row_mask], 0) for row, row_mask in zip(scores, mask)]).mean()
    if weights.shape != mask.shape:
        raise ValueError("weights do not match action mask")
    normalized = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    return -(weights * torch.where(mask, normalized, 0)).sum(1).mean()


def rank_conditional_actions(model, residual, config, max_actions=12, *, include_beam=False):
    """Separate deterministic conditional beam decoder with audit counters."""
    import torch

    if max_actions <= 0:
        raise ValueError("max_actions must be positive")
    started = time.perf_counter(); width = max(256, max_actions * 32)
    with torch.no_grad(): features = model.encode(residual.reshape(1, *residual.shape))
    beam = [((), 0.0)]; cache = {}; expansions = 0; evaluations = 1
    for position in range(12):
        candidates = []
        for prefix, score in beam:
            if position < 4:
                key = ("u",)
                if key not in cache:
                    cache[key] = torch.log_softmax(model.u_head(features).reshape(4, 3), -1).detach().cpu(); evaluations += 1
                row = cache[key][position]
            elif position < 8:
                key = ("v", prefix[:4])
                if key not in cache:
                    u = torch.tensor([prefix[:4]], device=features.device);cache[key] = torch.log_softmax(model.component_logits(features, u)[1][0], -1).detach().cpu();evaluations += 1
                row = cache[key][position - 4]
            else:
                key = ("w", prefix[:8])
                if key not in cache:
                    u = torch.tensor([prefix[:4]], device=features.device);v = torch.tensor([prefix[4:8]], device=features.device);cache[key] = torch.log_softmax(model.component_logits(features, u, v)[2][0], -1).detach().cpu();evaluations += 1
                row = cache[key][position - 8]
            for class_index, coefficient in enumerate(COEFFICIENTS): candidates.append((prefix + (coefficient,), score + float(row[class_index])))
        expansions += len(candidates);candidates.sort(key=lambda x:(-x[1],x[0]));beam=candidates[:width]
    accepted={};invalid=duplicates=decoded=0
    for values,score in beam:
        decoded+=1;action=RankOneAction(values[:4],values[4:8],values[8:])
        try:validate_action(action,config.shape,frozenset(COEFFICIENTS))
        except ValueError:invalid+=1;continue
        canonical=canonicalize_action(action)
        if canonical in accepted:duplicates+=1
        accepted[canonical]=max(score,accepted.get(canonical,-math.inf))
        if len(accepted)>=max_actions:break
    ranked=tuple(sorted(accepted.items(),key=lambda x:(-x[1],x[0].u,x[0].v,x[0].w)))
    audit={"requested":max_actions,"beam_width":width,"beam_expansions":expansions,"conditional_forward_evaluations":evaluations,"decoded_candidates":decoded,"invalid_rejected":invalid,"canonical_duplicates_rejected":duplicates,"returned":len(ranked),"generation_seconds":time.perf_counter()-started}
    if include_beam:audit["final_beam"]=beam
    return ranked,audit
