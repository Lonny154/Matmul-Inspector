import copy
import pytest

torch = pytest.importorskip("torch")

from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_conditional_policy import create_conditional_policy, conditional_policy_loss, conditional_sequence_scores, rank_conditional_actions
from matmul_inspector.tensor_game_learning import PolicyValueConfig, create_policy_value_model
from matmul_inspector.tensor_game_offline_learning import sampled_support_log_probabilities
from matmul_inspector.tensor_game_prefix_coverage import sign_forms


CONFIG=PolicyValueConfig(hidden_dim=16);ACTION=RankOneAction((1,0,0,0),(0,1,0,0),(0,0,1,0))


def setup():
    torch.manual_seed(3);base=create_policy_value_model(CONFIG);model=create_conditional_policy(base,CONFIG);residual=torch.zeros((1,4,4,4));actions=torch.tensor([[ACTION.u+ACTION.v+ACTION.w]]);return base,model,residual,actions


def test_initial_teacher_forced_scores_match_independent_logits():
    base,model,residual,actions=setup();independent,_=base(residual);logp=torch.log_softmax(independent,-1);indices=torch.tensor([2 if x==1 else 1 for x in ACTION.u+ACTION.v+ACTION.w]);expected=logp[0].gather(1,indices[:,None]).sum()
    assert torch.allclose(conditional_sequence_scores(model,residual,actions)[0,0],expected)


def test_initial_sampled_support_loss_matches_independent_loss():
    base,model,residual,actions=setup();second=actions.clone();second[0,0,8:]=torch.tensor((0,0,0,1));support=torch.cat((actions,second),1);mask=torch.tensor([[True,True]]);weights=torch.tensor([[.25,.75]])
    logits,_=base(residual);independent=-(weights*sampled_support_log_probabilities(logits,support,mask,CONFIG.shape)).sum(1).mean();conditional=conditional_policy_loss(model,residual,support,mask,weights)
    assert torch.allclose(independent,conditional,atol=1e-6)


def test_probabilities_normalize_and_gradients_reach_all_heads():
    _,model,residual,actions=setup();logits=model.teacher_forced_logits(residual,actions);assert torch.allclose(torch.softmax(logits,-1).sum(-1),torch.ones((1,1,12)))
    loss=-conditional_sequence_scores(model,residual,actions).mean();loss.backward()
    for name in ("u_head","v_head","w_head","uv_condition","uvw_condition"):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters())


def test_decoder_is_deterministic_legal_and_score_agrees():
    _,model,residual,_=setup();first,audit=rank_conditional_actions(model,residual[0],CONFIG,12);second,_=rank_conditional_actions(model,residual[0],CONFIG,12);assert first==second and len(first)==12 and audit["beam_width"]==384
    action,score=first[0];forms=torch.tensor(sign_forms(action));teacher=float(conditional_sequence_scores(model,residual.expand(len(forms),-1,-1,-1),forms[:,None,:])[:,0].max().detach());assert score==pytest.approx(teacher,abs=2e-6)
    assert all(any(a.u) and any(a.v) and any(a.w) for a,_ in first)


def test_small_overfit_reduces_positive_set_loss():
    _,model,residual,actions=setup();mask=torch.tensor([[True]]);optimizer=torch.optim.Adam(model.parameters(),lr=.01);before=float(conditional_policy_loss(model,residual,actions,mask).detach())
    for _ in range(10):optimizer.zero_grad();loss=conditional_policy_loss(model,residual,actions,mask);loss.backward();optimizer.step()
    assert conditional_policy_loss(model,residual,actions,mask)<before


def test_checkpoint_round_trip():
    _,model,residual,actions=setup();clone=copy.deepcopy(model);clone.load_state_dict(model.state_dict());assert torch.equal(conditional_sequence_scores(model,residual,actions),conditional_sequence_scores(clone,residual,actions))
