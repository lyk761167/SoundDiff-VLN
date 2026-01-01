import os
import torch
import torch.nn as nn
import torch.optim as optim

EPS = 1e-5

class BC(nn.Module):
    """
    Behavior Cloning (supervised imitation learning) for discrete actions.

    Requirements:
    - rollouts provides expert_actions (LongTensor) aligned with obs/actions batch
    - actor_critic.evaluate_actions(...) returns action_log_probs for given actions
    """

    def __init__(
        self,
        actor_critic,
        lr=None,
        eps=None,
        max_grad_norm=None,
        entropy_coef=0.0,     # can be small >0 to encourage exploration a bit
        diff_coef=1e-3,       # your diffusion loss weight
        use_diff_ramp=True,
        diff_warmup=2000,
        diff_ramp=2000,
    ):
        super().__init__()
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef

        self.diff_coef = diff_coef
        self.use_diff_ramp = use_diff_ramp
        self.diff_warmup = diff_warmup
        self.diff_ramp = diff_ramp

        self._step = 0

    def _diff_weight(self):
        if not self.use_diff_ramp:
            return self.diff_coef
        t = self._step
        if t < self.diff_warmup:
            return 0.0
        w = min(1.0, (t - self.diff_warmup) / max(1, self.diff_ramp))
        return w * self.diff_coef

    def update(self, rollouts):
        """
        Supervised update:
        action_loss = -log pi(a_expert | s)
        """
        action_loss_epoch = 0.0
        entropy_epoch = 0.0
        diff_loss_epoch = 0.0

        # ✅ 你原来 PPO 用的是 recurrent_generator；BC 也可复用
        data_generator = rollouts.recurrent_generator(
            advantages=None,  # BC 不需要优势
            num_mini_batch=rollouts.num_mini_batch if hasattr(rollouts, "num_mini_batch") else 1
        )

        num_updates = 0
        for sample in data_generator:
            (
                obs_batch,
                recurrent_hidden_states_batch,
                actions_batch,              # 这里我们不用它（它是执行动作，DAgger时可能不是专家）
                prev_actions_batch,
                value_preds_batch,          # 不用
                return_batch,               # 不用
                masks_batch,
                old_action_log_probs_batch, # 不用
                adv_targ,                   # 不用
                *rest
            ) = sample

            # ✅ 从 sample 或 rollouts 里拿 expert action
            # 你需要在 rollouts 里存 expert_actions，并在 generator 里吐出来
            # 这里假设 generator 额外返回了 expert_actions_batch
            if len(rest) == 0:
                raise RuntimeError("BC.update needs expert_actions_batch in rollout sample.")
            expert_actions_batch = rest[0]  # LongTensor [T*B, 1] or [T*B]

            # forward: evaluate log prob of EXPERT action
            values, expert_action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                obs_batch,
                recurrent_hidden_states_batch,
                prev_actions_batch,
                masks_batch,
                expert_actions_batch,
            )

            # ✅ BC loss
            action_loss = -expert_action_log_probs.mean()

            # optional entropy bonus (encourage exploration a bit)
            loss = action_loss - self.entropy_coef * dist_entropy

            # ✅ diffusion aux loss（你已验证 mean + small weight 才稳定）
            diff_loss = getattr(self.actor_critic.net, "_last_diff_loss", None)
            if diff_loss is not None:
                diff_loss = diff_loss.mean()
                w = self._diff_weight()
                loss = loss + (w * diff_loss)
                diff_loss_epoch += float(diff_loss.detach().item())

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            action_loss_epoch += float(action_loss.detach().item())
            entropy_epoch += float(dist_entropy.detach().item())
            num_updates += 1
            self._step += 1

        if num_updates == 0:
            return 0.0, 0.0, 0.0

        return action_loss_epoch / num_updates, entropy_epoch / num_updates, diff_loss_epoch / num_updates
