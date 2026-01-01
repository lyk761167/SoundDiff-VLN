# ss_baselines/savi/ppo/sounddiff_policy.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from ss_baselines.savi.models.sounddiff_vln_model import SoundDiffVLNNet


class SoundDiffVLNPolicy(nn.Module):
 

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size: int,
        config,
        goal_sensor_uuid: str,
    ):
        super().__init__()

        
        self.net = SoundDiffVLNNet(
            observation_space=observation_space,
            config=config,
        )


        self._hidden_size = int(getattr(self.net, "output_size", hidden_size))

        self.actor = nn.Linear(self._hidden_size, action_space.n)
        self.critic = nn.Linear(self._hidden_size, 1)

      
        self.last_aux: Optional[Dict[str, torch.Tensor]] = None

    @property
    def num_recurrent_layers(self) -> int:
        return 1

    def _dist(self, features: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.actor(features)
        return torch.distributions.Categorical(logits=logits)

    @staticmethod
    def _empty_em_features(features: torch.Tensor) -> torch.Tensor:
     
        return features.new_zeros((features.size(0), 0))

    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        external_memory: Optional[torch.Tensor] = None,
        external_memory_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
     
        features, new_rnn_hidden_states, aux = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
        )
        self.last_aux = aux

        dist = self._dist(features)
        if deterministic:
            actions = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            actions = dist.sample().unsqueeze(-1)

        action_log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        values = self.critic(features)

        em_features = self._empty_em_features(features)
        
        return values, actions, action_log_probs, new_rnn_hidden_states, em_features

    def get_value(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        external_memory: Optional[torch.Tensor] = None,
        external_memory_masks: Optional[torch.Tensor] = None,
    ):
        features, _, aux = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
        )
        self.last_aux = aux
        return self.critic(features)

    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        action: torch.Tensor,
        external_memory: Optional[torch.Tensor] = None,
        external_memory_masks: Optional[torch.Tensor] = None,
    ):
        features, new_rnn_hidden_states, aux = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
        )
        self.last_aux = aux

        dist = self._dist(features)
        action_log_probs = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()

        values = self.critic(features)
        em_features = self._empty_em_features(features)

        return values, action_log_probs, dist_entropy, new_rnn_hidden_states, em_features
