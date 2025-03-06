import logging
from collections import defaultdict

import logging
from collections import defaultdict

import numpy as np
import torch as th
from torch.nn import functional as F

from pvp.pvp_td3 import PVPTD3
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.haco.haco_buffer import concat_samples

logger = logging.getLogger(__name__)


class EIL(PVPTD3):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        stat_recorder = defaultdict(list)

        should_concat = False
        if self.replay_buffer.pos > 0 and self.human_data_buffer.pos > 0:
            replay_data_human = self.human_data_buffer.sample(
                int(batch_size), env=self._vec_normalize_env, return_all=True
            )
            human_data_size = len(replay_data_human.observations)
            human_data_size = max(1, self.extra_config["agent_data_ratio"] * human_data_size)
            human_data_size = int(human_data_size)
            should_concat = True

        elif self.human_data_buffer.pos > 0:
            replay_data = self.human_data_buffer.sample(
                batch_size, env=self._vec_normalize_env, return_all=True
            )
        elif self.replay_buffer.pos > 0:
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        else:
            gradient_steps = 0

        for step in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer

            if self.extra_config["adaptive_batch_size"]:
                if should_concat:
                    replay_data_agent = self.replay_buffer.sample(human_data_size, env=self._vec_normalize_env)
                    replay_data = concat_samples(replay_data_agent, replay_data_human)
            else:

                if self.replay_buffer.pos > batch_size and self.human_data_buffer.pos > batch_size:
                    replay_data_agent = self.replay_buffer.sample(int(batch_size / 2), env=self._vec_normalize_env)
                    replay_data_human = self.human_data_buffer.sample(int(batch_size / 2), env=self._vec_normalize_env)
                    replay_data = concat_samples(replay_data_agent, replay_data_human)
                elif self.human_data_buffer.pos > batch_size:
                    replay_data = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
                elif self.replay_buffer.pos > batch_size:
                    replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                else:
                    break

            # Get current Q-values estimates for each critic network
            current_q_behavior_values = self.critic(replay_data.observations, replay_data.actions_behavior)
            current_q_novice_values = self.critic(replay_data.observations, replay_data.actions_novice)

            stat_recorder["q_value_behavior"].append(current_q_behavior_values[0].mean().item())
            stat_recorder["q_value_novice"].append(current_q_novice_values[0].mean().item())

            new_action = self.actor(replay_data.observations)
            current_q_online_values = self.critic(replay_data.observations, new_action.detach())

            # Compute critic loss
            critic_loss = []
            for (current_q_behavior, current_q_novice, current_q_online) in zip(current_q_behavior_values,
                                                                                current_q_novice_values,
                                                                                current_q_online_values):
                # Good enough agent action
                objective_1 = th.mean(
                    th.minimum(
                        (1 - replay_data.interventions) * current_q_novice,
                        th.zeros_like(current_q_novice),
                    )
                )
                loss_1 = -objective_1

                # Bad state-actions
                objective_2 = th.mean(
                    th.minimum(
                        replay_data.interventions * (-current_q_novice),
                        th.zeros_like(current_q_novice),
                    )
                )
                loss_2 = -objective_2

                # Intervention state-actions:
                objective_3 = (current_q_behavior - current_q_novice) * replay_data.interventions
                objective_3 = th.minimum(objective_3, th.zeros_like(objective_3))
                objective_3 = th.mean(objective_3)
                loss_3 = -objective_3

                l = loss_1 + loss_2 + loss_3

                critic_loss.append(l)
            critic_loss = sum(critic_loss)

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            stat_recorder["train/critic_loss"] = critic_loss.item()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, new_action).mean()

                # BC loss on human data
                bc_loss = F.mse_loss(replay_data.actions_behavior, new_action, reduction="none").mean(axis=-1)
                masked_bc_loss = (replay_data.interventions.flatten() * bc_loss).sum() / (
                        replay_data.interventions.flatten().sum() + 1e-5
                )
                # masked_bc_loss = masked_bc_loss.mean()

                if self.extra_config["only_bc_loss"]:
                    raise ValueError()
                    actor_loss = bc_loss.mean()

                else:
                    if self.extra_config["add_bc_loss"]:
                        raise ValueError()
                        actor_loss += masked_bc_loss * self.extra_config["bc_loss_weight"]

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                stat_recorder["train/actor_loss"] = actor_loss.item()
                stat_recorder["train/masked_bc_loss"] = masked_bc_loss.item()
                stat_recorder["train/bc_loss"] = bc_loss.mean().item()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), np.mean(values))
