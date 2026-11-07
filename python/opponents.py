"""Who plays the opponents of simulated fights: their scripted bots, or a model (self-play).

Why self-play, and why a mix of the latest model, older versions and bots: docs/training-design.md#self-play.

A model-played opponent sees exactly what the model would see from its side (env.rs observe_opponent; the
`an_opponent_sees_as_the_model_does` test is the proof). Each round's opponent has its own LSTM memory, reset when a
new round starts, and plays stochastically (sampled, as the model does while it learns), so it plays like a player
and not a fixed script. Only the model's own experience is learned from; the opponents just play.
"""
import numpy as np
import torch

import mcsim

BOT = None  # this round's opponent is played by its scripted bot (duel.rs)


class Opponents:
    """For SimVecEnv: per round, who plays its first opponent; each tick, their actions."""

    def __init__(self, n):
        self.n = n
        self.players = [BOT] * n  # per round: BOT, or a key into `policies`
        self.policies = {}  # key -> a policy (MemoryPolicy) that plays opponents, e.g. "latest" or a snapshot's name
        self.choose = lambda meta: BOT  # who plays a new fight round (training sets this: the self-play mix)
        self.obs = np.zeros((n, mcsim.MOVE_OBS + mcsim.EXTRA), np.float32)
        self.start = np.ones(n, bool)  # a new round: the opponent's memory starts empty
        self.memory = None  # (h, c), each (layers, n, hidden), on the policies' device

    def new_round(self, i, meta):
        """Round `i` is a new one: choose who plays its opponent (only fights have one)."""
        self.players[i] = self.choose(meta) if meta.get("family") == "duel" else BOT
        self.start[i] = True

    def actions(self, batch):
        """The opponents' actions for the next tick, from what each sees now: a row of -1 leaves a round to its bot."""
        acts = np.full((self.n, mcsim.ACTIONS), -1, np.int64)
        keys = {p for p in self.players if p is not BOT}
        if not keys:
            return acts
        has = np.asarray(batch.observe_opponents(self.obs))
        for key in keys:
            rows = [i for i in range(self.n) if self.players[i] == key and has[i]]
            if rows:
                acts[rows] = self._play(self.policies[key], rows)
        return acts

    @torch.no_grad()
    def _play(self, policy, rows):
        device = policy.device
        if self.memory is None:
            shape = (policy.lstm_actor.num_layers, self.n, policy.lstm_actor.hidden_size)
            self.memory = (torch.zeros(shape, device=device), torch.zeros(shape, device=device))
        idx = torch.as_tensor(rows, device=device)
        state = (self.memory[0][:, idx], self.memory[1][:, idx])
        starts = torch.as_tensor(self.start[rows], dtype=torch.float32, device=device)
        obs = torch.as_tensor(self.obs[rows], device=device)
        dist, (h, c) = policy.get_distribution(obs, state, starts)
        self.memory[0][:, idx] = h
        self.memory[1][:, idx] = c
        self.start[rows] = False
        return dist.get_actions(deterministic=False).cpu().numpy()
