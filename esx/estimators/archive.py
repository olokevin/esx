"""Archive of parameter-space states with PUCT selection (TTT-Discover App. A.2), adapted to noisy rewards.

A state is a point theta_s = theta_0 + sum over its path of (seeds, coeffs) updates. Per state:
  R(s)  its own reward estimate (measured member reward, or the soft value of the batch it was tilted from);
  kids  rewards of every member sampled around s over all of s's expansions;
  n(s)  visits: +1 when s or any descendant is expanded (backed up to ancestors), as in the paper;
  Q(s)  = entropic soft value of kids if s was expanded, else R(s).  The paper uses the max child reward; with
        4-episode Bernoulli rewards the max only ever grows and keeps lucky states forever, so the soft value
        (between mean and max, same KL rule as the update) is used instead (stated deviation);
  P(s)  rank prior (|H| - rank(s)) / sum (|H| - rank), rank 0 = highest R.
score(s) = Q(s) + c * scale * P(s) * sqrt(1 + T) / (1 + n(s)),  scale = max R - min R over the archive, T = expansions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from esx.estimators.entropic import entropic


@dataclass
class State:
    sid: int
    path: list  # [(seeds, coeffs), ...] from theta_0
    R: float
    parent: Optional[int]
    kind: str  # root | tilt | member
    kids: List[float] = field(default_factory=list)
    n: int = 0
    expanded: int = 0

    def Q(self, gamma: float) -> float:
        return entropic(self.kids, gamma)[2] if self.kids else self.R


class Archive:
    def __init__(self, root_R: float, c: float = 1.0, cap: int = 64, gamma: float = math.log(2.0)):
        self.states = {0: State(0, [], root_R, None, "root")}
        self.c, self.cap, self.gamma, self.T, self.next = c, cap, gamma, 0, 1

    def add(self, path, R, parent, kind) -> State:
        s = State(self.next, path, float(R), parent, kind)
        self.states[s.sid] = s
        self.next += 1
        if len(self.states) > self.cap:  # drop the lowest-R unexpanded non-root state
            cand = [x for x in self.states.values() if x.sid != 0 and x.expanded == 0 and x.sid != s.sid]
            if cand:
                del self.states[min(cand, key=lambda x: x.R).sid]
        return s

    def scores(self) -> dict:
        H = list(self.states.values())
        order = sorted(H, key=lambda x: -x.R)
        rank = {x.sid: i for i, x in enumerate(order)}
        den = sum(len(H) - rank[x.sid] for x in H)
        Rs = [x.R for x in H]
        scale = (max(Rs) - min(Rs)) or 1.0
        return {x.sid: x.Q(self.gamma) + self.c * scale * ((len(H) - rank[x.sid]) / den) * math.sqrt(1 + self.T)
                / (1 + x.n) for x in H}

    def select(self) -> State:
        sc = self.scores()
        return self.states[max(sc, key=lambda k: (sc[k], -k))]

    def backup(self, s: State, rewards) -> None:
        s.kids.extend(map(float, rewards))
        s.expanded += 1
        a: Optional[int] = s.sid
        while a is not None:
            if a in self.states:
                self.states[a].n += 1
                a = self.states[a].parent
            else:
                break
        self.T += 1

    def incumbent(self) -> State:
        """Best verified state: the expanded state with the highest Q (falls back to the root)."""
        ex = [x for x in self.states.values() if x.expanded]
        return max(ex, key=lambda x: x.Q(self.gamma)) if ex else self.states[0]
