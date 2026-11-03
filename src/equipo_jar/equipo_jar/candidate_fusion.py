"""Observation fusion, no ROS: from many noisy detections to a few candidates with an
averaged position and an honest 2x2 covariance.

This is the math behind the `fusion` node. Mission receives every
candidate and decides what to report; it filters on `observaciones` and on the largest
eigenvalue of `covarianza_xy`. So the job here is not to say "confirmed" but to make the
count and the covariance truthful.

Model
-----
Each observation k is a point z_k = (x, y) in `map` with standard deviation sigma_k
(set by detection from the distance). Weight w_k = 1 / sigma_k^2.

Association: z_k joins the closest candidate within `association_dist`; otherwise it
opens a new one. After the update, two candidates closer than `merge_dist` are merged
(the lower id survives, as Candidata.msg promises).

Update (recursive weighted mean, i.e. a static Kalman filter with R_k = sigma_k^2 I):

    mu  <-  (W mu + w_k z_k) / (W + w_k),      W <- W + w_k

Covariance reported for the candidate, 2x2:

    cov = S / n  +  I / W

  - I / W is the covariance of the mean if every observation were independent with its
    own sigma_k: it shrinks as 1/n.
  - S is the weighted sample covariance of the observations around mu (West's online
    algorithm). S / n adds what the observations themselves disagree about, with their
    real shape (an elongated cloud gives an elongated ellipse).
  - Single-viewpoint floor: while the robot poses the candidate was seen from span less
    than `min_baseline`, the diagonal is raised to at least `single_view_sigma^2`. Many
    frames from one spot share the same systematic error (a reflection, a bad range), so
    the covariance must not shrink until a second viewpoint confirms the position.

Purge: a candidate with fewer than `min_obs_to_keep` observations and nothing new for
`stale_timeout_s` is dropped: decoys and stains seen a few times from one place.
"""
import math
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

import numpy as np

Point2 = Tuple[float, float]

# Defaults: the same values as the `fusion` section of params.yaml.
DEFAULTS = {
    #PARAM: association_dist_m = radius (m) to attach an observation to an existing candidate.
    #       Support: typical observation spread (~0.15 m) < value < half the minimum victim
    #       separation (rules: > 1 m apart, so < 0.5). 0.4 never merges two real boxes.
    'association_dist_m': 0.4,
    #PARAM: merge_dist_m = two candidates closer than this are merged. Support: > 0 and
    #       <= association_dist_m. 0.3 catches a box that split in two while the robot moved.
    'merge_dist_m': 0.3,
    #PARAM: min_baseline_m = robot displacement (m) needed before the single-view floor is
    #       lifted. Support: >= 0; 0 disables the floor. 0.3 m changes the viewpoint enough on
    #       a box 1-2 m away.
    'min_baseline_m': 0.3,
    #PARAM: single_view_sigma_m = covariance floor (std, m) while baseline < min_baseline.
    #       Support: > mission's max_desvio_m (0.10) so a one-spot candidate is never
    #       reported; 0.15.
    'single_view_sigma_m': 0.15,
    #PARAM: stale_timeout_s = seconds without observations before a weak candidate is
    #       dropped. Support: > time to see a box again after turning away (a few s); < 600.
    'stale_timeout_s': 20.0,
    #PARAM: min_obs_to_keep = candidates with at least this many observations are never
    #       purged. Support: integer >= 1. Match mission's min_observaciones (5).
    'min_obs_to_keep': 5,
    #PARAM: max_poses = robot poses stored per candidate for the baseline. Support: >= 2.
    #       Only bounds memory and the O(n^2) baseline computation.
    'max_poses': 50,
}


@dataclass
class Candidate:
    """A possible box: the aggregate of every observation associated to it."""
    id: int
    mu: np.ndarray                              # (2,) weighted mean position in `map`
    W: float = 0.0                              # sum of weights = information (1/sigma^2 summed)
    n_obs: int = 0
    M: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))  # West's accumulator
    t_first: float = 0.0
    t_last: float = 0.0
    robot_poses: List[Point2] = field(default_factory=list)

    def baseline(self) -> float:
        """Largest distance between two robot poses this candidate was seen from."""
        if len(self.robot_poses) < 2:
            return 0.0
        pts = np.asarray(self.robot_poses)
        d = pts[:, None, :] - pts[None, :, :]
        return float(np.sqrt((d ** 2).sum(axis=2)).max())

    def sample_cov(self) -> np.ndarray:
        """Weighted sample covariance S of the observations around mu."""
        if self.W <= 0:
            return np.zeros((2, 2))
        return self.M / self.W

    def covariance(self, min_baseline: float, single_view_sigma: float) -> np.ndarray:
        """The 2x2 covariance reported to mission (see module docstring)."""
        cov = self.sample_cov() / max(1, self.n_obs)
        if self.W > 0:
            cov = cov + np.eye(2) / self.W
        if min_baseline > 0 and self.baseline() < min_baseline:
            floor = single_view_sigma ** 2
            cov[0, 0] = max(cov[0, 0], floor)
            cov[1, 1] = max(cov[1, 1], floor)
        return cov

    def max_std(self, min_baseline: float, single_view_sigma: float) -> float:
        """Std along the worst direction: sqrt of the largest eigenvalue. Mission uses this."""
        c = self.covariance(min_baseline, single_view_sigma)
        a, b, d = c[0, 0], c[0, 1], c[1, 1]
        lam = (a + d) / 2 + math.sqrt(((a - d) / 2) ** 2 + b ** 2)
        return math.sqrt(max(lam, 0.0))


class CandidateFusion:
    """Keeps the candidate list and feeds observations into it.

    `params` is a mapping read at call time (the node passes its live-parameter dict), so a
    value changed during a run applies to the next observation.
    """

    def __init__(self, params: Optional[Mapping] = None):
        self.params = params if params is not None else dict(DEFAULTS)
        self.candidates: List[Candidate] = []
        self._next_id = 1
        self.changed = False          # set when the list changed since the last publish

    def _p(self, name):
        return self.params.get(name, DEFAULTS[name])

    # ------------------------------------------------------------------ input

    def observe(self, z: Point2, sigma: float, robot_pose: Optional[Point2], t: float) -> Candidate:
        """Incorporate one observation and return the candidate it joined.

        z:          (x, y) of the box center in `map`, meters.
        sigma:      standard deviation (m) of this observation; weight is 1/sigma^2.
        robot_pose: (x, y) of the robot in `map` at that time, or None if unknown (then the
                    observation does not count toward the baseline).
        t:          observation time (s).
        """
        z = np.asarray(z, dtype=float)
        if not np.isfinite(z).all():
            raise ValueError(f'non-finite observation: {z}')
        w = 1.0 / max(float(sigma), 1e-3) ** 2

        c = self._closest(z, self._p('association_dist_m'))
        if c is None:
            c = Candidate(id=self._next_id, mu=z.copy(), t_first=t)
            self._next_id += 1
            self.candidates.append(c)

        # Weighted mean update and West's covariance accumulator:
        #   M <- M + w (z - mu_old)(z - mu_new)^T
        new_W = c.W + w
        delta_before = z - c.mu
        c.mu = (c.W * c.mu + w * z) / new_W
        delta_after = z - c.mu
        c.M = c.M + w * np.outer(delta_before, delta_after)
        c.W = new_W
        c.n_obs += 1
        c.t_last = t

        if robot_pose is not None:
            px, py = float(robot_pose[0]), float(robot_pose[1])
            max_poses = int(self._p('max_poses'))
            if len(c.robot_poses) < max_poses:
                c.robot_poses.append((px, py))
            else:
                c.robot_poses[c.n_obs % max_poses] = (px, py)

        self.changed = True
        return self._merge_around(c)

    def purge(self, t: float) -> None:
        """Drop weak candidates with no recent observations."""
        keep = [
            c for c in self.candidates
            if c.n_obs >= int(self._p('min_obs_to_keep'))
            or (t - c.t_last) <= float(self._p('stale_timeout_s'))
        ]
        if len(keep) != len(self.candidates):
            self.changed = True
        self.candidates = keep

    # ------------------------------------------------------------------ output

    def report(self) -> List[dict]:
        """Candidates as plain dicts in the shape of Candidata.msg, sorted by id."""
        mb = float(self._p('min_baseline_m'))
        sv = float(self._p('single_view_sigma_m'))
        out = []
        for c in sorted(self.candidates, key=lambda c: c.id):
            cov = c.covariance(mb, sv)
            out.append({
                'id': c.id,
                'x': float(c.mu[0]),
                'y': float(c.mu[1]),
                'cov_xy': [float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 0]), float(cov[1, 1])],
                'n_obs': c.n_obs,
                't_last': c.t_last,
                'max_std': c.max_std(mb, sv),
                'baseline': c.baseline(),
            })
        self.changed = False
        return out

    # ------------------------------------------------------------------ internal

    def _closest(self, z: np.ndarray, radius: float) -> Optional[Candidate]:
        best, d_best = None, None
        for c in self.candidates:
            d = float(np.linalg.norm(z - c.mu))
            if d < radius and (d_best is None or d < d_best):
                best, d_best = c, d
        return best

    def _merge_around(self, c: Candidate) -> Candidate:
        """Merge into `c` any other candidate closer than merge_dist. Lower id survives."""
        radius = float(self._p('merge_dist_m'))
        for other in list(self.candidates):
            # `c` may have been merged away in a previous iteration; the snapshot still
            # lists the removed one, so check membership in the live list.
            if other is c or other not in self.candidates:
                continue
            if float(np.linalg.norm(other.mu - c.mu)) < radius:
                survivor, gone = (c, other) if c.id < other.id else (other, c)
                self._merge_into(survivor, gone)
                self.candidates.remove(gone)
                c = survivor
        return c

    @staticmethod
    def _merge_into(a: Candidate, b: Candidate) -> None:
        """Combine b into a. Means by weight; M by the parallel-axis theorem."""
        W = a.W + b.W
        if W <= 0:
            return
        mu = (a.W * a.mu + b.W * b.mu) / W
        da, db = a.mu - mu, b.mu - mu
        a.M = a.M + b.M + a.W * np.outer(da, da) + b.W * np.outer(db, db)
        a.mu = mu
        a.W = W
        a.n_obs += b.n_obs
        a.t_first = min(a.t_first, b.t_first)
        a.t_last = max(a.t_last, b.t_last)
        a.robot_poses = (a.robot_poses + b.robot_poses)[-int(DEFAULTS['max_poses']):]
