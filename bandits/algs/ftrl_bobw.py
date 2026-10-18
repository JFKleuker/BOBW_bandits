"""FTRL with real-time Stability-Penalty Matching (SPM), Algorithm 1 of

    Q. Nguyen, S. Ito, J. Komiyama & N. A. Mehta. "Data-dependent Bounds with T-Optimal
    Best-of-Both-Worlds Guarantees in Multi-Armed Bandits using Stability-Penalty Matching."
    arXiv:2502.08143, 2025.

Each round, FTRL computes

    q_t = argmin_{p in simplex} <L_{t-1}, p> + beta_t * psi_TE(p) + gamma * psi_LB(p),

with the alpha-Tsallis entropy psi_TE(p) = (1 - sum_i p_i^alpha) / alpha and the log-barrier
psi_LB(p) = -sum_i ln(p_i), mixes in 1/T uniform exploration, and tunes beta_t by SPM.

The algorithm minimises losses; we use loss = -reward. The paper assumes losses in [-1, 1].
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .bandit import Bandit


class FTRLBOBWState(eqx.Module):
    cum_loss_estimates: Float[Array, " K"]  # L_{t-1}
    beta: Float[Array, ""]  # beta_t
    probs: Float[Array, " K"]
    """Sampling probabilities p_t used in the most recent round."""


class FTRLBOBW(Bandit):
    """Real-time SPM with hybrid (Tsallis entropy + log-barrier) regularization.

    Attributes:
        alpha: Tsallis entropy parameter in (0, 1). Defaults to 1 - 1 / (2 ln K), as
            suggested in the paper (Appendix G).
        d: Constant in the stability term z_t (d = 2 in the paper).
        solver_iters: Bisection iterations for the Lagrange multiplier of q_t.
        newton_iters: Newton iterations per arm for each multiplier candidate.
    """

    alpha: float | None = None
    d: float = 2.0
    solver_iters: int = eqx.field(static=True, default=50)
    newton_iters: int = eqx.field(static=True, default=30)

    def _alpha(self, num_arms: int) -> float:
        if self.alpha is not None:
            return self.alpha
        return 1 - 1 / (2 * jnp.log(num_arms))

    def _gamma(self, num_arms: int) -> float:
        alpha = self._alpha(num_arms)
        return jnp.maximum(6.0, 48 * jnp.sqrt(alpha / (1 - alpha)))

    def init_state(self, key: PRNGKeyArray, num_arms: int) -> FTRLBOBWState:
        assert num_arms >= 3, "FTRL-BOBW (Algorithm 1) requires K >= 3."
        assert self.num_steps >= 4 * num_arms, "FTRL-BOBW requires T >= 4K."
        return FTRLBOBWState(
            cum_loss_estimates=jnp.zeros(num_arms),
            beta=jnp.asarray(8 * num_arms / (1 - self._alpha(num_arms))),
            probs=jnp.full(num_arms, 1 / num_arms),
        )

    def policy(self, state: FTRLBOBWState) -> Float[Array, " K"]:
        K = state.cum_loss_estimates.shape[0]
        q = self._ftrl_distribution(state.cum_loss_estimates, state.beta)
        return (1 - K / self.num_steps) * q + 1 / self.num_steps

    def select_arm(
        self, key: PRNGKeyArray, state: FTRLBOBWState
    ) -> tuple[Int[Array, ""], FTRLBOBWState]:
        probs = self.policy(state)
        arm = jax.random.choice(key, probs.shape[0], p=probs)
        return arm, eqx.tree_at(lambda s: s.probs, state, probs)

    def update(
        self, state: FTRLBOBWState, arm: Int[Array, ""], reward: Float[Array, ""]
    ) -> FTRLBOBWState:
        K = state.probs.shape[0]
        alpha, gamma, d = self._alpha(K), self._gamma(K), self.d
        p = state.probs
        p_arm = p[arm]
        loss = -reward
        loss_estimate = loss / p_arm

        # Stability term z_t, penalty term h_t = -psi_TE(p_t) and SPM learning rate update
        z = jnp.minimum(
            (6 * d) ** (2 - alpha)
            / (2 * (1 - alpha))
            * jnp.minimum(p_arm, 1 - p_arm) ** (2 - alpha)
            * loss_estimate**2,
            state.beta * 18 * d**2 / gamma * loss**2,
        )
        h = (jnp.sum(p**alpha) - 1) / alpha
        beta = state.beta + z / (state.beta * h)

        return FTRLBOBWState(
            cum_loss_estimates=state.cum_loss_estimates.at[arm].add(loss_estimate),
            beta=beta,
            probs=p,
        )

    def _ftrl_distribution(
        self, L: Float[Array, " K"], beta: Float[Array, ""]
    ) -> Float[Array, " K"]:
        """Solves q = argmin_{p in simplex} <L, p> + beta psi_TE(p) + gamma psi_LB(p).

        The KKT conditions give, for each arm, L_i + lam = beta q_i^(alpha-1) + gamma / q_i
        =: g(q_i). g is positive, convex and strictly decreasing, so q_i(lam) is unique and
        decreasing in lam. We bisect on lam such that sum_i q_i(lam) = 1, solving each q_i
        with Newton's method started left of the root (monotone convergence for convex g).
        """
        K = L.shape[0]
        alpha, gamma = self._alpha(K), self._gamma(K)
        L = L - L.min()  # shift invariance; now min L = 0

        def solve_q(lam):
            c = L + lam  # > 0
            q = jnp.maximum(gamma / c, (beta / c) ** (1 / (1 - alpha)))  # lower bounds

            def newton(_, q):
                g = beta * q ** (alpha - 1) + gamma / q - c
                dg = beta * (alpha - 1) * q ** (alpha - 2) - gamma / q**2
                return q - g / dg

            return jax.lax.fori_loop(0, self.newton_iters, newton, q)

        # At lam_lo the leading arm has q >= 1; at lam_hi every arm has q <= 1/K.
        lam_lo = beta + gamma
        lam_hi = beta * K ** (1 - alpha) + gamma * K

        def bisect(_, bounds):
            lo, hi = bounds
            mid = (lo + hi) / 2
            too_large = solve_q(mid).sum() > 1
            return jnp.where(too_large, mid, lo), jnp.where(too_large, hi, mid)

        lo, hi = jax.lax.fori_loop(0, self.solver_iters, bisect, (lam_lo, lam_hi))
        q = solve_q((lo + hi) / 2)
        return q / q.sum()
