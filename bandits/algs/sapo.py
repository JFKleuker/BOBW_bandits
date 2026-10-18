"""SAPO (Stochastic and Adversarial Pseudo-Optimal), Algorithm 1 of

    P. Auer & C.-K. Chiang. "An algorithm with nearly optimal pseudo-regret for both
    stochastic and adversarial bandits." COLT 2016.

SAPO is an elimination algorithm for stochastic bandits, augmented with tests that detect
non-stochastic arms. Once there is enough evidence for non-stochastic behaviour, it switches
to EXP3.P for the remaining rounds.

Deviations from the paper:
- The paper assumes rewards in [0, 1] and initialises lcb(0) = lcb_bar(0) = 0 and ucb(0) = 1.
  The importance-weighted estimate mu_bar exceeds 1 in early rounds (and Gaussian rewards are
  unbounded), which would trigger the Step 1.a test immediately. We therefore initialise
  the confidence bounds at -inf / +inf.
- Evictions that would empty the active set are skipped (Step 3 needs |A(t)| > 0).
- EXP3.P uses the parameters of Bubeck & Cesa-Bianchi (2012, Thm 3.3) with the remaining
  number of rounds as its horizon.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from .bandit import Bandit


class SAPOState(eqx.Module):
    t: Int[Array, ""]
    """Number of rounds played so far."""
    probs: Float[Array, " K"]
    """Probabilities p_i(t) used in the most recent round."""

    # Statistics
    counts: Int[Array, " K"]  # T_i
    reward_sums: Float[Array, " K"]  # sum x_i I[I_s = i]
    iw_sums: Float[Array, " K"]  # sum x_i I[I_s = i] / p_i(s)
    lcb_hat: Float[Array, " K"]  # lcb_i
    lcb_bar: Float[Array, " K"]  # \bar{lcb}_i
    ucb_bar: Float[Array, " K"]  # ucb_i
    lcb_star: Float[Array, ""]
    regret_1b: Float[Array, ""]  # sum_s [lcb*(s) - x_{I_s}(s)]

    # Active (A) and bad (B) arms
    active: Bool[Array, " K"]
    bad: Bool[Array, " K"]

    # Testing phases of bad arms
    mu_tilde: Float[Array, " K"]
    delta_tilde: Float[Array, " K"]
    L0: Float[Array, " K"]
    L: Float[Array, " K"]
    phase_start: Int[Array, " K"]  # n_i(t)
    max_D: Float[Array, " K"]  # max_{n_i <= s <= t} D_i(s, t)
    detections: Int[Array, " K"]  # E_i

    # EXP3.P
    switched: Bool[Array, ""]
    exp3_start: Int[Array, ""]
    exp3_gains: Float[Array, " K"]

    @property
    def mu_hat(self) -> Float[Array, " K"]:
        return self.reward_sums / jnp.maximum(self.counts, 1)

    @property
    def mu_bar(self) -> Float[Array, " K"]:
        return self.iw_sums / jnp.maximum(self.t, 1)


class SAPO(Bandit):
    """SAPO bandit algorithm (Auer & Chiang, 2016). Constants default to the values in
    Section 3.2 of the paper. These are chosen for the analysis and are very conservative:
    with them, arms are only evicted after a very large number of rounds."""

    delta: float = 0.05
    C_w: float = 16.0
    C_1b: float = 522.0
    C_init: float = 100 / 9
    C_gap: float = 60.0
    C_p: float = 1300.0
    C_4a: float = 1 / 10
    C_E: float = 15.0

    @property
    def log_term(self) -> float:
        return jnp.log(self.num_steps / self.delta)

    def init_state(self, key: PRNGKeyArray, num_arms: int) -> SAPOState:
        K = num_arms
        zeros_f, zeros_i = jnp.zeros(K), jnp.zeros(K, dtype=int)
        return SAPOState(
            t=jnp.array(0),
            probs=jnp.full(K, 1 / K),
            counts=zeros_i,
            reward_sums=zeros_f,
            iw_sums=zeros_f,
            lcb_hat=jnp.full(K, -jnp.inf),
            lcb_bar=jnp.full(K, -jnp.inf),
            ucb_bar=jnp.full(K, jnp.inf),
            lcb_star=jnp.array(-jnp.inf),
            regret_1b=jnp.array(0.0),
            active=jnp.ones(K, dtype=bool),
            bad=jnp.zeros(K, dtype=bool),
            mu_tilde=zeros_f,
            delta_tilde=zeros_f,
            L0=jnp.ones(K),
            L=jnp.ones(K),
            phase_start=zeros_i,
            max_D=zeros_f,
            detections=zeros_i,
            switched=jnp.array(False),
            exp3_start=jnp.array(0),
            exp3_gains=zeros_f,
        )

    def policy(self, state: SAPOState) -> Float[Array, " K"]:
        return self._probs(self._prepare(state))

    def select_arm(
        self, key: PRNGKeyArray, state: SAPOState
    ) -> tuple[Int[Array, ""], SAPOState]:
        state = self._prepare(state)
        probs = self._probs(state)
        arm = jax.random.choice(key, probs.shape[0], p=probs)
        return arm, eqx.tree_at(lambda s: s.probs, state, probs)

    def update(
        self, state: SAPOState, arm: Int[Array, ""], reward: Float[Array, ""]
    ) -> SAPOState:
        return jax.lax.cond(
            state.switched, self._exp3_update, self._sapo_update, state, arm, reward
        )

    def _prepare(self, state: SAPOState) -> SAPOState:
        """Steps 1 and 2 of round t = state.t + 1, using statistics up to round t - 1."""

        def prepare(state: SAPOState) -> SAPOState:
            K = state.counts.shape[0]
            t = state.t + 1

            # Step 1: tests for non-stochastic arms
            mu_bar = state.mu_bar
            outside_ci = (mu_bar < state.lcb_bar) | (mu_bar > state.ucb_bar)
            switch_1a = (state.t > 0) & jnp.any(state.active & outside_ci)
            switch_1b = state.regret_1b > self.C_1b * jnp.sqrt(
                K * self.num_steps * self.log_term
            )
            switch = switch_1a | switch_1b

            # Step 2: evict arms from A
            width_i = jnp.sqrt(self.C_w * self.log_term / jnp.maximum(state.counts, 1))
            evict = (
                state.active
                & (state.counts >= self.C_init * self.log_term)
                & (state.mu_hat + self.C_gap * width_i < state.lcb_star)
            )
            evict = evict & ~jnp.all(evict | ~state.active)  # keep A non-empty
            delta_tilde = self.C_gap * width_i
            L0 = jnp.ceil(self.C_p * K / delta_tilde**2)

            return eqx.tree_at(
                lambda s: (
                    s.active, s.bad, s.mu_tilde, s.delta_tilde, s.L0, s.L,
                    s.phase_start, s.max_D, s.detections, s.switched, s.exp3_start,
                ),
                state,
                (
                    state.active & ~evict,
                    state.bad | evict,
                    jnp.where(evict, state.mu_hat, state.mu_tilde),
                    jnp.where(evict, delta_tilde, state.delta_tilde),
                    jnp.where(evict, L0, state.L0),
                    jnp.where(evict, L0, state.L),
                    jnp.where(evict, t, state.phase_start),
                    jnp.where(evict, 0.0, state.max_D),
                    jnp.where(evict, 0, state.detections),
                    switch,
                    jnp.where(switch, t, state.exp3_start),
                ),
            )  # fmt: skip

        return jax.lax.cond(state.switched, lambda s: s, prepare, state)

    def _probs(self, state: SAPOState) -> Float[Array, " K"]:
        """Step 3 (or EXP3.P's distribution after switching)."""
        K = state.counts.shape[0]
        p_bad = jnp.where(state.bad, state.L0 / (K * state.L), 0.0)
        p_active = jnp.where(
            state.active, (1 - p_bad.sum()) / jnp.sum(state.active), 0.0
        )
        _, eta, gamma = self._exp3_params(state)
        p_exp3 = (1 - gamma) * jax.nn.softmax(eta * state.exp3_gains) + gamma / K
        return jnp.where(state.switched, p_exp3, p_bad + p_active)

    def _sapo_update(
        self, state: SAPOState, arm: Int[Array, ""], reward: Float[Array, ""]
    ) -> SAPOState:
        K = state.counts.shape[0]
        t = state.t + 1
        p = state.probs
        played = jax.nn.one_hot(arm, K)

        # Step 4: test and update bad arms.
        # max_s D_i(s, t) = D_i(t, t) + max(0, max_s D_i(s, t - 1))
        D = (reward - state.mu_tilde) * played + jnp.maximum(state.max_D, 0.0)
        detect = state.bad & (D >= self.C_4a * state.delta_tilde * state.L * p)
        exhausted = state.bad & ~detect & (t >= state.phase_start + state.L - 1)
        new_phase = detect | exhausted
        detections = state.detections + detect
        L = jnp.where(detect, jnp.maximum(state.L / 2, state.L0), state.L)
        L = jnp.where(exhausted, 2 * state.L, L)
        E0 = jnp.ceil(self.C_E * self.log_term)
        switch = jnp.any(state.bad & (detections >= E0))

        # Update statistics and confidence bounds to round t
        counts = state.counts + played.astype(int)
        reward_sums = state.reward_sums + reward * played
        iw_sums = state.iw_sums + reward * played / p[arm]
        mu_hat = reward_sums / jnp.maximum(counts, 1)
        width_i = jnp.sqrt(self.C_w * self.log_term / jnp.maximum(counts, 1))
        lcb_hat = jnp.maximum(
            state.lcb_hat, jnp.where(counts > 0, mu_hat - width_i, -jnp.inf)
        )
        mu_bar = iw_sums / t
        width = jnp.sqrt(self.C_w * K * self.log_term / t)
        lcb_bar = jnp.maximum(state.lcb_bar, mu_bar - width)
        ucb_bar = jnp.minimum(state.ucb_bar, mu_bar + width)
        lcb_star = jnp.maximum(lcb_hat.max(), lcb_bar.max())

        return eqx.tree_at(
            lambda s: (
                s.t, s.counts, s.reward_sums, s.iw_sums, s.lcb_hat, s.lcb_bar,
                s.ucb_bar, s.lcb_star, s.regret_1b, s.L, s.phase_start, s.max_D,
                s.detections, s.switched, s.exp3_start,
            ),
            state,
            (
                t, counts, reward_sums, iw_sums, lcb_hat, lcb_bar,
                ucb_bar, lcb_star, state.regret_1b + lcb_star - reward,
                jnp.where(state.bad, L, state.L),
                jnp.where(new_phase, t + 1, state.phase_start),
                jnp.where(new_phase | ~state.bad, 0.0, D),
                detections, switch, jnp.where(switch, t + 1, state.exp3_start),
            ),
        )  # fmt: skip

    def _exp3_params(self, state: SAPOState) -> tuple[Array, Array, Array]:
        """(beta, eta, gamma) of EXP3.P for the remaining rounds, see
        Bubeck & Cesa-Bianchi (2012), Theorem 3.3."""
        K = state.counts.shape[0]
        m = jnp.maximum(self.num_steps - state.exp3_start + 1, 1)
        beta = jnp.sqrt(jnp.log(K / self.delta) / (m * K))
        eta = 0.95 * jnp.sqrt(jnp.log(K) / (m * K))
        gamma = jnp.minimum(1.05 * jnp.sqrt(K * jnp.log(K) / m), 1.0)
        return beta, eta, gamma

    def _exp3_update(
        self, state: SAPOState, arm: Int[Array, ""], reward: Float[Array, ""]
    ) -> SAPOState:
        beta, _, _ = self._exp3_params(state)
        played = jax.nn.one_hot(arm, state.counts.shape[0])
        gains = state.exp3_gains + (reward * played + beta) / state.probs
        return eqx.tree_at(
            lambda s: (s.t, s.counts, s.exp3_gains),
            state,
            (state.t + 1, state.counts + played.astype(int), gains),
        )
