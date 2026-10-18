import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .bandit import Bandit


class FTLState(eqx.Module):
    counts: Int[Array, " K"]
    reward_sums: Float[Array, " K"]

    @property
    def means(self) -> Float[Array, " K"]:
        return self.reward_sums / jnp.maximum(self.counts, 1)


class FTL(Bandit):
    """Follow-the-leader: play every arm once, then always play the arm with the
    highest empirical mean (ties broken by lowest index)."""

    def init_state(self, key: PRNGKeyArray, num_arms: int) -> FTLState:
        return FTLState(
            counts=jnp.zeros(num_arms, dtype=int),
            reward_sums=jnp.zeros(num_arms),
        )

    def policy(self, state: FTLState) -> Float[Array, " K"]:
        unplayed = state.counts == 0
        leader = jnp.where(
            jnp.any(unplayed), jnp.argmax(unplayed), jnp.argmax(state.means)
        )
        return jax.nn.one_hot(leader, state.counts.shape[0])

    def update(
        self, state: FTLState, arm: Int[Array, ""], reward: Float[Array, ""]
    ) -> FTLState:
        return FTLState(
            counts=state.counts.at[arm].add(1),
            reward_sums=state.reward_sums.at[arm].add(reward),
        )
