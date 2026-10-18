from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

import jaxnasium as jym


class BanditState(eqx.Module):
    time: Int[Array, ""]


@jym.registry.register("StochasticBandit")
class StochasticBanditEnv(jym.Environment):
    """K-armed stochastic bandit with Gaussian rewards.

    Playing arm `i` yields a reward `x ~ N(means[i], variances[i])`.
    The bandit is stateless, so the observation is a constant dummy value.
    The info dict contains the pseudo-regret `max_j means[j] - means[i]` of the played arm.

    **Arguments:**

    - `means`: Mean reward per arm (length K).
    - `variance`: Reward variance, either per arm (length K) or a single float shared by all arms.
    - `max_episode_steps`: Number of pulls after which an episode is truncated.
    """

    means: Float[Array, " K"]
    variances: Float[Array, " K"]
    max_episode_steps: int = eqx.field(static=True)

    def __init__(
        self,
        means: Sequence[float] | Array,
        variance: float | Sequence[float] | Array = 1.0,
        max_episode_steps: int = 1000,
    ):
        self.means = jnp.asarray(means, dtype=float)
        assert self.means.ndim == 1, "`means` must be a 1D list of length K."
        variances = jnp.broadcast_to(jnp.asarray(variance, dtype=float), self.means.shape)
        assert jnp.all(variances >= 0), "Variances must be non-negative."
        self.variances = variances
        self.max_episode_steps = max_episode_steps

    @property
    def num_arms(self) -> int:
        return self.means.shape[0]

    @property
    def optimal_mean(self) -> Float[Array, ""]:
        return jnp.max(self.means)

    def step_env(
        self, key: PRNGKeyArray, state: BanditState, action: Int[Array, ""]
    ) -> tuple[jym.TimeStep, BanditState]:
        noise = jax.random.normal(key)
        reward = self.means[action] + jnp.sqrt(self.variances[action]) * noise
        state = BanditState(time=state.time + 1)
        timestep = jym.TimeStep(
            observation=self.get_observation(state),
            reward=reward,
            terminated=jnp.array(False),
            truncated=state.time >= self.max_episode_steps,
            info={"regret": self.optimal_mean - self.means[action]},
        )
        return timestep, state

    def reset_env(self, key: PRNGKeyArray) -> tuple[Array, BanditState]:
        state = BanditState(time=jnp.array(0))
        return self.get_observation(state), state

    def get_observation(self, state: BanditState) -> Array:
        return jnp.array(0)

    @property
    def observation_space(self) -> jym.Space:
        return jym.Discrete(n=1)

    @property
    def action_space(self) -> jym.Space:
        return jym.Discrete(n=self.num_arms)
