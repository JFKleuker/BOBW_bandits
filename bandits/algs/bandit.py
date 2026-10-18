from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import Environment


class Bandit(eqx.Module):
    """Base class for bandit algorithms.

    A heavily simplified version of jaxnasium's `RLAlgorithm`. The algorithm object holds the
    hyperparameters, while all learned quantities live in a separate state pytree (returned by
    `init_state` and `train`). There are no neural networks: subclasses maintain exact
    statistics and define the policy (a distribution over arms) directly from them.

    Subclasses implement `init_state`, `policy` and `update`. Algorithms that need to do
    bookkeeping before choosing an arm (e.g. SAPO's elimination step) override `select_arm`.

    Attributes:
        num_steps: Horizon n, i.e. the number of arm pulls during training.
    """

    num_steps: int = eqx.field(static=True, default=1000)

    @abstractmethod
    def init_state(self, key: PRNGKeyArray, num_arms: int) -> PyTree:
        """Builds the initial (untrained) state for a bandit with `num_arms` arms."""
        ...

    @abstractmethod
    def policy(self, state: PyTree) -> Float[Array, " K"]:
        """Probability of choosing each arm in the next round."""
        ...

    @abstractmethod
    def update(
        self, state: PyTree, arm: Int[Array, ""], reward: Float[Array, ""]
    ) -> PyTree:
        """Updates the state after pulling `arm` and observing `reward`."""
        ...

    def select_arm(
        self, key: PRNGKeyArray, state: PyTree
    ) -> tuple[Int[Array, ""], PyTree]:
        """Samples an arm from the policy. Returns the arm and the (possibly updated) state."""
        probs = self.policy(state)
        return jax.random.choice(key, probs.shape[0], p=probs), state

    @eqx.filter_jit
    def train(
        self, key: PRNGKeyArray, env: Environment, state: PyTree | None = None
    ) -> tuple[PyTree, dict[str, Array]]:
        """Plays `num_steps` rounds on `env`, updating the state online.

        Returns the final state and per-round metrics (arm, reward, pseudo-regret).
        """
        num_arms = self.__check_env__(env)
        init_key, reset_key, key = jax.random.split(key, 3)
        if state is None:
            state = self.init_state(init_key, num_arms)
        _, env_state = env.reset(reset_key)

        def round_(carry, _):
            state, env_state, key = carry
            key, arm_key, step_key = jax.random.split(key, 3)
            arm, state = self.select_arm(arm_key, state)
            (_, reward, _, _, info), env_state = env.step(step_key, env_state, arm)
            state = self.update(state, arm, reward)
            metrics = {"arm": arm, "reward": reward, "regret": info["regret"]}
            return (state, env_state, key), metrics

        (state, _, _), metrics = jax.lax.scan(
            round_, (state, env_state, key), None, length=self.num_steps
        )
        return state, metrics

    @eqx.filter_jit
    def evaluate(
        self, key: PRNGKeyArray, state: PyTree, env: Environment, num_steps: int = 1000
    ) -> Float[Array, ""]:
        """Average reward of the frozen policy (no updates) over `num_steps` pulls."""
        probs = self.policy(state)

        def pull(key):
            arm_key, reset_key, step_key = jax.random.split(key, 3)
            arm = jax.random.choice(arm_key, probs.shape[0], p=probs)
            _, env_state = env.reset(reset_key)
            (_, reward, _, _, _), _ = env.step(step_key, env_state, arm)
            return reward

        return jnp.mean(jax.vmap(pull)(jax.random.split(key, num_steps)))

    def __check_env__(self, env: Environment) -> int:
        """Checks that `env` is a single-agent bandit and returns its number of arms."""
        assert not env.multi_agent, "Bandit algorithms only support single-agent envs."
        assert isinstance(env.action_space, jym.Discrete), (
            "Bandit algorithms require a `Discrete` action space."
        )
        return env.action_space.n
