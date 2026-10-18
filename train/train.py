import bandits  # noqa: F401  (registers StochasticBandit)
import jax
import jax.numpy as jnp

import jaxnasium as jym
from bandits.algs import FTL, FTRLBOBW, SAPO, Bandit

# jym.enable_compilation_cache() # optional


def run(algo: Bandit, env: jym.Environment, key, num_seeds: int = 20):
    """Trains `algo` on `num_seeds` independent runs (vectorized) and reports the
    pseudo-regret and the expected reward of the final policies."""
    keys = jax.random.split(key, num_seeds)
    states, metrics = jax.vmap(lambda k: algo.train(k, env))(keys)

    regret = jnp.cumsum(metrics["regret"], axis=-1)[:, -1]
    final_probs = jax.vmap(algo.policy)(states)
    policy_value = final_probs @ env.means
    print(
        f"{type(algo).__name__:>8} | pseudo-regret {regret.mean():8.2f} ± {regret.std():7.2f}"
        f" | final policy value {policy_value.mean():.3f} (optimal {env.optimal_mean:.3f})"
    )
    return states, metrics


if __name__ == "__main__":
    # jym.make wraps a LogWrapper by default, which is not needed here
    env = jym.make(
        "StochasticBandit",
        wrappers=[],
        means=[0.2, 0.5, 0.45, 0.8, 0.75],
        variance=0.1,
    )
    rng = jax.random.PRNGKey(0)
    num_steps = 10_000

    print(f"Random agent pseudo-regret: {num_steps * (env.optimal_mean - env.means.mean()):.2f}")
    # run(FTL(num_steps=num_steps), env, rng)
    run(SAPO(num_steps=num_steps), env, rng)  # constants from the paper (very conservative)
    # run(SAPO(num_steps=num_steps, C_w=1.0, C_init=1.0, C_gap=2.0, C_p=10.0), env, rng)
    run(FTRLBOBW(num_steps=num_steps), env, rng)
