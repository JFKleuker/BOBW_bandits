"""Sweep of bandit algorithms over stochastic bandit environments.

For every number of arms K in `NUM_ARMS`, `ENVS_PER_ARM * K` environments are sampled with
random means (uniform in [0, 1]) and a random variance shared by all arms of that environment.
Each algorithm in `ALGORITHMS` is trained on every environment for every horizon in
`NUM_STEPS` and every train seed. All seeds and environments of one K are run in parallel.

The pseudo-regret of every (environment, algorithm, num_steps, train_seed) combination is
appended to a single CSV. Re-running the script skips (K, algorithm, num_steps) blocks that
are already in the CSV, so an interrupted sweep can be resumed.

Usage:
    python train/sweep.py                           # full sweep
    python train/sweep.py --num-seeds 4 --envs-per-arm 1 --num-steps 1000 --num-arms 2 3  # smoke test
"""

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bandits.algs import FTL, FTRLBOBW, SAPO, Bandit
from bandits.envs import StochasticBanditEnv

ALGORITHMS: dict[str, type[Bandit]] = {"FTL": FTL, "SAPO": SAPO, "FTRLBOBW": FTRLBOBW}
NUM_STEPS = [1_000, 10_000, 50_000, 100_000]
NUM_ARMS = list(range(2, 11))
ENVS_PER_ARM = 25  # number of environments for K arms = ENVS_PER_ARM * K
NUM_SEEDS = 100
VARIANCE_RANGE = (0.01, 0.25)


def sample_envs(num_arms: int, num_envs: int, env_seed: int) -> list[dict]:
    """Random means in [0, 1] and one variance per environment, shared by all its arms."""
    rng = np.random.default_rng([env_seed, num_arms])
    return [
        {
            "env_id": f"K{num_arms}_{i}",
            "means": rng.uniform(0.0, 1.0, size=num_arms),
            "variance": rng.uniform(*VARIANCE_RANGE),
        }
        for i in range(num_envs)
    ]


def stack_envs(env_specs: list[dict], max_episode_steps: int) -> StochasticBanditEnv:
    """Batches environments into a single env pytree with a leading batch axis."""
    envs = [
        StochasticBanditEnv(s["means"], s["variance"], max_episode_steps)
        for s in env_specs
    ]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *envs)


@eqx.filter_jit
def train_batch(algo: Bandit, envs: StochasticBanditEnv, keys) -> jax.Array:
    """Pseudo-regret of shape (num_envs, num_seeds)."""

    def regret(env, key):
        _, totals = algo.train(key, env, per_step_metrics=False)
        return totals["regret"]

    return jax.vmap(lambda env: jax.vmap(lambda key: regret(env, key))(keys))(envs)


def is_supported(algo: Bandit, num_arms: int) -> bool:
    """Algorithms assert their requirements (e.g. FTRLBOBW needs K >= 3) in `init_state`."""
    try:
        algo.init_state(jax.random.PRNGKey(0), num_arms)
    except AssertionError as e:
        print(f"  skipping {type(algo).__name__} for K={num_arms}: {e}")
        return False
    return True


def main(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        existing = pd.read_csv(out, usecols=["num_arms", "algorithm", "num_steps"])
        done = set(existing.drop_duplicates().itertuples(index=False, name=None))

    seeds = np.arange(args.num_seeds)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))

    for num_arms in args.num_arms:
        env_specs = sample_envs(num_arms, args.envs_per_arm * num_arms, args.env_seed)
        envs = stack_envs(env_specs, max_episode_steps=max(args.num_steps))
        env_info = pd.DataFrame(
            {
                "num_arms": num_arms,
                "env_id": [s["env_id"] for s in env_specs],
                "means": [";".join(f"{m:.6f}" for m in s["means"]) for s in env_specs],
                "variance": [s["variance"] for s in env_specs],
                "min_gap": [
                    np.min(s["means"].max() - s["means"][s["means"] < s["means"].max()])
                    for s in env_specs
                ],
            }
        )

        for name in args.algorithms:
            for num_steps in args.num_steps:
                if (num_arms, name, num_steps) in done:
                    print(f"K={num_arms} {name} n={num_steps}: already done")
                    continue
                algo = ALGORITHMS[name](num_steps=num_steps)
                if not is_supported(algo, num_arms):
                    continue

                start = time.time()
                regrets = []
                for i in range(0, len(env_specs), args.env_chunk_size):
                    chunk = jax.tree.map(lambda x: x[i : i + args.env_chunk_size], envs)
                    regrets.append(np.asarray(train_batch(algo, chunk, keys)))
                regrets = np.concatenate(regrets)  # (num_envs, num_seeds)

                rows = env_info.loc[env_info.index.repeat(len(seeds))].reset_index(drop=True)
                rows.insert(2, "algorithm", name)
                rows.insert(3, "num_steps", num_steps)
                rows.insert(4, "train_seed", np.tile(seeds, len(env_specs)))
                rows["regret"] = regrets.reshape(-1)
                rows.to_csv(out, mode="a", header=not out.exists(), index=False)
                print(
                    f"K={num_arms} {name} n={num_steps}: {len(env_specs)} envs x "
                    f"{len(seeds)} seeds, mean regret {regrets.mean():.2f} "
                    f"({time.time() - start:.1f}s)"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="results/sweep.csv")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--num-steps", nargs="+", type=int, default=NUM_STEPS)
    parser.add_argument("--num-arms", nargs="+", type=int, default=NUM_ARMS)
    parser.add_argument("--envs-per-arm", type=int, default=ENVS_PER_ARM)
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument(
        "--env-chunk-size",
        type=int,
        default=1_000,
        help="Environments trained in parallel (each with all seeds); lower if out of memory.",
    )
    main(parser.parse_args())
