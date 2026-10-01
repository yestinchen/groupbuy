'''build the retrieval caches up front. this is the only GPU step; the
benches read cached slots from disk and stay CPU-only.'''

from __future__ import annotations

import click

from gquery.alg.exp4.core import _slot_cache_path, load_instances
from gquery.utils import setup_logging

logger = setup_logging("exp4_prewarm")


@click.command()
@click.option("--settings", type=str, default="e10_1k_100q",
              help="Comma-separated query settings.")
@click.option("--betas", type=str, default="0.8")
@click.option("--alpha", type=float, default=0.5)
@click.option("--num-queries", type=int, default=100)
@click.option("--force", is_flag=True, default=False)
def main(settings, betas, alpha, num_queries, force):
    setting_list = [s.strip() for s in settings.split(",") if s.strip()]
    beta_list = [float(b) for b in betas.split(",") if b.strip()]
    for setting in setting_list:
        for beta in beta_list:
            path = _slot_cache_path(setting, num_queries, beta, alpha)
            if path.exists() and not force:
                logger.info(f"cached already: {setting} beta={beta}")
                continue
            try:
                insts, _ = load_instances(setting, num_queries, beta, alpha,
                                          use_disk_cache=not force)
            except Exception as exc:
                logger.error(f"{setting} beta={beta}: {exc!r}")
                continue
            if force:
                path.parent.mkdir(parents=True, exist_ok=True)
                import pickle
                with open(path, "wb") as fh:
                    pickle.dump(insts, fh, protocol=pickle.HIGHEST_PROTOCOL)
            n_c = sum(i.n_candidates for i in insts) / max(len(insts), 1)
            logger.info(f"{setting} beta={beta}: {len(insts)} instances, "
                        f"{n_c:,.0f} candidates/query -> {path}")


if __name__ == "__main__":
    main()
