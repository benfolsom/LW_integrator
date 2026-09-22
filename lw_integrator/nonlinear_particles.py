"""Run prepared many-particle full-spin checkpoints on a fixed recording grid."""

from .nonlinear_pair import capabilities as pair_capabilities
from .nonlinear_pair import main as checkpoint_main
from core.momentum_center_pair import MULTIPARTICLE_MODEL


def capabilities():
    result = pair_capabilities()
    result.update(
        model=MULTIPARTICLE_MODEL,
        minimum_particle_count=2,
        source_collection="All non-self particles, summed before nonlinear response",
        adaptive_runner=None,
        time_controller="Fixed shared laboratory recording times with optional internal DOP853 error control",
        pseudo_grid_supported=False,
        pseudogrid_reference="Existing subset selector with exact updates for every particle; no speedup",
        gui_supported=False,
        limitation="Prepared histories required. Reduced pseudogrid, bunch initialization, and GUI integration remain pending. Full conservation and strong-spin validity remain open.",
    )
    return result


def main(argv=None):
    return checkpoint_main(argv, _multiparticle=True)


if __name__ == "__main__":
    raise SystemExit(main())
