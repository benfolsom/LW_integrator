"""Run prepared many-particle full-spin checkpoints on a fixed recording grid."""

from typing import Any

from .nonlinear_pair import capabilities as pair_capabilities
from .nonlinear_pair import main as checkpoint_main
from core.momentum_center_pair import MULTIPARTICLE_MODEL


def capabilities() -> dict[str, Any]:
    result = pair_capabilities()
    result.update(
        model=MULTIPARTICLE_MODEL,
        minimum_particle_count=2,
        source_collection="All non-self particles, summed before nonlinear response",
        adaptive_runner=None,
        time_controller="Fixed shared laboratory recording times with optional internal DOP853 error control",
        pseudo_grid_supported=False,
        reduced_motion_supported=True,
        pseudogrid_reference="Existing subset selector with exact updates for every particle; no speedup",
        pseudogrid_reduced="Opt-in RK4/RK3 with existing reaction modes, preserved source increments, and normalized-constraint RK4 fallback; midpoint remains reaction-off; internal step control is not yet supported; every particle remains a retarded source",
        gui_supported=False,
        limitation="Prepared histories required. Mixed internal error control, reduced source approximation, bunch initialization, and GUI integration remain pending. Full conservation and strong-spin validity remain open.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    return checkpoint_main(argv, _multiparticle=True)


if __name__ == "__main__":
    raise SystemExit(main())
