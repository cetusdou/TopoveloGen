from data_generation_module import (
    GeneSynthesizer,
    attach_default_gene_modules,
    build_topology,
)


def test_build_topology_supported_types():
    for trajectory_type in ["linear", "bifurcation", "cycle", "cycle_bifurcation"]:
        system, start = build_topology(trajectory_type)
        assert system is not None
        assert isinstance(start, str)
        assert any(node.id == start for node in system.nodes)


def test_attach_default_gene_modules_bifurcation_no_cycle_module():
    system, _ = build_topology("bifurcation")
    gs = GeneSynthesizer(system, n_genes_total=20, seed=42)
    attach_default_gene_modules(gs, trajectory_type="bifurcation", n_genes=20)
    kinds = [m["kind"] for m in gs.modules]
    assert "cycle" not in kinds


def test_attach_default_gene_modules_cycle_has_cycle_module():
    system, _ = build_topology("cycle")
    gs = GeneSynthesizer(system, n_genes_total=20, seed=42)
    attach_default_gene_modules(gs, trajectory_type="cycle", n_genes=20)
    kinds = [m["kind"] for m in gs.modules]
    assert "cycle" in kinds
