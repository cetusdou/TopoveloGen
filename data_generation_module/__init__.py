"""
data_generation_module
======================

Package entry for the data generation helpers used in MSD-NPDE.

Expose the primary classes and utilities so that other modules and notebooks
can import them as `from data_generation_module import ...`.

"""
from .graph_core import GraphNode, CycleNode, SkeletonNode, GraphEdge, UniversalGraphSystem
from .synthesizer_gene import GeneSynthesizer
from .layout_engine import AutoLayout
from .utils import GraphSimulator, GraphVisualizer
from .presets import build_topology, attach_default_gene_modules

__all__ = [
    "GraphNode",
    "CycleNode",
    "SkeletonNode",
    "GraphEdge",
    "UniversalGraphSystem",
    "GeneSynthesizer",
    "AutoLayout",
    "GraphSimulator",
    "GraphVisualizer",
    "build_topology",
    "attach_default_gene_modules",
]
