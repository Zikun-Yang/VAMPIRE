from ._setting import set_default_plotstyle
from ._plot import trackplot
from ._plot import waterfall
from ._plot import waterfall_legend
from ._plot import logo
from ._plot import logo_from_matrix
from ._plot import heatmap_from_matrix
from ._plot import motif_abundance_heatmap
from ._plot import haplotype_evaluation
from ._plot import haplotype_distance_heatmap
from ._plot import motif_abundance_pca
from ._plot import motif_abundance_pca_variance
from ._plot import copy_number_violin
from ._plot import copy_number_stacked_violin

__all__ = [
    "set_default_plotstyle",
    "trackplot",
    "waterfall",
    "waterfall_legend",
    "logo",
    "logo_from_matrix",
    "heatmap_from_matrix",
    "motif_abundance_heatmap",
    "haplotype_evaluation",
    "haplotype_distance_heatmap",
    "motif_abundance_pca",
    "motif_abundance_pca_variance",
    "copy_number_violin",
    "copy_number_stacked_violin",
]