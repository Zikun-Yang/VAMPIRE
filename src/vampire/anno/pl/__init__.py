# setting
from ._setting import set_default_plotstyle
from ._plot import tracksplot

# waterfall plot
from ._plot import waterfall
from ._plot import waterfall_legend

# motif variation pattern
from ._logo import logo
from ._logo import logo_from_matrix
from ._motif_msa import motif_msa

# motif abundance
from ._heatmap import heatmap_from_matrix
from ._heatmap import motif_abundance_heatmap
from ._pca import motif_abundance_pca
from ._pca import motif_abundance_pca_variance

# haplotype clustering
from ._heatmap import haplotype_distance_heatmap
from ._plot import haplotype_leiden_res_scan

# copy number variation
from ._violin import copy_number_violin
from ._violin import copy_number_stacked_violin

__all__ = [
    "set_default_plotstyle",
    "tracksplot",
    "waterfall",
    "waterfall_legend",
    "logo",
    "logo_from_matrix",
    "heatmap_from_matrix",
    "motif_abundance_heatmap",
    "haplotype_leiden_res_scan",
    "haplotype_distance_heatmap",
    "motif_abundance_pca",
    "motif_abundance_pca_variance",
    "copy_number_violin",
    "copy_number_stacked_violin",
    "motif_msa",
]