"""Statistics on pseudobulk and bulk omics data.

``import cytepool`` gives the functions and classes an analysis starts from, by the
question they answer, the default first:

- Cubes, tables and files: :class:`Cube`, :class:`CubeCollection`, :class:`Table`,
  :func:`from_cells`, :func:`from_bulk`, :func:`read_cube`, :func:`write_cube`,
  :func:`from_anndata`, :func:`to_anndata`, :func:`from_mudata`, :func:`to_mudata`.
- Designs: :class:`Design`.
- Which genes change in each cell type: :func:`limma_voom`; for repeated measures,
  :func:`dream_voom`; :func:`limma_trend`; :class:`DifferentialExpressionResult`,
  :class:`GroupFit`.
- Which cell types expand or contract: :func:`propeller`, :class:`CompositionResult`.
- What drives the variation between samples: :func:`variance_partitioning`,
  :class:`VariancePartitioningResult`, :func:`principal_components`,
  :class:`PrincipalComponents`.
- Gene sets and networks: :class:`GeneSets`, :func:`read_gmt`, :func:`write_gmt`,
  :class:`Network`, :func:`collectri`, :func:`progeny`.
- Pathways for one contrast: :func:`camera`, :func:`fry`, :func:`roast`,
  :func:`mroast`, :func:`romer`.
- Pathways along a ranking, or among the changed genes: :func:`gsea`,
  :func:`over_representation`.
- Pathway scores per sample: :func:`gsva`, :func:`ssgsea`, :func:`combined_z_scores`,
  :func:`plage`, :class:`PathwayScores`.
- Regulator activity: :func:`ulm`, :func:`mlm`, :func:`weighted_aggregate`,
  :class:`Activities`.
- Testing scores with a design: :func:`limma`.
- Several studies: :func:`analyse_each_study`, :class:`StudyResults`; combining them,
  :func:`meta_analysis`, :class:`MetaAnalysisResult`; replication, :func:`sumrank`,
  :class:`SumRankResult`, :func:`replication`, :class:`ReplicationResult`,
  :func:`held_out_prediction`, :class:`HeldOutPrediction`.

The building blocks, which work on arrays, are in ``cytepool.stats`` (linear and
mixed models, empirical Bayes moderation, voom, count models, multiple testing,
combined p-values and the distributions) and :mod:`cytepool.normalization`; the parts
of a design are in :mod:`cytepool.design`, and the figures in
:mod:`cytepool.plotting`, which needs matplotlib.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version


def _installed_version() -> str:
    # The metadata of an install names the commit it was built from. The version file
    # that the build also writes can lag behind it: an install reused from the cache
    # leaves the file of a later build in place.
    try:
        return _distribution_version("cytepool")
    except _PackageNotFoundError:  # a source tree on the path, never installed
        try:
            from ._version import version
        except ImportError:
            return "0+unknown"
        return version


# Set before the modules below are imported: some of them record it.
__version__: str = _installed_version()

from .activity import Activities, mlm, ulm, weighted_aggregate  # noqa: E402
from .composition import CompositionResult, propeller  # noqa: E402
from .cube import Cube, CubeCollection, from_bulk, from_cells  # noqa: E402
from .design import Design  # noqa: E402
from .differential_expression import (  # noqa: E402
    DifferentialExpressionResult,
    GroupFit,
    dream_voom,
    limma_trend,
    limma_voom,
)
from .gene_set_enrichment import gsea  # noqa: E402
from .gene_set_tests import camera, fry  # noqa: E402
from .gene_sets import GeneSets, read_gmt, write_gmt  # noqa: E402
from .held_out_studies import HeldOutPrediction, held_out_prediction  # noqa: E402
from .io import read_cube, write_cube  # noqa: E402
from .io.anndata import from_anndata, from_mudata, to_anndata, to_mudata  # noqa: E402
from .meta_analysis_of_studies import MetaAnalysisResult, meta_analysis  # noqa: E402
from .networks import Network, collectri, progeny  # noqa: E402
from .over_representation_analysis import over_representation  # noqa: E402
from .pathway_scores import (  # noqa: E402
    PathwayScores,
    combined_z_scores,
    gsva,
    plage,
    ssgsea,
)
from .principal_component_analysis import (  # noqa: E402
    PrincipalComponents,
    principal_components,
)
from .replication_across_studies import ReplicationResult, replication  # noqa: E402
from .rotation_gene_set_tests import mroast, romer, roast  # noqa: E402
from .score_tests import limma  # noqa: E402
from .study_results import StudyResults, analyse_each_study  # noqa: E402
from .sumrank_across_studies import SumRankResult, sumrank  # noqa: E402
from .table import Table  # noqa: E402
from .variance_components import (  # noqa: E402
    VariancePartitioningResult,
    variance_partitioning,
)

__all__ = [
    # Cubes, tables and files.
    "Cube",
    "CubeCollection",
    "Table",
    "from_cells",
    "from_bulk",
    "read_cube",
    "write_cube",
    "from_anndata",
    "to_anndata",
    "from_mudata",
    "to_mudata",
    # Designs.
    "Design",
    # Which genes change in each cell type?
    "limma_voom",
    "dream_voom",
    "limma_trend",
    "DifferentialExpressionResult",
    "GroupFit",
    # Which cell types expand or contract?
    "propeller",
    "CompositionResult",
    # What drives the variation between samples?
    "variance_partitioning",
    "VariancePartitioningResult",
    "principal_components",
    "PrincipalComponents",
    # Gene sets and networks.
    "GeneSets",
    "read_gmt",
    "write_gmt",
    "Network",
    "collectri",
    "progeny",
    # Pathways for one contrast.
    "camera",
    "fry",
    "roast",
    "mroast",
    "romer",
    # Pathways along a ranking, or among the changed genes.
    "gsea",
    "over_representation",
    # Pathway scores per sample.
    "gsva",
    "ssgsea",
    "combined_z_scores",
    "plage",
    "PathwayScores",
    # Regulator activity.
    "ulm",
    "mlm",
    "weighted_aggregate",
    "Activities",
    # Testing scores with a design.
    "limma",
    # Several studies.
    "analyse_each_study",
    "StudyResults",
    "meta_analysis",
    "MetaAnalysisResult",
    "sumrank",
    "SumRankResult",
    "replication",
    "ReplicationResult",
    "held_out_prediction",
    "HeldOutPrediction",
    # The version.
    "__version__",
]
