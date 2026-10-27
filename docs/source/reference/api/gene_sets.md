---
myst:
  html_meta:
    description: Gene-set collections in CytePool, read from and written to the GMT files of MSigDB, Reactome, WikiPathways and the Gene Ontology, with each collection's name, version and licence, matched to the genes of an analysis and filtered by size.
---

# Gene-set files

{func}`~cytepool.read_gmt` reads a GMT file, one gene set per line, as MSigDB,
Reactome, WikiPathways and the GMT exports of the Gene Ontology write them, into a
{class}`~cytepool.GeneSets`: a mapping of set name to gene ids that every gene-set
function of CytePool takes as its `gene_sets`, with each set's description and the
name, version and licence of the collection, which you state when you read it, and the
name and SHA-256 digest of the file. CytePool bundles no collection.
{func}`~cytepool.write_gmt` writes a collection back.

```{eval-rst}
.. automodule:: cytepool.gene_sets
   :no-members:
   :no-index:

.. autoclass:: cytepool.GeneSets

.. autofunction:: cytepool.read_gmt

.. autofunction:: cytepool.write_gmt
```
