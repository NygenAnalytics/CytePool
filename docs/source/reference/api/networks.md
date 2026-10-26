---
myst:
  html_meta:
    description: Networks of transcription factors and pathways and their target genes in CytePool, with signed weights, such as CollecTRI's regulons, at a licence level for commercial or academic use, and PROGENy's pathway signatures, downloaded from their releases with pinned checksums, or any network given as a table.
---

# Networks

A {class}`~cytepool.networks.Network` maps each source, a transcription factor or a
pathway, to its target genes with signed weights, and records its name, version,
licence, citation and the file it was read from. {func}`~cytepool.networks.collectri`
returns CollecTRI's regulons of human transcription factors, at the licence level you
choose, and {func}`~cytepool.networks.progeny` the pathway signatures of PROGENy for
human or mouse; both download their files on first use, check them against pinned
SHA-256 digests and keep them in a cache. The activity functions of
{mod}`cytepool.activity` take a network, or a table with the columns `source`,
`target` and `weight`.

```{eval-rst}
.. automodule:: cytepool.networks
```
