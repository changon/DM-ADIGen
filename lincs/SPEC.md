# LINCS L1000 as a Causal `X, A, Y` Dataset

## 1. Data download

This uses the Phase II LINCS L1000 release, [`GSE70138`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138).

```bash
mkdir -p lincs/GSE70138
cd lincs/GSE70138

base="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl"

# Metadata
curl -LO "$base/GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz"
curl -LO "$base/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
curl -LO "$base/GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz"
curl -LO "$base/GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz"

# Normalized well-level expression used for the causal analysis
curl -LO "$base/GSE70138_Broad_LINCS_Level3_INF_mlr12k_n345976x12328_2017-03-06.gctx.gz"
```

The Level 3 download is approximately 12.61 GiB compressed. It is a
gzip-compressed GCTX/HDF5 file, not a CSV. Download the complete file before
extracting selected wells and genes.

```bash
python -m pip install cmapPy pandas
gzip -dk GSE70138_Broad_LINCS_Level3_INF_mlr12k_n345976x12328_2017-03-06.gctx.gz
```

The URLs and metadata files were tested on August 31, 2026. The complete
multi-gigabyte expression matrix was not downloaded during verification.

## 2. Data description

Phase II contains:

- `345,976` Level 3 instances. An **instance** is one experimental well.
- `12,328` expression features: 978 measured landmark genes and 11,350 inferred
  genes.
- 98 cell records in `cell_info`.
- 2,170 perturbation records in `pert_info`.

The files connect as follows:

```text
inst_info.inst_id    -> Level 3 matrix columns
gene_info.pr_gene_id -> Level 3 matrix rows
pert_info.pert_id    -> inst_info.pert_id
```

`inst_info` supplies cell line, compound, dose, exposure time, plate, and well.
`gene_info.pr_is_lm == 1` selects the 978 directly measured genes.

Inside GCTX, genes are rows and wells are columns. Transpose the selected matrix
to obtain the usual `n wells x p genes` layout.

Level 5 is not used in the primary causal specification. Its 118,050 columns
are replicate-collapsed differential **signatures**, not independent wells.

## 3. Multi-dose causal specification

Start with one well-covered cell line and fix exposure time, for example 24
hours. Retain chemical perturbations and corresponding vehicle-control wells.

```text
Unit: one Level 3 experimental well
X: baseline context
E: plate/batch covariates
A: compound and dose assignment
Y: post-treatment expression of 978 measured landmark genes
```

The analysis shapes after filtering are:

```text
X: n x p_X
A: n x 1 for one compound, or n x 2 for compound plus dose
Y: n x 978
```

Here, `n` is the number of retained treated and vehicle-control wells. It must be counted after filtering and is smaller than 345,976. `p_X` depends on the chosen plate/batch encoding; LINCS does not provide a rich pretreatment vector for every well.

```text
A_i in {0, d1, d2, ..., dD}
```

Use `A_i = 0` for vehicle and the administered concentration for treated wells.
The dose-specific causal contrast is:

```text
tau(d) = E[Y(d) - Y(0)]
```

For several compounds across several doses:

```text
A_i = (compound_i, log_dose_i)
```

For the primary outcome, either model the 978-gene vector or predefine a smaller set of pathway scores.


## Related papers using LINCS

- [Subramanian et al. (2017), *A Next Generation Connectivity Map*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5990023/): foundational L1000 platform, processing levels, and first large public release.
- [Hodos et al. (2018), *Cell-specific prediction and application of drug-induced gene expression profiles*](https://pubmed.ncbi.nlm.nih.gov/29218867/): models the drug-by-cell-line-by-gene response tensor and predicts missing profiles.
- [DeepCOP (2020)](https://academic.oup.com/bioinformatics/article/36/3/813/5554893): uses cell-line-specific Level 5 data and restricts analysis primarily to 24-hour, 10-micromolar treatments.
- [MultiDCP (2022)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010367): represents observations by drug, cell line, dose, and time with 978-gene outcomes.
- [DOSE-L1000 (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10663987/): performs dose-response modeling over 33,395 compounds, 82 cell lines, and 978 landmark genes.
