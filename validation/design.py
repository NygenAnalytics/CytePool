"""Check design matrices, design checks and contrasts against R, limma and lme4.

The script builds random sample tables (24 samples; factors with 2 to 4 levels,
including mixed-case labels, labels with spaces and a non-syntactic column name, a
boolean, two covariates that are multiples of 1/16, an integer covariate, and
covariates built to be badly scaled or nearly collinear) and compares
``cytepool.design`` with R run as a separate program:

- ``model.matrix`` on a list of formulas covering main effects, numeric covariates,
  factor:factor, factor:numeric and numeric:numeric interactions, no intercept,
  nesting with ``/`` and ``%in%``, powers, ``-`` removal, booleans, terms whose
  margins are missing and badly scaled covariates, each under three level orders
  passed explicitly to R: sorted, a custom order with other reference levels and
  an unused last level, and reversed with an unused reference level. Column names,
  the ``assign`` attribute, term labels and every matrix entry must be identical;
  products are exact in both programs, so matrices are compared for equality.
- The same comparison on 300 random formulas from the full grammar (``+ - : * / ^
  %in%``, parentheses, ``0``, ``1``, ``-1``), and R's ``terms()`` on 282 grammar
  cases without data. Formulas that R rejects must be rejected too; formulas that
  CytePool rejects by design (a ``-`` that removes nothing, ``^`` of a single
  term, a fractional power, function calls) are counted separately.
- Default level order: CytePool's code-point order against R with
  ``LC_COLLATE=C`` and no explicit levels; R's own locale order is shown for
  information.
- ``Design.check()`` against R's pivoted QR (``qr()``, tolerance 1e-7, as
  ``lm.fit`` uses it): the rank, the aliased columns, which the pivot puts last,
  and the leverages, the row sums of squares of the first rank columns of Q. Then
  contrast estimability on the rank-deficient designs: R draws a contrast ``X'u``,
  which is estimable by construction, and a random contrast and the unit contrast
  of an aliased column, which are not.
- Random-effect terms against lme4's ``lFormula``: the name of every grouping
  factor and the columns of its effects.
- limma's ``makeContrasts`` on linear expressions in syntactic column names.

Requirements:
    Python: numpy, and cytepool from the source tree.
    R (4.4 or newer) with the packages limma (Bioconductor) and lme4 (CRAN); the
    rest is base R. Set RSCRIPT to the Rscript executable if it is not on PATH.

Usage, from the repository root:
    PYTHONPATH=. python validation/design.py
    PYTHONPATH=. python validation/design.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from cytepool.design import Design, make_contrasts, model_matrix, parse_formula

SEED = 20261023
N_SAMPLES = 24
N_RANDOM_FORMULAS = 300
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "design"
SEPARATOR = "\x1f"  # joins names in the files exchanged with R

FACTOR_LEVELS = {
    "a": ["ctl", "trtA", "trtB"],
    "b": ["X", "Y"],
    "c": ["hi", "lo", "mid"],
    "d": ["d1", "d2", "d3", "d4"],
    "e": ["A", "B", "a", "b"],
    "cell type": ["B cell", "CD4 T", "NK"],
}
NUMERIC_COLUMNS = [
    "x",
    "z",
    "w",
    "big",
    "offset5",
    "offset9",
    "x_eps6",
    "x_eps8",
    "x_eps10",
    "tiny",
    "huge",
    "below",
    "above",
]
LEVEL_ORDERS: dict[str, dict[str, list[Any]]] = {
    "sorted": {name: sorted(levels) for name, levels in FACTOR_LEVELS.items()}
    | {"flag": [False, True]},
    "custom": {
        "a": ["trtB", "ctl", "trtA", "trtC"],  # trtC has no samples
        "b": ["Y", "X"],
        "c": ["lo", "mid", "hi"],
        "d": ["d3", "d1", "d4", "d2"],
        "e": ["b", "A", "a", "B"],
        "cell type": ["NK", "B cell", "CD4 T"],
        "flag": [True, False],
    },
    "reversed": {name: sorted(levels)[::-1] for name, levels in FACTOR_LEVELS.items()}
    | {"b": ["Z", "Y", "X"], "flag": [True, False]},  # the reference Z has no samples
}

FORMULAS = {
    "main effects": [
        "~ a",
        "~ a + b",
        "~ b + a",
        "~ a + b + c + d",
        "~ e",
        "~ a + x",
        "~ x + a",
        "~ a + b + x + z + flag",
        "~ w + a",
        "~ `cell type` + x",
        "~ 1 + a",
    ],
    "numeric covariates": ["~ x", "~ x + z", "~ x + z + w", "~ 1 + x", "~ w"],
    "factor:factor": [
        "~ a * b",
        "~ a + b + a:b",
        "~ a*b*c",
        "~ a*b + a:b:c",
        "~ a:b:c",
        "~ b + a:b:c",
        "~ a:b + c:d",
        "~ (a+b):(c+d)",
        "~ c + a:c",
        "~ b:a + a",
        "~ a*e",
        "~ `cell type`*b",
    ],
    "factor:numeric": [
        "~ a * x",
        "~ a:x",
        "~ x + a:x",
        "~ a + a:x",
        "~ x:a + b",
        "~ a/x",
        "~ x/a",
        "~ a*b*x",
        "~ (a+b)^2:x",
        "~ a + x:z:b",
        "~ flag * x",
        "~ `cell type`:x",
    ],
    "numeric:numeric": ["~ x * z", "~ x:z", "~ x:z:w", "~ (x + z + w)^2", "~ 0 + x:z"],
    "no intercept": [
        "~ 0 + a",
        "~ -1 + a",
        "~ a - 1",
        "~ a + 0",
        "~ 0 + a + b",
        "~ 0 + b + a",
        "~ 0 + x + a",
        "~ x + 0 + a",
        "~ 0 + a*b",
        "~ 0 + c*d",
        "~ 0 + e + b",
        "~ 0 + flag + a",
        "~ 0 + x:z + a",
        "~ 0 + a:x",
        "~ 0 + x + a:b",
        "~ 0 + a:b + c",
        "~ a*b - 1",
    ],
    "nesting (/ and %in%)": [
        "~ a/b",
        "~ b %in% a",
        "~ a + b %in% a",
        "~ (a+b)/c",
        "~ a/b/c",
        "~ a/(b + c)",
        "~ a/(1 + x) - 1",
        "~ x %in% a",
        "~ a + b %in% a + c %in% b",
        "~ (a + b) %in% c",
    ],
    "powers (^)": [
        "~ (a+b+c)^2",
        "~ (a+b+c)^3",
        "~ (a + b + c + d)^2",
        "~ (a:b + c + d)^2",
        "~ a + (b+c)^2",
        "~ (x + a + b)^2",
    ],
    "removal (-)": [
        "~ (a+b+c)^2 - a:b",
        "~ a*b*c - a:b:c",
        "~ a*b - a",
        "~ a*b + c - a:b",
        "~ (a + b + c)^3 - a:b:c - a:c",
        "~ a/b - a",
    ],
    "booleans": ["~ flag", "~ 0 + flag", "~ flag:a", "~ flag*b", "~ 0 + flag:x"],
    "missing margins": [
        "~ a:b",
        "~ 0 + a:b",
        "~ 0 + x + x:a",
        "~ x + x:a",
        "~ 0 + x + a + x:a",
        "~ 0 + a:b + a",
        "~ a:x + b:x",
        "~ 0 + x:a + x:b",
        "~ c + a:c:d",
        "~ a + a:b",
        "~ 0 + a + a:b",
    ],
}
FORMULAS["badly scaled covariates"] = [
    "~ big",
    "~ a + big",
    "~ offset5",
    "~ offset9",
    "~ 0 + a + offset9",
    "~ x + x_eps6",
    "~ x + x_eps8",
    "~ x + x_eps10",
    "~ tiny + z",
    "~ huge + z",
    "~ a + tiny:z",
    "~ x + below",
    "~ x + above",
    "~ x*big",
]
DEFAULT_ORDER_FORMULAS = ["~ e", "~ a * e", "~ 0 + e + `cell type`", "~ flag + c"]
# Formulas R also rejects, and formulas R accepts that CytePool rejects on purpose
# because their R meaning is almost never what was intended.
REJECTED_BY_BOTH = [
    "~ (a+b)^1",
    "~ a + 2",
    "~ (a + b)^(2)",
    "~ (a+b)^x",
    "~ a +",
    "~ (a + b",
]
REJECTED_BY_DESIGN = [
    "~ a - b",
    "~ x - z",
    "~ a + (-b)",
    "~ x^2",
    "~ (a + b)^2.5",
    "~ log(z)",
    "~ I(x^2)",
    "~ factor(w)",
]
# Message fragments of the rejections that CytePool makes on purpose.
DESIGN_REJECTIONS = (
    "changes nothing",
    "crosses the terms",
    "whole number",
    "function calls",
)
RANDOM_VARIABLES = ["a", "b", "c", "e", "flag", "x", "z"]
# Grammar cases compared with R's terms() alone (no data): intercept settings
# inside operands, removal, operands without terms, nesting, term order of
# powers crossed with other terms, powers past the point where every term is
# crossed, and names that need backticks.
TERMS_FORMULAS = [
    "~ a:1",
    "~ 1:a",
    "~ a/(1+x)",
    "~ a/(1+x) - 1",
    "~ 0:a",
    "~ a:0",
    "~ -a",
    "~ -1",
    "~ 1",
    "~ 0",
    "~ -0",
    "~ a - 0",
    "~ 0 + a + 1",
    "~ a - 1 + 1",
    "~ 1 + a - 1",
    "~ -1 + 1 + a",
    "~ 0 + a - 0",
    "~ a + 0 + 1",
    "~ a + 1 + 0",
    "~ 1 - 1",
    "~ 0 + 1",
    "~ 1 + 0",
    "~ (a+b)^1",
    "~ a^2",
    "~ (a+b)^0",
    "~ (a+b)^2.5",
    "~ (a+b)^-1",
    "~ (a+b+c)^3",
    "~ (a + b:c)^2",
    "~ (a:b + c)^2",
    "~ a + (-b)",
    "~ a - (b + c)",
    "~ (a + b + c) - (a + b)",
    "~ a*b - a",
    "~ a - a:b",
    "~ -a + a",
    "~ a + b - (b - a)",
    "~ a %in% (b + c)",
    "~ (a + b) %in% c",
    "~ (a+b) %in% (c+d)",
    "~ a %in% b %in% c",
    "~ (a:b) %in% c",
    "~ a %in% b:c",
    "~ (a + b)/(c + d)",
    "~ a/b/c",
    "~ a/(b + c)",
    "~ (a + b:c)/d",
    "~ a/b - a",
    "~ a + b %in% a",
    "~ a:a",
    "~ a*a",
    "~ a + b + a",
    "~ b + a:b + a",
    "~ a:b + b:a",
    "~ z - z + x:z",
    "~ b - b + a:b",
    "~ (a - 1):b",
    "~ a:(b - 1)",
    "~ (a + 0):b",
    "~ (0 + a)*b",
    "~ a + (b - 1)",
    "~ a + (0 + b)",
    "~ (1 + a)",
    "~ x:x",
    "~ x^2",
    "~ x*x",
    "~ 0 + x + x:a",
    "~ 0 + a:x + b",
    "~ 0 + x:a + x:b",
    "~ 0 + x:z + a",
    "~ 0 + x + a:b",
    "~ 2 + a",
    "~ a + 2",
    "~ a - 2",
    "~ -a:b",
    "~ a:-b",
    "~ a * b %in% c",
    "~ a / b %in% c",
    "~ a:b^2",
    "~ (a+b)^2:c",
    "~ a*(b + 1)",
    "~ a:(1 + b)",
    "~ (1 + a):(1 + b)",
    "~ a/1",
    "~ 1/a",
    "~ a %in% 1",
    "~ 1 %in% a",
    "~ -(a + b) + a*b",
    "~ a - (1)",
    "~ a + (1)",
    "~ a + (0)",
    "~ a + -1",
    "~ a + +b",
    "~ +a",
    "~ a - -b",
    "~ (a)",
    "~ ((a + b))",
    "~ 0 + (1 + a):b",
    "~ 0 + a/(1 + x)",
    "~ 1 + (0 + a):b",
    "~ a - (b + 1)",
    "~ a - (b - 1)",
    "~ a - (b + 0)",
    "~ a - (0 + b)",
    "~ 0 + a - (b - 1)",
    "~ (a - 1) - (b - 1)",
    "~ a + b - (1)",
    "~ a - (-1)",
    "~ a - (-0)",
    "~ -(a - 1)",
    "~ -(1)",
    "~ -(0)",
    "~ - -1",
    "~ a*b - (a:b - 1)",
    "~ (a - b)",
    "~ (a - 1)",
    "~ (a + b - a):c",
    "~ (a + b - a)*c",
    "~ a:(b - b)",
    "~ (a - a)*b",
    "~ a %in% (b - 1)",
    "~ (a - 1) %in% b",
    "~ (a - 1)^2",
    "~ (a + b - 1)^2",
    "~ (a + 0)^2",
    "~ a/(b - 1)",
    "~ (a - 1)/b",
    "~ a/(0 + x)",
    "~ a/(1 + x) + 0",
    "~ 0 + a/(1 + x) - 1",
    "~ 1 + a:(0 + b)",
    "~ (1)",
    "~ (0)",
    "~ -(-a)",
    "~ (-1)",
    "~ (-1) + a",
    "~ a + (-1)",
    "~ a + (-0)",
    "~ (a - 1) + 1",
    "~ 1 + (a - 1)",
    "~ a - b:1",
    "~ a*b - b:1",
    "~ a - (a:b):1",
    "~ (a + 1)^2",
    "~ (1 + a)^2",
    "~ (a + b)^2 - 1",
    "~ 1:1",
    "~ 0:1",
    "~ 1:0",
    "~ -1:a",
    "~ a:1 - 1",
    "~ a %in% 0",
    "~ 0 %in% a",
    "~ a/0",
    "~ 0/a",
    "~ a * 0",
    "~ a * 1",
    "~ 1 * a",
    "~ 0 * a",
    "~ a + b - a:b - b",
    "~ b + a - b",
    "~ (a:b - a:b) + c",
    "~ a %in% (b + 0)",
    "~ a %in% (1 + b)",
    "~ (1 + a) %in% b",
    "~ (0 + a) %in% b",
    "~ x %in% a",
    "~ a %in% x",
    "~ (1 + x):a",
    "~ a:(x + 1) - 1",
    "~ x + (0 + a)",
    "~ (x - 1) + a",
    "~ 1 - a",
    "~ 0 - a",
    "~ -a - 1",
    "~ -1 - a",
    "~ -0 - 1 + a",
    "~ 1 * (0 + a)",
    "~ 1/(0 + a)",
    "~ (a - a)*(b - 1)",
    "~ 1 %in% (0 + a)",
    "~ a + 1.0",
    "~ a - 1.0",
    "~ a + 0.0",
    "~ a + 1e0",
    "~ a + 1L",
    "~ (a+b)^(2)",
    "~ a^1",
    "~ (a + b)^2L",
    "~ (a+b)^3.9",
    "~ (a+b)^x",
    "~ a + b - (a - 1)",
    "~ (a - a)^2",
    "~ a^2^2",
    "~ (a+b)^2^1",
    "~ -a^2",
    "~ (a:b - 1):c",
    "~ (a + b)*(c - 1)",
    "~ 0 + x + a",
    "~ 0 + x + a + b",
    "~ x + 0 + a",
    "~ (a + b):(c + d)",
    "~ a + b - a + a",
    "~ a:b - b:a",
    "~ (a + b - b)/c",
    "~ a %in% (b + c - c)",
    "~ (a + 1)/b",
    "~ 1*z + x:z",
    "~ (a + b):a",
    "~ flag",
    "~ 0 + flag",
    "~ flag:x",
    "~ `cell type`",
    "~ `cell type`:x",
    "~ `my var` + a",
    "~ `a b`:`c d`",
    "~ `if`",
    "~ `1x`",
    "~ `x.y`",
    "~ x.y",
    "~ x_y",
    "~ lv",
    "~ 0 + lv",
    "~ (a + b + c)^2 - a:b",
    "~ a*b*c - a:b:c",
    "~ (a + b + c + d)^2",
    "~ (a:b + c + d)^2",
    "~ a + b + a:b + c",
    "~ c + a:c",
    "~ a:b:c",
    "~ a*b + a:b:c",
    "~ b + a:b:c",
    "~ a + x:z:b",
    "~ (a+b)^2:x",
    "~ a*b + c - a:b",
    "~ x:a + b",
    "~ 0 + c*d",
    "~ x/a",
    "~ a/x",
    "~ 0 + x + a + x:a",
    "~ x + x:a",
    "~ a:b + c:d",
    "~ 0 + a:b + a",
    "~ 0 + a:b + c",
    "~ x*z",
    "~ x:z:x",
    "~ (x + z)^2",
    "~ 0 + x:z",
    "~ a + b %in% a + c %in% b",
    "~ (c*x):(z + a:b)",
    "~ ((c + x)^2):(z + a:b)",
    "~ ((c + x)/d):(z + a:b:e)",
    "~ (c*x*w):(z + a:b:e)",
    "~ ((c + x + w)^2):(z + a:b)",
    "~ ((c + x + w)^3):(z + a:b:e)",
    "~ ((c + x:w)^2):(z + a:b)",
    "~ ((c:w + x)^2):(z + a:b)",
    "~ ((c + x)*(w + d)):(z + a:b)",
    "~ (c*(x + w)):(z + a:b)",
    "~ ((c + x) %in% d):(z + a:b)",
    "~ (c + x:d + w):(z + a:b)",
    "~ ((c + x)^2 - c:x + c:x):(z + a:b)",
    "~ ((c*x) - x + x):(z + a:b)",
    "~ ((c + x)^3):(z + a:b)",
    "~ ((c + x + w)^3):(z + a:b)",
    "~ ((c/x)*w):(z + a:b:e)",
    "~ (c/(x + w)):(z + a:b)",
    "~ ((c + x)/(w + d)):(z + a:b:e)",
    "~ ((c:x + w)*d):(z + a:b)",
    "~ (w * (c + x)^2):(z + a:b)",
    "~ (g + e + c:d + a:d:g + e:f + b:d:e)^2",
    "~ ((g + e + c:d + a:d:g + e:f + b:d:e)^2):(z + y:w:v:u)",
    "~ (g + e + c:d + a:d:g + e:f + b:d:e)^3",
    "~ ((g + e + c:d + a:d:g + e:f + b:d:e)^3):(z + y:w:v:u)",
    "~ (g + e + c:d + a:d:g + e:f + b:d:e)^4",
    "~ ((g + e + c:d + a:d:g + e:f + b:d:e)^4):(z + y:w:v:u)",
    "~ (g + e + c:d + a:d:g + e:f + b:d:e)^5",
    "~ ((g + e + c:d + a:d:g + e:f + b:d:e)^5):(z + y:w:v:u)",
    "~ (g + e + c:d + a:d:g + e:f + b:d:e)^6",
    "~ ((g + e + c:d + a:d:g + e:f + b:d:e)^6):(z + y:w:v:u)",
    "~ (g + e + c:d + a:d:g + e:f + b:d:e)^7",
    "~ ((g + e + c:d + a:d:g + e:f + b:d:e)^7):(z + y:w:v:u)",
]
RANDOM_EFFECTS = [
    "(1 | a)",
    "(1 | a/b)",
    "(1 | a:b/c)",
    "(1 | a/b:c)",
    "(1 | c:b/a)",
    "(1 | a/(b:c))",
    "(1 | (a/b)/c)",
    "(1 | a:b)",
    "(1 | b:a)",
    "(1 | d/a)",
    "(x | d)",
    "(x || d)",
    "(0 + x | d)",
    "(1 + x + z | e)",
    "(b | d)",
    "(0 + b | d)",
    "(1 | `cell type`)",
    "(1 | `cell type`:d)",
    "(1 | d/`cell type`)",
    "(1 | a) + (x | b)",
    "(z || c:b)",
]

# The contrasts R draws for each rank-deficient design: X'u, a random vector and
# the unit vector of an aliased column.
ESTIMABLE_BY_CONSTRUCTION = [True, False, False]

CONTRAST_COLUMNS = ["groupA", "groupB", "groupC", "batchb2", "x"]
CONTRASTS = [
    "groupB - groupA",
    "groupC - groupB",
    "groupA - (groupB + groupC)/2",
    "(groupA + groupB + groupC)/3 - x",
    "2*groupC - 0.5*groupB",
    "batchb2 - x/4",
    "-groupA",
    "1e-1*groupA + 2.5*groupB - .5*groupC",
    "((groupA + groupB)/2 - groupC)*2",
    "groupB/3*3 - groupA",
    "(groupB - groupA) - (groupC - groupA)",
    "+groupB - +groupA",
    "groupC*0.25 + 0.75*groupB - groupA",
]
NONLINEAR_CONTRASTS = ["groupB - 1", "groupA*groupB", "groupA/groupB", "groupA^2"]

R_PROGRAM = textwrap.dedent(
    r"""
    # Reference model matrices, QR ranks, leverages and estimability from R, terms
    # from R's terms(), contrasts from limma and random-effect terms from lme4.
    args <- commandArgs(trailingOnly = TRUE)
    dir <- args[1]
    sep <- "\x1f"
    path <- function(name) file.path(dir, name)
    lines <- function(name) readLines(path(name), encoding = "UTF-8", warn = FALSE)
    write_doubles <- function(con, values) writeBin(as.double(values), con, size = 8,
                                                    endian = "little")
    clean <- function(message) gsub("[\t\n\r]", " ", message)
    text <- read.delim(path("samples.tsv"), quote = "", comment.char = "",
                       colClasses = "character", check.names = FALSE,
                       na.strings = character(0), encoding = "UTF-8")
    n <- nrow(text)
    con <- file(path("numeric.bin"), "rb")
    for (name in lines("numeric_names.txt")) {
      text[[name]] <- readBin(con, "double", n, size = 8, endian = "little")
    }
    close(con)
    text$flag <- text$flag == "TRUE"
    level_orders <- list()
    for (file in list.files(dir, pattern = "^levels_.*[.]txt$")) {
      order_name <- sub("^levels_(.*)[.]txt$", "\\1", file)
      parsed <- strsplit(lines(file), sep, fixed = TRUE)
      level_orders[[order_name]] <- setNames(lapply(parsed, `[`, -1),
                                             vapply(parsed, `[`, "", 1))
    }
    data_for <- function(order_name) {
      frame <- text
      if (order_name %in% names(level_orders)) {
        for (name in names(level_orders[[order_name]])) {
          values <- if (is.logical(frame[[name]])) {
            ifelse(frame[[name]], "TRUE", "FALSE")
          } else frame[[name]]
          frame[[name]] <- factor(values, levels = level_orders[[order_name]][[name]])
        }
      }
      frame
    }

    # Model matrices, QR, leverages; estimability on rank-deficient matrices.
    cases <- strsplit(lines("cases.txt"), "\t", fixed = TRUE)
    frames <- list()
    out <- character(length(cases))
    estimability <- character(0)
    matrices <- file(path("matrices.bin"), "wb")
    leverages <- file(path("leverage.bin"), "wb")
    contrast_file <- file(path("estimability.bin"), "wb")
    default_collation <- Sys.getlocale("LC_COLLATE")
    for (i in seq_along(cases)) {
      order_name <- cases[[i]][1]
      formula <- cases[[i]][2]
      if (is.null(frames[[order_name]])) frames[[order_name]] <- data_for(order_name)
      invisible(Sys.setlocale("LC_COLLATE",
                              if (order_name == "implicit_C") "C" else default_collation))
      result <- tryCatch({
        X <- model.matrix(as.formula(formula), frames[[order_name]])
        q <- qr(X)
        p <- ncol(X)
        aliased <- if (q$rank < p) colnames(X)[q$pivot[(q$rank + 1):p]] else character(0)
        write_doubles(matrices, X)
        write_doubles(leverages, rowSums(qr.Q(q)[, seq_len(q$rank), drop = FALSE]^2))
        if (q$rank < p) {
          # Contrasts whose estimability is known by construction: X'u lies in
          # the row space; a random vector and the unit vector of an aliased
          # column do not.
          set.seed(i)
          weights <- cbind(crossprod(X, rnorm(n)), rnorm(p), as.numeric(seq_len(p) == q$pivot[p]))
          write_doubles(contrast_file, weights)
          estimability <- c(estimability, as.character(i))
        }
        tt <- terms(as.formula(formula))
        paste("OK", p, paste(colnames(X), collapse = sep),
              paste(attr(X, "assign"), collapse = ","),
              paste(attr(tt, "term.labels"), collapse = sep), attr(tt, "intercept"),
              q$rank, paste(aliased, collapse = sep), "", sep = "\t")
      }, error = function(e) paste("ERROR", 0, "", "", "", "", "", "", clean(conditionMessage(e)), sep = "\t"))
      out[i] <- result
    }
    close(matrices)
    close(leverages)
    close(contrast_file)
    invisible(Sys.setlocale("LC_COLLATE", default_collation))
    writeLines(out, path("results.txt"), useBytes = TRUE)
    writeLines(estimability, path("estimability.txt"), useBytes = TRUE)

    # Terms of formulas without data.
    terms_out <- vapply(lines("terms_cases.txt"), function(formula) tryCatch({
      tt <- terms(as.formula(formula))
      paste("OK", attr(tt, "intercept"), paste(attr(tt, "term.labels"), collapse = sep),
            sep = "\t")
    }, error = function(e) paste("ERROR", "", clean(conditionMessage(e)), sep = "\t")), "")
    writeLines(terms_out, path("terms_results.txt"), useBytes = TRUE)

    # limma contrasts.
    suppressPackageStartupMessages(library(limma))
    spec <- lines("contrasts.txt")
    columns <- strsplit(spec[1], sep, fixed = TRUE)[[1]]
    weights <- makeContrasts(contrasts = spec[-1], levels = columns)
    con <- file(path("contrasts.bin"), "wb")
    write_doubles(con, weights)
    close(con)
    writeLines(paste(colnames(weights), collapse = sep), path("contrast_names.txt"),
               useBytes = TRUE)
    nonlinear <- suppressWarnings(makeContrasts(contrasts = lines("nonlinear_contrasts.txt"),
                                                levels = columns))
    writeLines(apply(nonlinear, 2, function(v) paste(format(v), collapse = " ")),
               path("nonlinear_results.txt"), useBytes = TRUE)

    # lme4 grouping factors and effect columns.
    suppressPackageStartupMessages(library(lme4))
    frame <- data_for("sorted")
    set.seed(1)
    frame$response <- rnorm(n)
    control <- lmerControl(check.nobs.vs.nlev = "ignore", check.nobs.vs.nRE = "ignore",
                           check.nlev.gtr.1 = "ignore", check.rankX = "ignore")
    random_out <- vapply(lines("random_effects.txt"), function(spec) tryCatch({
      parsed <- lFormula(as.formula(paste("response ~ 1 +", spec)), frame, control = control)
      cnms <- parsed$reTrms$cnms
      paste(paste(names(cnms), vapply(cnms, paste, "", collapse = "+"), sep = "="),
            collapse = sep)
    }, error = function(e) paste("ERROR:", clean(conditionMessage(e)))), "")
    writeLines(random_out, path("random_results.txt"), useBytes = TRUE)
    writeLines(c(R.version.string, as.character(packageVersion("limma")),
                 as.character(packageVersion("lme4")), default_collation),
               path("versions.txt"), useBytes = TRUE)
    """
)


def simulate_samples(rng: np.random.Generator, n: int) -> dict[str, Any]:
    """A sample table in which every level occurs, with exact and stress covariates.

    x and z are multiples of 1/16 and w is an integer, so their products are exact.
    The other covariates sit around R's aliasing tolerance of 1e-7: big hardly
    varies around 3.2e7, offset5 and offset9 are x shifted by 1e5 and 1e9, x_eps*
    differ from x by noise of relative size 1e-6 to 1e-10, tiny and huge are x
    scaled by 1e-200 and 1e200, and below and above differ from x, orthogonally to
    the intercept and x, by 0.9 and 1.1 times the tolerance.
    """

    def balanced(levels: list[str]) -> list[str]:
        values = [levels[i % len(levels)] for i in range(n)]
        return [values[i] for i in rng.permutation(n)]

    columns: dict[str, Any] = {name: balanced(lv) for name, lv in FACTOR_LEVELS.items()}
    columns["flag"] = np.array(balanced(["TRUE", "FALSE"])) == "TRUE"
    x = rng.integers(-48, 49, size=n) / 16
    columns["x"] = x
    columns["z"] = rng.integers(1, 161, size=n) / 8
    columns["w"] = rng.integers(18, 81, size=n)
    columns["big"] = 3.2e7 + rng.integers(0, 4, size=n)
    columns["offset5"] = 1e5 + x
    columns["offset9"] = 1e9 + x
    noise = rng.standard_normal(n)
    for exponent in (6, 8, 10):
        columns[f"x_eps{exponent}"] = x + 10.0**-exponent * noise
    columns["tiny"] = 1e-200 * x
    columns["huge"] = 1e200 * x
    basis = np.linalg.qr(np.column_stack([np.ones(n), x]))[0]
    away = noise - basis @ (basis.T @ noise)
    away /= np.linalg.norm(away)
    for name, ratio in (("below", 0.9e-7), ("above", 1.1e-7)):
        columns[name] = x + ratio * np.linalg.norm(x) * away
    return columns


def random_formulas(rng: np.random.Generator, count: int) -> list[str]:
    """Random formulas from the full grammar, without exact repeats."""

    def expression(depth: int) -> str:
        if depth == 0 or rng.random() < 0.3:
            return str(rng.choice(RANDOM_VARIABLES))
        choice = rng.random()
        if choice < 0.18:
            return f"({expression(depth - 1)} + {expression(depth - 1)})"
        if choice < 0.34:
            return f"{expression(depth - 1)}:{expression(depth - 1)}"
        if choice < 0.50:
            return f"{expression(depth - 1)}*{expression(depth - 1)}"
        if choice < 0.62:
            return f"{expression(depth - 1)}/{expression(depth - 1)}"
        if choice < 0.72:
            return f"{expression(depth - 1)} %in% {expression(depth - 1)}"
        if choice < 0.84:
            members = rng.choice(
                RANDOM_VARIABLES, size=int(rng.integers(2, 4)), replace=False
            )
            return f"({' + '.join(members)})^{int(rng.integers(2, 4))}"
        return f"({expression(depth - 1)} + 1)"

    formulas: list[str] = []
    while len(formulas) < count:
        body = expression(3)
        if rng.random() < 0.5:
            body = f"{body} + {expression(2)}"
        roll = rng.random()
        if roll < 0.15:
            body = f"0 + {body}"
        elif roll < 0.25:
            body = f"{body} - 1"
        elif roll < 0.30:
            body = f"-1 + {body}"
        formula = f"~ {body}"
        if rng.random() < 0.35:
            formula = _with_removal(rng, formula)
        if formula not in formulas and _n_columns_bound(formula) <= 80:
            formulas.append(formula)
    return formulas


def _with_removal(rng: np.random.Generator, formula: str) -> str:
    """Append '- term' for a term of the formula (a term R would also find)."""
    try:
        terms = parse_formula(formula)
    except ValueError:
        return formula
    if len(terms.labels) < 2:
        return formula
    return f"{formula} - {terms.labels[int(rng.integers(len(terms.labels)))]}"


def _n_columns_bound(formula: str) -> int:
    """An upper bound of the number of columns, to keep the matrices small."""
    try:
        terms = parse_formula(formula)
    except ValueError:
        return 0
    sizes = {name: len(levels) + 1 for name, levels in FACTOR_LEVELS.items()}
    sizes["flag"] = 2
    total = 1
    for term in terms.terms:
        width = 1
        for name in term:
            width *= sizes.get(name, 1)
        total += width
    return total


def run_r(
    directory: Path, samples: dict[str, Any], cases: list[tuple[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the R program on the cases; return R's result for each case and more."""
    text_columns = [name for name in samples if name not in NUMERIC_COLUMNS]
    n = len(samples["a"])
    with open(directory / "samples.tsv", "w", encoding="utf-8") as handle:
        handle.write("\t".join(text_columns) + "\n")
        for row in range(n):
            values = []
            for name in text_columns:
                value = samples[name][row]
                if name == "flag":
                    value = "TRUE" if value else "FALSE"
                values.append(value)
            handle.write("\t".join(values) + "\n")
    (directory / "numeric_names.txt").write_text("\n".join(NUMERIC_COLUMNS) + "\n")
    with open(directory / "numeric.bin", "wb") as handle:
        for name in NUMERIC_COLUMNS:
            handle.write(np.asarray(samples[name], dtype="<f8").tobytes())
    for order_name, levels in LEVEL_ORDERS.items():
        rows = []
        for name, values in levels.items():
            labels = [
                ("TRUE" if v else "FALSE") if name == "flag" else v for v in values
            ]
            rows.append(SEPARATOR.join([name, *labels]))
        (directory / f"levels_{order_name}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
    files = {
        "cases.txt": [f"{order}\t{formula}" for order, formula in cases],
        "terms_cases.txt": TERMS_FORMULAS,
        "contrasts.txt": [SEPARATOR.join(CONTRAST_COLUMNS), *CONTRASTS],
        "nonlinear_contrasts.txt": NONLINEAR_CONTRASTS,
        "random_effects.txt": RANDOM_EFFECTS,
    }
    for name, content in files.items():
        (directory / name).write_text("\n".join(content) + "\n", encoding="utf-8")
    program = directory / "reference.R"
    program.write_text(R_PROGRAM, encoding="utf-8")
    rscript = os.environ.get("RSCRIPT", "Rscript")
    subprocess.run([rscript, "--vanilla", str(program), str(directory)], check=True)

    values = np.fromfile(directory / "matrices.bin", dtype="<f8")
    leverages = np.fromfile(directory / "leverage.bin", dtype="<f8")
    contrast_values = np.fromfile(directory / "estimability.bin", dtype="<f8")
    with_contrasts = {
        int(line) - 1 for line in (directory / "estimability.txt").read_text().split()
    }
    columns_read = cases_read = contrasts_read = 0
    results = []
    lines = (directory / "results.txt").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        status, width, columns, assign, labels, intercept, rank, aliased, error = (
            line.split("\t")
        )
        result: dict[str, Any] = {"status": status, "error": error}
        if status == "OK":
            p = int(width)
            block = values[columns_read * n : (columns_read + p) * n]
            result |= {
                "columns": columns.split(SEPARATOR) if p else [],
                "assign": [int(v) for v in assign.split(",")] if p else [],
                "terms": labels.split(SEPARATOR) if labels else [],
                "intercept": intercept == "1",
                "matrix": block.reshape(p, n).T,
                "rank": int(rank),
                "aliased": aliased.split(SEPARATOR) if aliased else [],
                "leverage": leverages[cases_read * n : (cases_read + 1) * n],
            }
            columns_read += p
            cases_read += 1
            if i in with_contrasts:
                weights = contrast_values[contrasts_read : contrasts_read + 3 * p]
                contrasts_read += 3 * p
                result["contrasts"] = weights.reshape(3, p)
                result["estimable"] = ESTIMABLE_BY_CONSTRUCTION
        results.append(result)
    versions = (directory / "versions.txt").read_text().splitlines()
    weights = np.fromfile(directory / "contrasts.bin", dtype="<f8")
    names = (directory / "contrast_names.txt").read_text(encoding="utf-8").strip()
    extra = {
        "terms": [
            _terms_result(line) for line in _lines(directory, "terms_results.txt")
        ],
        "random": _lines(directory, "random_results.txt"),
        "contrast_names": names.split(SEPARATOR),
        "contrast_matrix": weights.reshape(len(CONTRASTS), len(CONTRAST_COLUMNS)).T,
        "nonlinear": _lines(directory, "nonlinear_results.txt"),
        "versions": {
            "R": versions[0],
            "limma": versions[1],
            "lme4": versions[2],
            "collation": versions[3],
        },
    }
    return results, extra


def _lines(directory: Path, name: str) -> list[str]:
    return (directory / name).read_text(encoding="utf-8").splitlines()


def _terms_result(line: str) -> dict[str, Any]:
    status, intercept, labels = line.split("\t")
    if status != "OK":
        return {"error": labels}
    return {
        "intercept": intercept == "1",
        "labels": labels.split(SEPARATOR) if labels else [],
    }


def levels_for(order_name: str) -> dict[str, list[Any]] | None:
    """The explicit level order of a case; None for R's default order."""
    return LEVEL_ORDERS.get(order_name)


def compare_case(
    samples: dict[str, Any], order_name: str, formula: str, reference: dict[str, Any]
) -> tuple[str, float]:
    """Return 'identical', 'both rejected', 'rejected by design' or a difference."""
    try:
        ours = model_matrix(samples, formula, levels=levels_for(order_name))
    except ValueError as error:
        message = str(error)
        if reference["status"] == "ERROR":
            return "both rejected", 0.0
        if any(reason in message for reason in DESIGN_REJECTIONS):
            return "rejected by design", 0.0
        return f"CytePool rejects, R accepts: {message.splitlines()[0]}", np.inf
    if reference["status"] == "ERROR":
        return f"R rejects ({reference['error']}), CytePool accepts", np.inf
    if list(ours.columns) != reference["columns"]:
        return (
            f"column names differ: {list(ours.columns)} vs {reference['columns']}",
            np.inf,
        )
    if list(ours.assign) != reference["assign"]:
        return "assign differs", np.inf
    if (
        list(ours.terms) != reference["terms"]
        or ours.intercept != reference["intercept"]
    ):
        return "terms differ", np.inf
    if not np.array_equal(ours.matrix, reference["matrix"]):
        difference = np.abs(ours.matrix - reference["matrix"])
        scale = np.maximum(np.abs(reference["matrix"]), 1e-300)
        return "values differ", float(np.max(difference / scale))
    return "identical", 0.0


def compare_terms(formula: str, reference: dict[str, Any]) -> str:
    """Compare parse_formula with R's terms() on one formula."""
    try:
        ours = parse_formula(formula)
    except ValueError as error:
        if "error" in reference:
            return "both rejected"
        if any(reason in str(error) for reason in DESIGN_REJECTIONS):
            return "rejected by design"
        return f"CytePool rejects, R accepts: {str(error).splitlines()[0]}"
    if "error" in reference:
        return f"R rejects ({reference['error']}), CytePool accepts"
    if list(ours.labels) != reference["labels"]:
        return f"terms differ: {list(ours.labels)} vs {reference['labels']}"
    if ours.intercept != reference["intercept"]:
        return "intercept differs"
    return "identical"


def compare_check(
    samples: dict[str, Any], order_name: str, formula: str, reference: dict[str, Any]
) -> dict[str, Any]:
    """Compare Design.check() with R's QR: rank, aliased, leverage, estimability."""
    levels = levels_for(order_name)
    report = Design(samples, formula, levels=levels).check()
    outcome: dict[str, Any] = {
        "rank": report.rank == reference["rank"],
        # R's pivot lists the columns it never reached (once the rank equals the
        # number of samples) before the columns it moved, so compare as sets.
        "aliased": set(report.aliased) == set(reference["aliased"]),
        "leverage": float(
            np.max(np.abs(np.array(report.leverage) - reference["leverage"]), initial=0)
        ),
    }
    if "contrasts" in reference:
        named = {f"c{k}": vector for k, vector in enumerate(reference["contrasts"])}
        checked = Design(samples, formula, levels=levels, contrasts=named).check()
        flagged = {
            issue.message.split("'")[1]
            for issue in checked.errors
            if issue.kind == "contrast_not_estimable"
        }
        ours = [f"c{k}" not in flagged for k in range(len(named))]
        outcome["estimable"] = (ours, reference["estimable"])
    return outcome


def compare_random(samples: dict[str, Any], spec: str, reference: str) -> str:
    """Compare the grouping names and effect columns with lme4's lFormula."""
    if reference.startswith("ERROR"):
        return f"lme4 rejects: {reference}"
    ours = []
    for effect in Design(samples, "~ 1", random=spec).random:
        names = model_matrix(samples, "~ " + effect.effects.formula).columns
        if effect.correlated:
            ours.append(f"{effect.group}={'+'.join(names)}")
        else:  # lme4 makes one term per effect of a double-bar term
            ours.extend(f"{effect.group}={name}" for name in names)
    theirs = reference.split(SEPARATOR)
    if sorted(ours) != sorted(theirs):
        return f"differ: {sorted(ours)} vs {sorted(theirs)}"
    return "identical"


def write_fixtures(rng: np.random.Generator, random_cases: list[str]) -> None:
    """Write a small sample table with R's results for it, and limma's contrasts.

    Files: the listed formulas under R's default order (LC_COLLATE=C) and under
    explicit custom levels; R's QR results (rank, aliased columns, leverages,
    estimability) for the same cases; random formulas with at most 30 columns; and
    R's terms() of the grammar cases.
    """
    samples = simulate_samples(rng, 8)
    listed = [f for formulas in FORMULAS.values() for f in formulas]
    implicit = list(dict.fromkeys(listed + DEFAULT_ORDER_FORMULAS))
    cases = [("implicit_C", f) for f in implicit] + [("custom", f) for f in listed]
    n_listed = len(cases)
    cases += [("custom", f) for f in random_cases]
    with tempfile.TemporaryDirectory() as temporary:
        results, extra = run_r(Path(temporary), samples, cases)
    records, checks = [], []
    for (order_name, formula), reference in zip(cases, results, strict=True):
        record: dict[str, Any] = {"levels": order_name, "formula": formula}
        check: dict[str, Any] = {"levels": order_name, "formula": formula}
        if reference["status"] == "OK":
            record |= {
                "columns": reference["columns"],
                "assign": reference["assign"],
                "terms": reference["terms"],
                "intercept": reference["intercept"],
                "matrix": _compact(reference["matrix"]),
            }
            # 12 significant digits keep the files small: the tests compare
            # leverages to 1e-10, and rounding a contrast moves its cosine with
            # the null space by about 1e-12, far below the 1e-6 that decides.
            check |= {
                "rank": reference["rank"],
                "aliased": reference["aliased"],
                "leverage": [_round(h) for h in reference["leverage"]],
            }
            if "contrasts" in reference:
                check["contrasts"] = [
                    [_round(w) for w in row] for row in reference["contrasts"]
                ]
        else:
            record["error"] = reference["error"]
        records.append(record)
        checks.append(check)
    random_records = [
        record for record in records[n_listed:] if len(record.get("columns", [])) <= 30
    ][:60]
    source = f"R {extra['versions']['R']}, limma {extra['versions']['limma']}"
    common = {
        "source": source,
        "samples": {
            name: (values.tolist() if isinstance(values, np.ndarray) else values)
            for name, values in samples.items()
        },
        "level_orders": LEVEL_ORDERS,
    }
    payloads = {
        "r_model_matrix.json": common
        | {
            "cases": records[:n_listed],
            "contrasts": {
                "columns": CONTRAST_COLUMNS,
                "expressions": CONTRASTS,
                "names": extra["contrast_names"],
                "matrix": extra["contrast_matrix"].tolist(),
            },
        },
        "r_qr.json": {"source": source, "cases": checks[:n_listed]},
        "r_random_formulas.json": common | {"cases": random_records},
        "r_terms.json": {
            "source": source,
            "cases": [
                {"formula": formula} | reference
                for formula, reference in zip(
                    TERMS_FORMULAS, extra["terms"], strict=True
                )
            ],
        },
    }
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = FIXTURE_DIR / name
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", "utf-8")
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def _round(value: float) -> float | int:
    rounded = float(f"{value:.12g}")
    return int(rounded) if rounded == int(rounded) else rounded


def _compact(matrix: np.ndarray) -> list[list[float | int]]:
    """Rows of a matrix for JSON, with whole numbers written as integers."""
    return [
        [int(value) if value == int(value) else value for value in row]
        for row in np.asarray(matrix).tolist()
    ]


def print_table(rows: list[tuple[str, str, str, Any, Any]]) -> bool:
    print("| check | reference | metric | value | tolerance | result |")
    print("|---|---|---|---|---|---|")
    passed = True
    for check, reference, metric, value, tolerance in rows:
        if tolerance is None:
            print(f"| {check} | {reference} | {metric} | {value} | none | info |")
            continue
        if isinstance(tolerance, float):
            ok = bool(value <= tolerance)
            shown, limit = f"{value:.2g}", f"{tolerance:.2g}"
        else:
            ok = value == tolerance
            shown, limit = str(value), str(tolerance)
        passed &= ok
        print(
            f"| {check} | {reference} | {metric} | {shown} | {limit} | "
            f"{'pass' if ok else 'FAIL'} |"
        )
    return passed


def _fraction(found: list[str], value: str = "identical") -> tuple[str, str]:
    return f"{found.count(value)}/{len(found)}", f"{len(found)}/{len(found)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    rng = np.random.default_rng(SEED)
    samples = simulate_samples(rng, N_SAMPLES)
    random_cases = random_formulas(rng, N_RANDOM_FORMULAS)
    cases: list[tuple[str, str, str]] = []
    for category, formulas in FORMULAS.items():
        for order_name in LEVEL_ORDERS:
            cases += [(category, order_name, formula) for formula in formulas]
    cases += [("random formulas", "custom", formula) for formula in random_cases]
    cases += [("default order", "implicit_C", f) for f in DEFAULT_ORDER_FORMULAS]
    cases += [("locale order", "implicit_locale", f) for f in DEFAULT_ORDER_FORMULAS]
    cases += [("rejected by both", "sorted", f) for f in REJECTED_BY_BOTH]
    cases += [("rejected by design", "sorted", f) for f in REJECTED_BY_DESIGN]
    with tempfile.TemporaryDirectory() as temporary:
        results, extra = run_r(
            Path(temporary), samples, [(order, f) for _, order, f in cases]
        )
    versions = extra["versions"]
    print(
        f"Reference versions: {versions['R']}, limma {versions['limma']}, lme4 "
        f"{versions['lme4']}; R session collation {versions['collation']}"
    )
    print(f"CytePool side: numpy {np.__version__}")
    print(
        f"Data: {N_SAMPLES} samples; factors a (3 levels), b (2), c (3), d (4), e (4, "
        "mixed case), `cell type` (3, with spaces), flag (boolean); covariates x, z "
        "(multiples of 1/16), w (integer) and 10 badly scaled or nearly collinear "
        f"covariates; seed {SEED}"
    )
    print()
    rows: list[tuple[str, str, str, Any, Any]] = []
    outcomes: dict[str, list[str]] = {}
    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    for (category, order_name, formula), reference in zip(cases, results, strict=True):
        outcome, _ = compare_case(samples, order_name, formula, reference)
        outcomes.setdefault(category, []).append(outcome)
        if category == "locale order":
            continue
        if outcome not in ("identical", "both rejected", "rejected by design"):
            failures.append(f"{category} [{order_name}] {formula}: {outcome}")
        if outcome == "identical":
            check = compare_check(samples, order_name, formula, reference)
            check["badly scaled"] = category == "badly scaled covariates"
            checks.append(check)
            if not (check["rank"] and check["aliased"]):
                failures.append(
                    f"{category} [{order_name}] {formula}: rank or aliased columns "
                    f"differ from R's {reference['aliased']} (rank {reference['rank']})"
                )
    for category, found in outcomes.items():
        if category == "locale order":
            value = f"{found.count('identical')}/{len(found)}"
            rows.append(
                (
                    f"default level order vs R's locale collation ({len(found)} "
                    "formulas, information)",
                    "R model.matrix, character columns, session locale",
                    "identical",
                    value,
                    None,
                )
            )
        elif category.startswith("rejected"):
            formulas = (
                REJECTED_BY_BOTH
                if category == "rejected by both"
                else REJECTED_BY_DESIGN
            )
            expected = "both rejected" if category == "rejected by both" else category
            rows.append(
                (
                    f"{category}: " + ", ".join(f"`{f}`" for f in formulas),
                    "R stats::model.matrix"
                    + (" (accepts them)" if category == "rejected by design" else ""),
                    "rejected",
                    *_fraction(found, expected),
                )
            )
        elif category == "random formulas":
            by_design = found.count("rejected by design")
            both = found.count("both rejected")
            compared = len(found) - by_design - both
            rows.append(
                (
                    f"model.matrix, {len(found)} random formulas ({by_design} rejected "
                    f"by design, {both} rejected by both)",
                    "R stats::model.matrix",
                    "identical names, assign, terms, values",
                    f"{found.count('identical')}/{compared}",
                    f"{compared}/{compared}",
                )
            )
        else:
            label = (
                "model.matrix, default level order, no levels given"
                if category == "default order"
                else f"model.matrix, {category}, 3 level orders"
            )
            reference_name = (
                "R model.matrix with LC_COLLATE=C"
                if category == "default order"
                else "R stats::model.matrix"
            )
            rows.append(
                (
                    label,
                    reference_name,
                    "identical names, assign, terms, values",
                    *_fraction(found),
                )
            )
    terms_outcomes = [
        compare_terms(formula, reference)
        for formula, reference in zip(TERMS_FORMULAS, extra["terms"], strict=True)
    ]
    for formula, outcome in zip(TERMS_FORMULAS, terms_outcomes, strict=True):
        if outcome not in ("identical", "both rejected", "rejected by design"):
            failures.append(f"terms() {formula}: {outcome}")
    by_design = terms_outcomes.count("rejected by design")
    both = terms_outcomes.count("both rejected")
    compared = len(terms_outcomes) - by_design - both
    rows.append(
        (
            f"terms(), {len(TERMS_FORMULAS)} grammar cases ({by_design} rejected by "
            f"design, {both} rejected by both)",
            "R stats::terms",
            "identical intercept and term labels",
            f"{terms_outcomes.count('identical')}/{compared}",
            f"{compared}/{compared}",
        )
    )
    deficient = [check for check in checks if "estimable" in check]
    scaled = [check for check in checks if check["badly scaled"]]
    rows += [
        (
            f"rank, {len(checks)} matrices",
            "R qr() (tolerance 1e-7)",
            "identical",
            f"{sum(check['rank'] for check in checks)}/{len(checks)}",
            f"{len(checks)}/{len(checks)}",
        ),
        (
            f"aliased columns, {len(deficient)} rank-deficient matrices",
            "R qr() pivot (as lm.fit)",
            "identical",
            f"{sum(check['aliased'] for check in deficient)}/{len(deficient)}",
            f"{len(deficient)}/{len(deficient)}",
        ),
        (
            f"leverages, {len(checks) - len(scaled)} matrices",
            "R rowSums(qr.Q(qr(X))[, 1:rank]^2)",
            "max abs diff",
            max(check["leverage"] for check in checks if not check["badly scaled"]),
            1e-10,
        ),
        # A kept column that differs from the others by a fraction r of its size
        # carries rounding errors of eps / r into the basis, about 2e-9 for the
        # column 1.1e-7 away from x, in R and in CytePool alike.
        (
            f"leverages, {len(scaled)} matrices with badly scaled covariates",
            "R rowSums(qr.Q(qr(X))[, 1:rank]^2)",
            "max abs diff",
            max(check["leverage"] for check in scaled),
            1e-8,
        ),
    ]
    verdicts = [
        (mine, theirs)
        for check in deficient
        for mine, theirs in zip(*check["estimable"], strict=True)
    ]
    agree = sum(mine == theirs for mine, theirs in verdicts)
    rows.append(
        (
            f"contrast estimability, {len(verdicts)} contrasts on {len(deficient)} "
            "rank-deficient designs",
            "construction: X'u estimable; random and aliased-unit contrasts not "
            "(drawn by R)",
            "right verdict",
            f"{agree}/{len(verdicts)}",
            f"{len(verdicts)}/{len(verdicts)}",
        )
    )
    random_outcomes = [
        compare_random(samples, spec, reference)
        for spec, reference in zip(RANDOM_EFFECTS, extra["random"], strict=True)
    ]
    for spec, outcome in zip(RANDOM_EFFECTS, random_outcomes, strict=True):
        if outcome != "identical":
            failures.append(f"random effect {spec}: {outcome}")
    rows.append(
        (
            f"random-effect terms, {len(RANDOM_EFFECTS)} specifications",
            "lme4 lFormula: grouping names and effect columns",
            "identical",
            *_fraction(random_outcomes),
        )
    )
    ours = make_contrasts(CONTRASTS, CONTRAST_COLUMNS)
    reference_weights = extra["contrast_matrix"]
    scale = np.maximum(np.abs(reference_weights), 1e-300)
    relative = float(
        np.max(np.abs(ours.matrix - reference_weights) / scale, initial=0.0)
    )
    rows.append(
        (
            f"make_contrasts, {len(CONTRASTS)} linear expressions",
            "limma makeContrasts",
            "max rel diff",
            relative,
            1e-10,
        )
    )
    rows.append(
        (
            "make_contrasts names",
            "limma makeContrasts",
            "identical",
            "yes" if list(ours.names) == extra["contrast_names"] else "no",
            "yes",
        )
    )
    rejected = 0
    for expression in NONLINEAR_CONTRASTS:
        try:
            make_contrasts([expression], CONTRAST_COLUMNS)
        except ValueError:
            rejected += 1
    shown = "; ".join(
        f"{expression} -> ({weights.strip()})"
        for expression, weights in zip(
            NONLINEAR_CONTRASTS, extra["nonlinear"], strict=True
        )
    )
    rows.append(
        (
            f"non-linear expressions rejected ({shown} in limma)",
            "limma makeContrasts (deviation by design)",
            "rejected",
            f"{rejected}/{len(NONLINEAR_CONTRASTS)}",
            f"{len(NONLINEAR_CONTRASTS)}/{len(NONLINEAR_CONTRASTS)}",
        )
    )
    passed = print_table(rows)
    if failures:
        print()
        print("Differences:")
        for failure in failures[:40]:
            print("  " + failure)
    if arguments.write_fixtures:
        write_fixtures(np.random.default_rng(SEED + 1), random_cases)
    return 0 if passed and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
