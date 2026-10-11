# Functional-failure source audit (pre-experiment)

## First source checked

Source: `/home/mao/D2Pred/良信验收报告.docx`, 产品电气寿命预测研究总结报告.

The report does contain a usable operational definition, although it does not use
the English phrase “functional failure”. It states that the project studies the
electrical life of the NDC1-3810 AC contactor and that the failure is determined
from the end-of-life voltage/current curve. It identifies the observed failure
forms as `熔焊` (welding) and `炸开` (explosive opening) and describes obtaining
the life information by judging the failure moment from the end-of-product
voltage/current waveform. It also distinguishes interrupted experiments, empty
files, missing data, and low-precision test-bench data from the failure event.

## Operational interpretation for the final protocol

For the final rerun, a trajectory is eligible as a functional-failure trajectory
only when the original record identifies its endpoint as an electrical failure of
the contactor, with the failure form recorded as welding or explosive opening (or
an equivalent source-record label). The endpoint is the source-record failure
moment/cycle inferred from the end-of-life voltage/current record, not a DI
threshold, model output, or arbitrary last row.

The report's record is sufficient to define the concept at protocol level, but the
raw-record audit must still map each anonymised unit to its endpoint and failure
form before the final manifest is frozen.

## Current DI match

The current anonymised DI table is `data/DI/DI_df.csv`, with 38 unit identifiers
and columns `unit`, `time_cycles`, `RUL`, `RUL_Percent`, and `HI`. Its unit lengths
are retained as a matching fingerprint. The report's tables describe original
device numbers, batches, lifetime values, missing files, interruptions, and
failure forms; these fields must be matched to the anonymised unit inventory.

## Chinese final-paper cross-check

The relevant file was found at
`/home/mao/D2Pred/PythonProject2/chinese_paper.pdf` (SHA-256
`23ccabf4642d0d750c24ad469f189af8cee6a375ec905c67faada5c06ff7fed1`).
It is the AC-contactor RUL thesis, not the unrelated Sun Lele anomaly-detection
thesis. Section 6.2.2 reports 183 monitored contactors, of which 33 have complete
trajectories from initial normal operation to functional failure. It states that
each close-open operation is one time unit, testing continues until failure, and
the total number of operations is the true lifetime used for RUL supervision.

This independently agrees with the final cohort size after excluding five units
from the 38-row inventory. In `data/DI/DI_df.csv`, all 38 units satisfy the exact
endpoint identity: cycles start at 1, initial RUL equals `N-1`, final RUL equals
0, and the final cycle equals `N`. Thus the stored endpoint is the final observed
operation count, not a learned DI threshold.

The thesis does not provide a per-anonymised-unit table linking each unit to a
specific failure form. Therefore it validates the endpoint convention and 33-unit
complete-cohort statement, while the acceptance report supplies the engineering
failure forms (welding/explosive opening).

## Final endpoint rule

Neither the Chinese final paper nor the acceptance report contains a per-anonymised-
unit endpoint table. They provide the experiment-level convention: testing runs
until failure, total close-open operations define lifetime, and the failure moment
is judged from the end-of-life voltage/current waveform. Under the approved
fallback rule, the endpoint in the anonymised analysis table is therefore the row
where `RUL == 0`. For every unit in `data/DI/DI_df.csv`, this is exactly the final
observed cycle. Formally, for unit `u`, `T_u = max(time_cycles | RUL = 0)` and the
label at cycle `t` is `RUL_u(t) = T_u - t`. No DI threshold is involved.

## Raw archive status

`/home/mao/D2Pred/raw.rar` exists (about 6.5 GB; SHA-256
`735748afcd9dede1c633450d9036a2089e097a35fe4f670693cb12f2f214f2be`). A
read-only archive inventory was obtained with `bsdtar`: 12,403 entries belonging
to unrelated benchmark collections (BDC, CWRU, HAR, KNPHM, NEU, PenDigits,
SpokenArabicDigits, SEU Gearset, SteelPlatesFaults, UWave, and CIFAR). It contains
no identifiable AC-contactor/NDC1 lifetime experiment tree and cannot be the raw
source of `data/DI/DI_df.csv`. Full extraction was therefore intentionally not
performed.
