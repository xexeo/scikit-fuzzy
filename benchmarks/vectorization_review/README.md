# Evidence for the vectorization pull requests

These files retain the deterministic workloads, differential comparisons, raw timing samples, and independently tested source hashes for the individual scikit-fuzzy proposals. They are published on a separate evidence branch so each upstream PR contains only its own production code and tests.

Each result file identifies its exact reference and candidate commits. Most proposals compare directly with upstream be7e344. The alpha arithmetic follow-up instead compares with the existing PR 357 head, 7777a10, so it does not claim the earlier arithmetic gains again. FIRE filtering and shared interpolation are excluded from the submissions.

The full package suite passed on Python 3.11.9 with both NumPy 2.4.6 and SciPy 1.17.1, and NumPy 1.26.4 and SciPy 1.11.4. Those local checks do not replace the upstream supported-version CI matrix. Exact floating point equality is not claimed universally: the bisector fixes a mathematical defect, FPC uses a different reduction order, and batched mean of maxima has small rounding differences.

## Reproduce a proposal

Create separate checkouts for the reference and candidate commits named in the result file. Install the pinned scientific dependencies in an isolated environment. The worker imports the explicitly supplied checkout and verifies its source location, avoiding accidental tests of an installed package.

The workload definitions are in `benchmark.py`. To check a chosen family without collecting timings, run:

```powershell
python worker.py C:/path/to/reference --verify reference.pkl --prefixes '["centroid/", "bisector/"]'
python worker.py C:/path/to/proposal --verify proposal.pkl --prefixes '["centroid/", "bisector/"]'
```

For timing, run `worker.py C:/path/to/checkout` and send a JSON string containing a workload ID on each line of standard input. It returns per-call milliseconds from a calibrated group of calls; the first response also records separately measured traced peak allocation. Workload IDs and all seven reference and candidate samples are retained in each result JSON. The comparison driver alternated the two persistent workers and reversed their order on alternate samples. Inputs and controller construction are prepared outside timed calls; assigning array inputs and computing controller outputs are inside timed calls.

Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONHASHSEED=0`, and `MPLBACKEND=Agg` for the recorded configuration. Measurements came from an Intel Core i7 3770S Windows desktop. Ratios near one should be interpreted conservatively. Traced allocation is not total process resident memory.

## Attribution

This work was produced with ChatGPT under the supervision of Geraldo Xexéo. The model attribution supplied by the supervising author is GPT-6.1 Sol High. The original arithmetic proposal in PR 357 was prepared with Claude Code; the ChatGPT attribution applies to the follow-up optimization and review.
