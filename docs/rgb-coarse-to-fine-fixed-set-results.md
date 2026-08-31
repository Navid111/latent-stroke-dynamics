# Fixed RGB coarse-to-fine result

## Status

The one fixed five-target qualitative execution completed successfully on
2026-08-31. This is a bounded exact-pixel engineering result, not a learned
model experiment. It does not reopen or alter the archived Phase B0
`not_eligible` decision.

- Implementation commit: `2c50e88e4c499f30433eb6884b51d147fe23bfa5`
- Branch: `phase-b/saliency-latent`
- Runner status: `rgb_coarse_to_fine_fixed_set_complete`
- Target-set SHA-256: `31e1fcc2bf344f8b72d3f04dfbc9109c61c39fc8cbc10668c1b78d575a673b42`
- Aggregate-summary SHA-256: `7f2c32cef077bef1737a3f00ee584cf1075b1feb5711995ab87dd9812f233c05`
- Runtime: 94.117691846 seconds on a standard Google Colab CPU runtime
- Completed targets: 5 of 5
- Recorded artifacts verified by the post-run audit: 94
- Training performed: no
- Learned model used: no
- Source images committed: no
- Full raw output: preserved privately in Google Drive

## Fixed configuration

- Planning canvas: 96 x 96 RGB
- Replay canvas: 512 x 512 RGB
- Initial canvas: white
- Candidate pool: 64 unique changing rendered outcomes
- Proposal mixture: 80% error-guided, 20% uniform
- Candidate score: exact rendered target-pixel RGB MSE
- Stroke color: target-fitted mean RGB under the sampled line mask
- Seed: 73
- Minimum accepted improvement: `1e-9`
- Stage patience: 12 non-improving fresh pools
- Replay supersampling: 2x
- Global stage: 40 strokes, length 0.25-0.75, width 0.10-0.22
- Structure stage: 70 strokes, length 0.08-0.35, width 0.035-0.10
- Detail stage: 100 strokes, length 0.02-0.15, width 0.012-0.045
- Maximum trajectory: 210 accepted strokes

## Integrity outcome

All frozen acceptance checks passed:

- all five targets completed;
- every executed stroke strictly improved the exact 96 x 96 objective;
- every best frame was no worse than its final frame;
- all frozen decisions were preserved;
- the post-run audit verified every one of the 94 hashes recorded in the
  aggregate summary.

All targets used the complete 210-stroke budget. No stage reached the patience
limit. Best step was 210 and best equaled final for every target, so the exact
selector did not exhibit the overpainting observed in the earlier latent
qualitative trajectory.

## Per-target results

| Target | Initial MSE | Final/best MSE | Reduction from white | Final MAE | 512 replay MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple symbol | 0.0262094540 | 0.0027757169 | 89.4% | 0.0126663773 | 0.0100005242 |
| Geometric composition | 0.1933822311 | 0.0135598688 | 93.0% | 0.0666263844 | 0.0285548006 |
| Isolated object | 0.0976485707 | 0.0016771739 | 98.3% | 0.0236074233 | 0.0034593858 |
| Landscape | 0.2470302822 | 0.0041934281 | 98.3% | 0.0425327365 | 0.0085964505 |
| Dense scene | 0.5483736079 | 0.0053596996 | 99.0% | 0.0549998298 | 0.0109840269 |

Absolute MSE is not a perceptual ranking across these unrelated images because
white padding, target contrast, color distribution, and high-frequency detail
differ. The simple symbol, for example, begins with low error because much of
the target is white.

## Stage-end planning MSE

| Target | Global (40) | Structure (+70) | Detail (+100) |
| --- | ---: | ---: | ---: |
| Simple symbol | 0.0166308932 | 0.0085854015 | 0.0027757169 |
| Geometric composition | 0.0390491043 | 0.0222825778 | 0.0135598688 |
| Isolated object | 0.0092882345 | 0.0035155310 | 0.0016771739 |
| Landscape | 0.0172993822 | 0.0075994131 | 0.0041934281 |
| Dense scene | 0.0121506546 | 0.0075972165 | 0.0053596996 |

Each stage reduced exact planning error for every target.

## Qualitative reading

1. **Simple symbol:** the circle and three-pointed star remain recognizable,
   while the ring is jagged and surrounded by fragmented gray marks.
2. **Geometric composition:** the broad grid, dominant color blocks, and dark
   circles are retained, but crisp boundaries and precise geometry become
   faceted overlapping marks.
3. **Isolated object:** the vase is the clearest object reconstruction; its
   silhouette, narrow neck, white upper body, and deep-blue base are retained.
4. **Landscape:** the result preserves pale sky, distant blue-gray mountains,
   and layered green hills, while clouds and horizon contours are simplified.
5. **Dense scene:** the dark palette, vertical mass, and bright central accents
   are retained, but fine architectural identity collapses into an abstract
   blocky structure.

## Interpretation

The result demonstrates that the renderer and proposal mechanism can produce
recognizable RGB approximations when candidate selection is exactly aligned
with the pixel evaluation objective. The contrast with the earlier latent
qualitative run, which was best at step 50 and then overpainted, strengthens
the diagnosis that learned top-of-list ranking and stopping remain major
long-horizon bottlenecks.

Selection is not the only limitation. The dense scene and curved symbol show
that straight opaque line primitives and 96 x 96 planning cannot preserve fine
architecture, smooth curves, or crisp graphic boundaries. Supersampled
512-pixel replay smooths presentation but cannot restore detail absent from the
planning decisions. Its MSE is evaluated against a separately resized
512 x 512 target and therefore should not be interpreted as the planning loss.

## Scope and publication boundary

The five web-sourced targets were fixed for private evaluation before outputs
were viewed. Their inclusion in this private run does not establish permission
to reproduce them in a submitted thesis or public repository. The source
images and generated binary outputs remain uncommitted. Metrics and method
conclusions may be reported, while any final reproduced image requires a
verified licence, institutional fair-use basis, or clearly disclosed
presentation-only substitution with an original or reusable source.

No post-result target replacement, configuration tuning, or rerun is authorized.
