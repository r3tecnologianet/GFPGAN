# How many faces Wikimedia Commons can supply, measured

The branch's position is that no face weight here can be distributed, because every published face
restorer rests on FFHQ and FFHQ is non-commercial (`LICENSE_CLEANUP.md`). The repair anyone reaches for
is to collect a corpus that is licensed for commercial use. This measures the best remaining source to
its end, so the size of that repair is a number rather than a hope.

Measured 2026-09-21. The collection tooling belongs to the sibling OpenGFPGAN project
(`docs/spikes/commons_order.py` and `docs/spikes/face_quality.py`); nothing here was modified. Data lives
outside the repository, under `/mnt/dados/gfpgan-clean/commons-order`.

## Why Commons, and not the alternatives

**PD12M was already measured and is not usable.** Its detected faces are mostly engravings, half-tone
prints and damaged scans: hand labels rate 104 of 395 crops usable, and the median detected face is
205 px. No automatic signal separates the usable ones. Sharpness scores AUC 0.446 against hand labels,
and the direction is not an accident -- hatching and paper grain are high-frequency energy, so an
engraving measures *sharper* than a photograph. Of ten hand-made features only colourfulness reaches
0.743, and a classifier on it is a colour detector that would discard the monochrome portraits a
public-domain archive is made of.

**Every Flickr source is excluded, and the exclusion cannot be repaired.** Megalith-10m and FFHQ both
draw on Flickr, so a Megalith photo can be one FFHQ was built from. The list that would remove them,
FFHQ's `ffhq-dataset-v2.json`, is itself CC BY-NC-SA 4.0: using it even only to exclude would import the
licence this fork exists to avoid. Megalith's high-resolution links are also Flickr's `_b` size, about
1,024 px on the long side, so a 512 px face would have to fill the frame.

Commons is what remains: per-file licence and author in the API, originals rather than thumbnails, and
Flickr imports identifiable by their own metadata with no FFHQ artefact involved.

## Method

Candidates are files with structured "depicts: human" and "photograph" statements, minus files marked as imported from
Flickr. Admitted licences were CC0, public domain and CC BY in any version, with CC BY-SA excluded because whether
share-alike reaches trained weights is unsettled. That exclusion was lifted later the same day, for the reason
measured below; the figures in the next two sections are the collection made under it. Each file must be at least 512
px on its short side, and is downloaded bounded to a 2,048 px box.

## Result: the pool is 7,433 files and yields 876 crops

| Stage | Count |
|---|---|
| Candidate titles matching the query | 42,971 |
| Admitted after licence and Flickr exclusion | **7,433** |
| Attempted (the whole pool) | 7,433 |
| Downloaded | 6,621 (812 failed) |
| Images yielding at least one face ≥512 px | 800 |
| **Face crops ≥512 px** | **876** |

That is 117.9 crops per 1,000 images, five times PD12M's 23.7, and the faces are large where PD12M's were
not: the 25th, 50th, 75th and 90th percentiles of the shorter side are 563, 634, 734 and 866 px, against
a median of 205 px for PD12M. As a source, Commons is qualitatively better in exactly the way the licence
and metadata filters predict.

**But most of it had already been collected.** Against the earlier `commons-faces` harvest, 501 of the 800
images overlap. The genuinely new material is **299 images and 327 crops**.

## What this settles

The whole admissible Commons pool -- every file that is CC0, public domain or CC BY, not imported from
Flickr, tagged as a photograph of a human, and at least 512 px -- produces 876 face crops, 327 of them new.
FFHQ holds 70,000 curated faces. The gap is not one more crawl: it is two orders of magnitude, and this
pool is exhausted rather than sampled -- exhausted under the licence filter of the day, which is what the
decision below goes on to change.

Training a StyleGAN2 prior at this scale is the decision that follows, not "collect more". For reference,
the measured throughput on the RTX 3060 Ti is 28.2 img/s at 512² with batch 4 -- and only batch 4 fits in
8 GB, against the paper's minibatch of 32, which needs gradient accumulation that `ogan/training.py` does
not implement. Adaptive augmentation, which the sibling project has implemented, is the technique that
exists for corpora this small.

## The licence filter manufactures a demographic skew

Inspection of the crops showed most of the people in them to be Black, which is not what a corpus drawn
from Wikimedia Commons at large would look like. The cause is measurable, and it is the licence filter.

Counting the candidate titles that fall into a handful of upload clusters -- francophone Wikimedia
outreach events, Wikimedia user groups, Côte d'Ivoire, Nigeria, an Iranian cluster -- the concentration
rises at every stage of the funnel:

| Population | In those clusters |
|---|---|
| All 42,971 candidates | 9.4% |
| The 7,433 admitted by licence, non-Flickr and size | 30.8% |
| The 800 that yielded a face crop | 44.0% |

Split by licence over the same query and the same describe step, so the populations are matched. These rows
count the licence gate alone, before the Flickr and size exclusions, which is why the admitted row here is
8,861 rather than the 7,433 that survive all three. The three families leave out 39 files, 0.09%, licensed
GFDL, GODL-India, OGL, "No restrictions" or bare Attribution, so the rows do not sum to 42,971:

| Licence | Files | In those clusters |
|---|---|---|
| CC BY-SA (excluded) | 34,071 | **5.2%** |
| CC0 and public domain | 5,058 | 30.7% |
| CC BY | 3,803 | 19.5% |
| **Admitted by licence alone (CC0/PD + CC BY)** | 8,861 | **25.9%** |

**How to reproduce the two tables above.** The keyword list that defines "those clusters" was not recorded when
the figures were produced. Rebuilt from the five clusters named above it reproduces all seven of them to within
0.9 points: match a candidate title, case-insensitively, against `wikick`, `wikiconvfr`, `wikican`, `wikimousso`,
`éditathon`, `editathon`, `edit-a-thon`, `matinée contributive`, `renforcement de capacité`, `journée d'étude`,
`ran24`, `ran25`, `ran2024`, `ran2025`, `usergroup`, `user group`, `wikimedians`, `club wikimedia`, `giehpci`,
`wikiciv`, `cote d'ivoire`, `côte d'ivoire`, `abidjan`, `femua`, `daloa`, `san-pedro`, `nigeria`, `in iran`,
`meraji`, `wikimedia`, `wikipédia`. That gives 9.2%, 31.4% and 44.1% down the funnel against the 9.4%, 30.8% and
44.0% published, and 4.7%, 31.6%, 19.3% and 26.3% across the licence rows against 5.2%, 30.7%, 19.5% and 25.9%.
The exact figures depend on the list; the shape does not, and no variant tried moved the funnel below 8.5% to
28.7% to 42.8%.

The two tables also check each other, which needs no reconstruction. The admitted row is the sum of its parts:
30.7% of 5,058 plus 19.5% of 3,803 is 2,294 titles, against 2,295 from 25.9% of 8,861. And the three licence
rows recover the funnel's first row: 5.2% of 34,071 plus those 2,294 is 9.46% of 42,971, published as 9.4%.

The breadth is in the share-alike population that this project excludes. CC BY-SA is the default licence
for most Commons uploads, so what survives a CC0/PD/CC BY filter is disproportionately organised outreach
photography from particular communities, and the face-crop step concentrates it further because event
photographs are close-range group portraits that yield large faces.

This is a cost of the licence decision that was not visible before the corpus existed. Relaxing CC BY-SA would buy
roughly 3,950 further crops at the rate measured on the admitted pool, and -- the part that matters for a corpus
collected to increase diversity -- a population five times less concentrated. Whether share-alike reaches trained
weights remains unsettled, so this is a trade to decide rather than a repair to apply.

**Decided on 2026-09-21 by the project owner: CC BY-SA is admitted, and collection extended to it.** The
order was built from the metadata already cached for all 42,971 described candidates, so no new API pass
was needed, and it applies the same gates as the first collection -- JPEG or PNG, not imported from
Flickr, thumbnail at least 512 px on the short side -- differing only in the licence predicate. It holds
33,564 files.

**The rate does not transfer, and the whole order is now collected.** CC BY-SA yields 79.3 crops per 1,000 against
117.9 for the admitted pool: all 33,564 files downloaded and 2,662 crops came out, against the 3,950 first estimated
and the 2,560 estimated once the real rate was visible. The mechanism is the one that makes this population worth
collecting. The outreach clusters the licence filter had concentrated are close-range group portraits, which is
exactly what yields a face of 512 px, so breadth and yield move against each other here and the first estimate was
built on a rate the skew measurement itself should have warned against.

Commons has now given 3,538 crops in total, 876 under the original licence filter and 2,662 under the widened one.

### Two fifths of the crops are the same people

The reviewer opened the sheet and reported significant duplication. Pixel-level duplication is not what it is: a
difference hash over all 2,662 crops finds 3 identical pairs and 10 near-duplicates within a Hamming distance of 4,
and multiple faces cropped from one photograph account for only 226 crops, since 2,662 crops come from 2,436 distinct
images. Measuring that first answered the wrong question.

The duplication is by event series. Grouping source titles by their series name -- the file title with its
extension and any trailing sequence number removed, where the number may carry punctuation on either side, as
`Title (4).jpg` and `Title - 04.jpg` do -- gives 1,388 series, and the 69 series holding five or more
photographs account for 1,079 crops, 40.5% of the total. The largest are conference and outreach sets: 93
photographs and 126 crops from one Iberoconf 2023 session, 44 and 69 from a single lecture, 51 and 59 from one
editathon opening. These are the same speakers and attendees photographed dozens of times, which inflates the
count without adding identities and biases a prior toward those faces.

The clusters also moved with the licence. The CC0 and CC BY pool concentrated in francophone African outreach events;
the CC BY-SA pool concentrates in Ibero-American ones. Different communities, the same structure: Wikimedia event
photography is what survives any licence filter narrow enough to be usable.

**Capped at three crops per series, chosen from distinct photographs.** The cap takes the corpus from 2,662 crops to
1,757, drawn from 1,704 separate images, and touches only the 97 series that exceed it. Three crops of the same
photograph would be one person by construction, so the selection cycles through a series' images before taking a
second crop from any of them. A cap of one was rejected as too aggressive: a 93-photograph conference plausibly holds
dozens of distinct people.

Series is a proxy for identity, not a measurement of it. Three photographs from one series can still be the same
speaker, and photographs from different series can be the same person. Measuring identity would need a face
recognition model, and this fork has none it may use -- the same constraint that removed ArcFace. The grouping also
relies on titles being numbered in sequence, which is the Wikimedia event convention but not a guarantee.

What this costs is worth stating plainly next to what it buys. Share-alike is a condition on derivatives,
and whether a trained weight is a derivative of its training images has no settled answer; this fork
exists to avoid exactly that kind of open question, and admitting CC BY-SA accepts one deliberately in
exchange for a corpus roughly five times larger and materially less concentrated. Every row keeps its
`Artist`, `Credit` and licence URL, so the attribution the licence requires can be produced. The decision
does not touch the code or the background weights, only the face corpus.

## What this does not settle

**The subject's rights.** CC0 section 4(c) has the affirmer disclaim responsibility for clearing the
rights of other persons, and CC BY 4.0 section 2(b) does not license personality rights. A licence clears
the photographer. Neither this repository nor the sibling project documents a position on the depicted
person, and for weights meant to be distributed that is the gate that matters, not the crop count.

**The usable fraction is 81%, and it is not the bottleneck.** The 876 crops were reviewed by hand, and
709 of the 874 reviewed crops are usable, against PD12M's 44% (872 carry a verdict; two were left
blank). Applying that rate to the 327 new crops leaves about 265 usable faces that were not already held.

The count needs its tie-breaking rule stated, because thirty crops were repeated inside the sheet to
measure the reviewer and three of them carry labels that disagree. Counting a crop usable only when every
one of its labels says so gives 709; taking the first label gives 711, the last 710, and any label 712.
The unanimous rule is used here and for the corpus built from it, because for training data the
conservative reading is the right one, and the spread of three crops in 874 is the reviewer's own
consistency rather than an error.

Colour is no longer a confound either -- taking each crop's first label, 619 are colour against 10 monochrome -- so
the classifier that failed on PD12M by being a colour detector would have had almost nothing to detect.

### The corpus that came out of it

The usable crops were aligned into a training set with `build_dataset.py`. Each step drops faces for a
different reason, and one of them is a trap worth naming:

| Stage | CC0 / CC BY | CC BY-SA | Combined |
|---|---|---|---|
| Crops collected | 876 | 2,662 | 3,538 |
| Offered for review | 874 | 1,757 (capped by series) | — |
| Usable, unanimous rule | 709 (81.1%) | 815 (46.5%) | **1,524** |
| Kept after pruning those without component boxes | — | — | 1,484 |
| Aligned to 512x512 | — | — | 1,478 (6 failed) |
| With facial component boxes | — | — | 1,478 of 1,478 |
| Split of the aligned, seed 0 | — | — | **1,414 training, 64 validation** |

The two pools review very differently: 81.1% of the CC0 and CC BY crops are usable against 46.5% of the CC BY-SA ones,
and within CC BY-SA the rate falls with series size -- 52.5% for photographs belonging to no series, then 37.6%, 30.1%
and 30.5% for series of 2-4, 5-9 and 10 or more. The reviewer was rejecting repeated people as well as poor
photographs, and event photography is worse on both counts at once.

**The prune is not optional.** A first build aligned all 1,524 usable crops and then found component boxes for only
1,478 of them: 40 faces are unreadable, not square, or show no face on the second detection pass. Boxes are a lookup
rather than a filter, and `FFHQDegradationDataset` reads them with a bare subscript, `self.component_boxes[name]`,
guarded only by `crop_components`. Since `options/train_gfpgan_clean.yml` sets `crop_components: true`, those 40 faces
would have raised a `KeyError` inside a dataloader worker at a random iteration rather than at startup. Pruning them
before aligning costs 2.6% of an already small corpus and makes the shipped config work as written. The six alignment
failures recur in both builds, since they fail alignment rather than box generation.

The prune also has to map an aligned filename back to the crop it came from, and guessing that mapping is how the
first attempt failed: `build_dataset.py` names an aligned face after the *parent* directory of the crop folder, not
the folder itself, so a prefix assembled by hand matched 11 of the 40 and pruned nothing. `manifest.jsonl` records
`name` beside `source_file` for every face, which is the authoritative mapping and was there the whole time.

It lives at `/mnt/dados/gfpgan-clean/faces-commons-final`, outside the repository, in the same shape as the
other corpora: `aligned512/`, `train_gt/`, `val_gt/`, `val_lq/`, `component_boxes.pth` and a
`manifest.jsonl` recording where each face came from. Every face in both splits has a box, which was
verified rather than assumed.

Ready is not the same as sufficient. 1,414 training faces is 2% of FFHQ. Admitting CC BY-SA a little more than doubled
the corpus, from 619 training faces to 1,414, and that is the ceiling: both licence pools are exhausted rather than
sampled. What this supports is a stability or ablation run, not a prior worth distributing.

`face_quality.py report` cannot read this run: it expects the `eligible` and `total` keys that its own
`order` subcommand writes, and a Commons order is built by `commons_order.py`, which records `candidates`
and `kept` instead. The figures above were computed from `labels.jsonl` directly rather than by changing
the sibling project's tooling.

**812 downloads failed in the first collection, and the CC BY-SA run identified the cause.** Two hypotheses had been
tested and refuted: concurrency throttling, since a retry with a single worker downloaded 15 of 429, and transfer
size, since the served bytes are identical in both groups and the failed group's originals are smaller rather than
larger. A sampled HEAD probe returned 200 for 30 of 30, which proved only that the URLs resolve, and reading it as
evidence of a transient fault was a mistake: HEAD transfers no body.

The CC BY-SA collection made the mechanism visible by being large enough to show its shape. Download success was 100%
for the first 22,000 images and then collapsed: 36% in the block ending at 24,000 and 0% for every block after it.
That is a cliff in time, not a property of the images, and it burned through the remaining 10,847 rows at 7.5 images a
second because a request that fails transfers nothing. Re-running exactly those 10,847 rows hours later, at two
workers instead of four, downloaded 100% of them and yielded 78.7 crops per 1,000, the same rate as the rest of the
run.

So the cause is sustained request pressure on a server that renders thumbnails on demand, not the URLs and not the
file sizes. The single-worker retry that seemed to refute throttling had been run inside the same blocked window,
which is why it recovered nothing. `download()` in `corpus_yield.py` catches `URLError`, `TimeoutError` and `OSError`
alike and returns `None` without recording a reason, which is what made a visible block look like scattered random
loss for two collections in a row.
