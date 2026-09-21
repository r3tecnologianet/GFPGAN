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
licence this branch exists to avoid. Megalith's high-resolution links are also Flickr's `_b` size, about
1,024 px on the long side, so a 512 px face would have to fill the frame.

Commons is what remains: per-file licence and author in the API, originals rather than thumbnails, and
Flickr imports identifiable by their own metadata with no FFHQ artefact involved.

## Method

Candidates are files with structured "depicts: human" and "photograph" statements, minus files marked as
imported from Flickr. Admitted licences are CC0, public domain and CC BY in any version; CC BY-SA is
excluded because whether share-alike reaches trained weights is unsettled. Each file must be at least
512 px on its short side, and is downloaded bounded to a 2,048 px box.

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
FFHQ holds 70,000 curated faces. The gap is not one more crawl: it is two orders of magnitude, and the
pool is now exhausted rather than sampled.

Training a StyleGAN2 prior at this scale is the decision that follows, not "collect more". For reference,
the measured throughput on the RTX 3060 Ti is 28.2 img/s at 512² with batch 4 -- and only batch 4 fits in
8 GB, against the paper's minibatch of 32, which needs gradient accumulation that `ogan/training.py` does
not implement. Adaptive augmentation, which the sibling project has implemented, is the technique that
exists for corpora this small.

## What this does not settle

**The subject's rights.** CC0 section 4(c) has the affirmer disclaim responsibility for clearing the
rights of other persons, and CC BY 4.0 section 2(b) does not license personality rights. A licence clears
the photographer. Neither this repository nor the sibling project documents a position on the depicted
person, and for weights meant to be distributed that is the gate that matters, not the crop count.

**The usable fraction.** The 876 crops are unreviewed. PD12M's hand review kept 44%, and no automatic
filter works, so this number needs a person. `labels.html` is generated and waiting in the data directory.

**812 downloads failed, and the cause is unknown.** Two hypotheses were tested and both were refuted.
Concurrency throttling: a retry with a single worker and no concurrency downloaded 15 of 429. Transfer
size: the served size is identical in both groups (median 2.80 Mpx), and the failed group's *originals*
are smaller, not larger (median 7.99 against 12.98 Mpx), which is the opposite of a timeout on large
files. A sampled HEAD probe returned 200 for 30 of 30, which proves only that the URLs resolve -- HEAD
transfers no body, so it could not distinguish the hypotheses, and reading it as evidence of a transient
fault was a mistake. `download()` in `corpus_yield.py` catches `URLError`, `TimeoutError` and `OSError`
alike and returns `None` without recording a reason, so the information was discarded at the point of
failure. The failures are worth about 96 crops, which does not change any conclusion above.
