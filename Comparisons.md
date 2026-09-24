# Comparisons

Upstream's version comparison used to be here: tables of V1, V1.2 and V1.3 outputs over 28 photographs of
named public figures, hotlinked from the upstream repository.

Both halves of it are out of this fork. The models are the official release weights, trained on FFHQ, which
this repository can neither ship nor recommend (`LICENSE_CLEANUP.md`). The photographs are pictures of
identifiable people with no documented licence, which is why `inputs/` was removed as well — and showing a
named person beside a manipulated version of their own face is precisely the personality-rights problem this
fork argues no copyright licence solves.

What is measured here instead, on held-out faces rather than on public figures:

- [docs/training_stability.md](docs/training_stability.md) — the screening rounds and runs behind
  `options/train_gfpgan_clean.yml`, the comparison against the official GFPGAN v1.4, the `mild_prob` result
  and the `-w/--weight` sweep.
- [docs/background_super_resolution.md](docs/background_super_resolution.md) — the background model that
  `--bg_model` takes, and what was refuted along the way.
- [docs/face_restoration_alternatives.md](docs/face_restoration_alternatives.md) — CodeFormer, GPEN, VQFR and
  CFRNet assessed against this fork.
