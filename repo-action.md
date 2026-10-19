# Repo actions: branches, pull requests, upstream syncs, releases

This is RunAnywhere's fork of `ggml-org/llama.cpp`. Every merge, pull request and
release in this fork follows the rules below, whether a person or an agent does it.
`AGENTS.md` and `CLAUDE.md` are upstream's files. Where they disagree with this file
about how work lands in this fork, this file wins.

## Branches

| Branch | What it is | Who writes to it |
|---|---|---|
| `ra/main` | RunAnywhere's integration and release branch, and the default branch. Every fork change lands here. | Pull requests only |
| `master` | An exact mirror of `ggml-org/llama.cpp` `master`. It holds no RunAnywhere commits. | Upstream sync only |
| `<name>/<task>` | One branch per change, for example `siddhesh/pplx-v1.2`. | The author |

- Never push fork work to `master`. It stays a clean mirror, so it's always clear
  which commits are ours: `git log master..ra/main`.
- Never force-push or delete `ra/main`, or any branch a release tag was cut from.
- Branch protection isn't configured yet, so nothing enforces these rules
  automatically. Follow them by hand.

## Pull requests

- Every change reaches `ra/main` through a pull request, including one-line fixes
  and documentation.
- Open one pull request per change. Don't mix a feature, a fix and an upstream
  sync in one PR.
- Commit messages are one line: a scope prefix and the concrete change, for
  example `llama: read the pplx v1.2 readout at each marker`.
- The PR description says what changed, why, and how it was tested. Give the
  build and the real model run with its result. "Builds" alone isn't a test.
- Merge only after the change has been built on macOS (Metal) and Linux (CPU),
  and run against the decision and model fixtures it touches.
- Merge feature PRs with **squash** and upstream syncs with a **merge commit**
  (see below). Never rebase `ra/main`.
- Upstream's CI workflows run here, but many of their jobs are scoped to
  `ggml-org` or need hardware we don't have. A red upstream job doesn't block a
  merge. A failure the change itself causes does.

## Taking upstream changes

1. Fetch upstream:
   `git remote add upstream https://github.com/ggml-org/llama.cpp.git && git fetch upstream --tags`.
2. Fast-forward `master` to `upstream/master` and push it. If `master` can't
   fast-forward, something wrote to it; stop and find out what.
3. Branch `sync/<upstream-tag>` from `ra/main` and merge the upstream tag into it,
   for example `git merge b10500`. Merge a release tag (`bNNNNN`), not a moving
   `master`.
4. Resolve conflicts in favour of upstream, except in our fork changes. Our
   commits are listed by `git log master..ra/main`.
5. Open a PR from `sync/<upstream-tag>` into `ra/main` and merge it with a merge
   commit, so the upstream history stays intact.
6. Check whether any of our commits landed upstream in the meantime. If one did,
   drop our copy in the same PR.

## Releases

- Tag only a commit on `ra/main` that has been built and tested. Never tag a
  side branch.
- Tag names are `runanywhere-b<upstream build>.<n>`, for example
  `runanywhere-b10453.9`. `<upstream build>` is the upstream tag `ra/main` was
  last synced to, and `<n>` counts our releases on that base.
- Use annotated tags: `git tag -a runanywhere-b10453.9 -m "<one line: what changed>"`.
- Never move, reuse or delete a pushed tag. The SDK pins tags and SHAs, so a
  moved tag silently changes what a shipped build contains. Fix forward with
  the next number.
- After tagging, bump the SDK pin (`core/VERSIONS` `LLAMACPP_VERSION` in
  `runanywhere-sdks`) in its own PR.
- Record what each release adds in the GitHub release notes for that tag.

## Fork-specific engineering notes

`RUNANYWHERE.md` holds the model and backend notes (Maple, Prism `Q2_0`, clef and
the other decision models) and the per-release test checklist.
