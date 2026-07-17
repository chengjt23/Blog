# Implementation Status

> State: `github_pages_bridge_diffusion_v1_live`
> Date: 2026-07-17

## Completed

- Astro 7 / TypeScript strict / Node 24 / npm project and lockfile.
- Four-environment publication and indexing policy.
- Conservative allowlist/denylist and explicit dry-run/write sync modes.
- Markdown AST transformation for 15 Bridge chapters.
- Schrödinger Bridge B0-B14 is published as formal release v1.0 with `publishedAt` and `updatedAt` set to 2026-07-17.
- Markdown AST transformation for 14 Diffusion chapters, including public math-delimiter normalization and conservative research-link removal.
- Diffusion D0-D13 is published as formal release v1.0 with `publishedAt` and `updatedAt` set to 2026-07-17.
- Relative source/output SHA-256 manifest with 97 generated entries across both Blogs.
- 68 original figures with stable dimensions, alt text, captions, and automatic PNG dimension detection.
- Internal status removal and source-note official URL mapping.
- Core KaTeX conversion in B0, B2, B3, B5, B6, B9, and B12.
- Mermaid rendering with no-JavaScript text fallback in B11 and B13.
- Responsive three-column reading layout, mobile Blog navigation, system-driven light/dark themes, and code copy tools.
- Header uses only the `R` letter mark on the left; navigation contains only Home and Blog on desktop and mobile.
- Home is intentionally visually blank; Blog, Bridge, Diffusion, Privacy, License, Search, RSS, robots, and 404 routes are implemented.
- Blog titles and representative images are direct links; no separate entry button is required.
- Legacy `/series/*` and `/about/` routes redirect to the current Blog and Home routes.
- GitHub Pages `/Blog/` base-path support covers navigation, content links, images, RSS, Pagefind, manifest, and redirects.
- GitHub Actions Pages workflow builds a noindex public beta and runs static plus Chrome checks before deployment.
- Pagefind indexes only approved public content; privacy/search/404/blank Home are excluded.
- Default OG metadata, favicon, web manifest, JSON-LD, and security headers.

## Verification Evidence

- `astro check`: 0 errors, 0 warnings, 0 hints.
- ESLint and Prettier: pass.
- Vitest: 7 tests pass.
- Static build: 37 content pages plus 33 compatibility redirects generated.
- Pagefind: 33 approved content pages indexed.
- Generated HTML audit: 37 content pages and 33 redirects pass links, redirect targets, H1, images, robots, and leakage checks.
- Playwright: 6 tests pass for empty Home, both direct Blog entries, Diffusion math/images/navigation, desktop/mobile navigation, system dark mode, Mermaid, and Pagefind.
- GitHub Pages public-beta build and all 6 Playwright tests pass under `/Blog/`.
- GitHub Actions build and deploy jobs pass for Bridge v1.0 commit `36e089a` in run `29560298779`.
- GitHub Actions build and deploy jobs pass for Diffusion v1.0 commit `b6c8e5b` in run `29564516973`.
- Online smoke passes for Home, Blog, both direct topic-title entries, all 29 chapters, formal v1.0 metadata, images, KaTeX, Pagefind, RSS, mobile navigation, and zero horizontal overflow.
- Screenshots: `artifacts/qa/`.

## Required User Gate

Bridge v1.0 and Diffusion v1.0 are live at `https://chengjt23.github.io/Blog/`. The repository is public, `main` deploys
through GitHub Actions, and indexing remains disabled. No custom domain or production indexing action has
been performed.

The remaining release gate is a separate explicit decision to enable search-engine indexing. Sitemap
submission and any custom domain change also require separate approval.

## Deferred P1/P2

- Convert the remaining non-core equation fences to KaTeX.
- Per-article generated OG images.
- Projects, Notes, Tags, and Archive until real approved content exists.
- Giscus, analytics, newsletter, and i18n.
- Lighthouse thresholds and Mermaid bundle optimization.
