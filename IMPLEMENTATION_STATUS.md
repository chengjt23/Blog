# Implementation Status

> State: `github_pages_public_beta_live`
> Date: 2026-07-15

## Completed

- Astro 7 / TypeScript strict / Node 24 / npm project and lockfile.
- Four-environment publication and indexing policy.
- Conservative allowlist/denylist and explicit dry-run/write sync modes.
- Markdown AST transformation for 15 Bridge chapters.
- Relative source/output SHA-256 manifest with 27 generated entries.
- 12 original figures with stable dimensions, alt text, and captions.
- Internal status removal and source-note official URL mapping.
- Core KaTeX conversion in B0, B2, B3, B5, B6, B9, and B12.
- Mermaid rendering with no-JavaScript text fallback in B11 and B13.
- Responsive three-column reading layout, mobile Blog navigation, system-driven light/dark themes, and code copy tools.
- Header uses only the `R` letter mark on the left; navigation contains only Home and Blog on desktop and mobile.
- Home is intentionally visually blank; Blog, Bridge, Privacy, License, Search, RSS, robots, and 404 routes are implemented.
- Blog titles and representative images are direct links; no separate entry button is required.
- Legacy `/series/*` and `/about/` routes redirect to the current Blog and Home routes.
- GitHub Pages `/Blog/` base-path support covers navigation, content links, images, RSS, Pagefind, manifest, and redirects.
- GitHub Actions Pages workflow builds a noindex public beta and runs static plus Chrome checks before deployment.
- Pagefind indexes only approved public content; privacy/search/404/blank Home are excluded.
- Default OG metadata, favicon, web manifest, JSON-LD, and security headers.

## Verification Evidence

- `astro check`: 0 errors, 0 warnings, 0 hints.
- ESLint and Prettier: pass.
- Vitest: 4 tests pass.
- Static build: 22 content pages plus 18 compatibility redirects generated.
- Pagefind: 18 approved content pages indexed.
- Generated HTML audit: 22 content pages and 18 redirects pass links, redirect targets, H1, images, robots, and leakage checks.
- Playwright: 5 tests pass for empty Home, direct Blog entry, desktop/mobile navigation, system dark mode, Mermaid, and Pagefind.
- GitHub Pages public-beta build and all 5 Playwright tests pass under `/Blog/`.
- GitHub Actions build and deploy jobs pass for commit `8638e11`.
- Online smoke passes for Home, Blog, topic entry, images, Pagefind, mobile navigation, and zero failed network responses.
- Screenshots: `artifacts/qa/`.

## Required User Gate

The public beta is live at `https://chengjt23.github.io/Blog/`. The repository is public, `main` deploys
through GitHub Actions, and indexing remains disabled. No custom domain or production indexing action has
been performed.

The remaining release gate is the user's online public-beta approval. Production indexing, sitemap
submission, and any custom domain change require a separate explicit decision.

## Deferred P1/P2

- Convert the remaining non-core equation fences to KaTeX.
- Per-article generated OG images.
- Projects, Notes, Tags, and Archive until real approved content exists.
- Giscus, analytics, newsletter, and i18n.
- Lighthouse thresholds and Mermaid bundle optimization.
