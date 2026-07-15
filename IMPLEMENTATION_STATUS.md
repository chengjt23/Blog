# Implementation Status

> State: `github_pages_ready_waiting_for_release_approval`
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
- Screenshots: `artifacts/qa/`.

## Required User Gate

The user created `git@github.com:chengjt23/Blog.git`; SSH access is available and the remote is empty. The
local `main` repository is initialized and bound to `origin`. No commit, push, Pages deployment, custom
domain, or indexing action has been performed.

Before those actions, the user must complete the public-release fields in `DECISIONS.md` and approve the
local preview. The implementation will then replace preview identity metadata, finalize License/Privacy,
set publication dates and draft flags, and run the production gate.

## Deferred P1/P2

- Convert the remaining non-core equation fences to KaTeX.
- Per-article generated OG images.
- Projects, Notes, Tags, and Archive until real approved content exists.
- Giscus, analytics, newsletter, and i18n.
- Lighthouse thresholds and Mermaid bundle optimization.
