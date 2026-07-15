# Publication Decisions

This file separates local preview defaults from public-release approvals.

## Preview Defaults

- Site name: `研究笔记`
- Author: `Preview Author`
- Avatar: none
- Favicon: letter mark
- Content license: all rights reserved
- Analytics: disabled
- Comments: disabled
- Publishable sources: Bridge B0--B14 and the 12 referenced original figures only
- Source notes, derivations, verification code, PDFs, staging, worklogs, and Diffusion drafts: denied

## Public Release Gate

The following values are intentionally unresolved and must be approved by the user before C120:

- [x] Public site name: `研究笔记`; public author: `chengjt23`
- [x] Content, figure, and code licenses: all rights reserved for the initial release
- [x] Public content boundary: Bridge B0--B14 and the 12 referenced original figures only
- [x] Contact and social links: no direct contact link in the initial site
- [x] Analytics/privacy choice: analytics disabled
- [x] Repository ownership: `chengjt23/Blog`
- [x] Release-candidate approval for a noindex GitHub Pages public beta

No unresolved value in this section may be inferred from the preview defaults.

## Deployment Target

- Initial host: GitHub Pages
- Repository: `git@github.com:chengjt23/Blog.git`
- Public-beta URL: `https://chengjt23.github.io/Blog/`
- Base path: `/Blog`
- Initial indexing: disabled
- Custom domain: deferred

Production indexing remains unapproved. Enabling `index,follow`, sitemap submission, or a production
release requires a separate explicit user decision after the public-beta smoke test.

## Optional Future Home Content

Home is intentionally blank and may remain blank for release. Personal introduction content is added only
after the user supplies and approves it; it does not create an About route or block the release gate.
