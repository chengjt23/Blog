# Research Notes Site

Astro static site for publishing the approved subset of the research workspace.

## Current State

- Local release candidate is implemented.
- All preview pages are `noindex,nofollow`.
- Public identity and release approval are intentionally unresolved.
- Do not create a public repository or deploy this directory before `DECISIONS.md` is approved.

See [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) for completed checks and remaining gates.

## Toolchain

- Node.js 24
- npm 11 with `package-lock.json`
- Astro 7 + TypeScript strict
- Vitest + Playwright
- Pagefind, KaTeX, Mermaid, and Shiki

## Local Development

Port 4321 is reserved by Windows on the current machine. Start the managed background server on 4173:

```powershell
npx astro dev --host 127.0.0.1 --port 4173 --background
npx astro dev status
npx astro dev logs
npx astro dev stop
```

Open `http://127.0.0.1:4173/`.

The public header contains only `Home` and `Blog`. Home is intentionally blank until approved personal
content is supplied. The current long-form topic is available at `/blog/schrodinger-bridge/`; legacy
`/series/*` URLs redirect to their `/blog/*` equivalents.

## GitHub Pages

The deployment target is `https://chengjt23.github.io/Blog/`. The workflow in
`.github/workflows/deploy-pages.yml` runs checks, builds with the `/Blog/` base path, runs Chrome tests,
and deploys `dist` after a push to `main`.

Before the first deployment, open the repository's `Settings -> Pages` and set `Source` to
`GitHub Actions`. The initial deployment is a `public-beta` with `noindex,nofollow`.

To reproduce the Pages build locally:

```powershell
$env:SITE_ENV = 'public-beta'
$env:PUBLIC_SITE_URL = 'https://chengjt23.github.io'
$env:PUBLIC_BASE_PATH = '/Blog'
$env:PUBLIC_ALLOW_INDEXING = 'false'
$env:PUBLIC_BETA_APPROVED = 'true'
$env:PUBLIC_SITE_NAME = '研究笔记'
$env:PUBLIC_AUTHOR_NAME = 'chengjt23'
$env:PUBLIC_SITE_DESCRIPTION = '生成建模、随机过程与最优传输的系统教程与研究笔记。'
$env:PUBLIC_CONTENT_LICENSE = 'all-rights-reserved'
npm run build
npm run test:e2e
```

`PUBLIC_SITE_URL` is the HTTPS origin only. Project paths belong in `PUBLIC_BASE_PATH`. When moving to a
custom domain later, set `PUBLIC_BASE_PATH=/` and run the full build and browser suite again.

## Content Sync

Research files are read-only inputs. A dry run is the default:

```powershell
npm run sync-content
```

After reviewing the plan, explicitly generate the public layer:

```powershell
npm run sync-content:write
npm run validate-content
```

The sync step writes only:

- `src/content/series/schrodinger-bridge/`
- `public/images/bridge/`
- `generated-manifest.json`

It never modifies `../bridge/`, `../diffusion/`, or `../references/`.

## Verification

```powershell
npm run check
npm run lint
npm run format:check
npm test
npm run build
npm run test:e2e
```

`npm run build` also creates the Pagefind index and audits generated HTML for internal broken links,
missing H1/alt/dimensions, preview indexing, and private-path leakage.

## Environments

`SITE_ENV` must be one of:

- `local`
- `private-preview`
- `public-beta`
- `production`

Only production may enable indexing. Production validation requires explicit public identity, HTTPS URL,
content license, and `PUBLIC_RELEASE_APPROVED=true`. Preview defaults cannot pass production validation.

GitHub Pages does not apply the generated Cloudflare-style `_headers` file. The site remains a static,
no-analytics build; page-level robots metadata controls indexing until production approval.
