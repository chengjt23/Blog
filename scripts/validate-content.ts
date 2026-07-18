import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readJson, sha256 } from './publish-utils';

type GeneratedManifest = {
  version: number;
  transformerVersion: string;
  entries: Array<{
    source: string;
    sourceSha256: string;
    output: string;
    outputSha256: string;
  }>;
};

type Denylist = { contentPatterns: string[] };
type Allowlist = { sources: string[]; assets: string[] };

const siteRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const manifest = await readJson<GeneratedManifest>(path.join(siteRoot, 'generated-manifest.json'));
const denylist = await readJson<Denylist>(path.join(siteRoot, 'publish-denylist.json'));
const bridgeAllowlist = await readJson<Allowlist>(path.join(siteRoot, 'content-allowlist.json'));
const diffusionAllowlist = await readJson<Allowlist>(
  path.join(siteRoot, 'content-allowlist.diffusion.json'),
);
const expectedEntries =
  bridgeAllowlist.sources.length +
  bridgeAllowlist.assets.length +
  diffusionAllowlist.sources.length +
  diffusionAllowlist.assets.length;

if (manifest.entries.length !== expectedEntries) {
  throw new Error(
    `Expected ${expectedEntries} generated entries from the publication allowlists, received ${manifest.entries.length}.`,
  );
}

const outputSet = new Set<string>();
for (const entry of manifest.entries) {
  if (path.isAbsolute(entry.source) || path.isAbsolute(entry.output)) {
    throw new Error(`Manifest contains an absolute path: ${entry.source} -> ${entry.output}`);
  }
  if (outputSet.has(entry.output)) throw new Error(`Duplicate generated output: ${entry.output}`);
  outputSet.add(entry.output);

  const outputPath = path.join(siteRoot, entry.output);
  const output = await readFile(outputPath);
  const digest = sha256(output);
  if (digest !== entry.outputSha256) {
    throw new Error(`Generated output digest mismatch: ${entry.output}`);
  }

  if (entry.output.endsWith('.md')) {
    const text = output.toString('utf8');
    for (const pattern of denylist.contentPatterns.filter((value) => value !== 'Preview Author')) {
      if (text.includes(pattern))
        throw new Error(`${entry.output} contains denied text: ${pattern}`);
    }
    if (/\]\((?!https?:|mailto:|#|\/)[^)]+\)/.test(text)) {
      throw new Error(`${entry.output} contains an unresolved relative Markdown link.`);
    }
  }
}

console.log(`Validated ${manifest.entries.length} generated outputs and their digests.`);
