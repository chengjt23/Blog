import { rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { assertInside } from './publish-utils';

const siteRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const distRoot = path.join(siteRoot, 'dist');

assertInside(siteRoot, distRoot);
await rm(distRoot, { recursive: true, force: true });
console.log('Removed the previous dist directory.');
