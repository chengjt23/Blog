import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { visit } from 'unist-util-visit';

const publicImagesRoot = fileURLToPath(new URL('../public/images/', import.meta.url));

function pngDimensions(file) {
  const bytes = readFileSync(file);
  if (bytes.length < 24 || bytes.toString('ascii', 1, 4) !== 'PNG') {
    throw new Error(`Unsupported public image format: ${file}`);
  }
  return [bytes.readUInt32BE(16), bytes.readUInt32BE(20)];
}

const dimensions = Object.fromEntries(
  ['bridge', 'diffusion'].flatMap((directory) =>
    readdirSync(path.join(publicImagesRoot, directory))
      .filter((name) => name.toLowerCase().endsWith('.png'))
      .map((name) => [name, pngDimensions(path.join(publicImagesRoot, directory, name))]),
  ),
);

export default function rehypePublicImages({ basePath = '' } = {}) {
  return (tree) => {
    visit(tree, 'element', (node, _index, parent) => {
      for (const property of ['href', 'src']) {
        const value = node.properties?.[property];
        if (
          basePath &&
          typeof value === 'string' &&
          value.startsWith('/') &&
          value !== basePath &&
          !value.startsWith(`${basePath}/`)
        ) {
          node.properties[property] = `${basePath}${value}`;
        }
      }

      if (node.tagName !== 'img' || typeof node.properties?.src !== 'string') return;
      const name = path.basename(node.properties.src);
      const size = dimensions[name];
      if (!size) return;
      node.properties.width = size[0];
      node.properties.height = size[1];
      node.properties.loading = 'lazy';
      node.properties.decoding = 'async';

      if (parent?.tagName === 'p' && parent.children?.length === 1) {
        const alt = typeof node.properties.alt === 'string' ? node.properties.alt : '';
        parent.tagName = 'figure';
        parent.properties = { className: ['content-figure'] };
        parent.children = [
          node,
          {
            type: 'element',
            tagName: 'figcaption',
            properties: {},
            children: [{ type: 'text', value: alt }],
          },
        ];
      }
    });
  };
}
