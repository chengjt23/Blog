import path from 'node:path';
import { visit } from 'unist-util-visit';

const dimensions = {
  'B0_migration_likelihood.png': [2828, 724],
  'B1_brownian_bridge.png': [1870, 676],
  'B2_path_kl_projection.png': [1871, 712],
  'B3_schrodinger_system.png': [1924, 603],
  'B4_reciprocal_markov.png': [1926, 649],
  'B5_entropic_couplings.png': [1887, 647],
  'B7_sinkhorn_diagnostics.png': [2587, 760],
  'B8_population_regression.png': [1964, 776],
  'B9_markovian_projection.png': [2594, 776],
  'B10_sf2m_population_identity.png': [2015, 774],
  'B12_same_marginals_paths.png': [2802, 758],
  'B13_rare_event_doob.png': [2568, 778],
};

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
