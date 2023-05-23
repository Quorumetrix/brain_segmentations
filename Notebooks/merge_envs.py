import yaml

with open('cle.yml') as f:
    env1 = yaml.safe_load(f)
with open('brain_segmentations.yml') as f:
    env2 = yaml.safe_load(f)
with open('napari.yml') as f:
    env3 = yaml.safe_load(f)

merged = env1
merged['dependencies'].extend(x for x in env2['dependencies'] if x not in env1['dependencies'])

merged['dependencies'].extend(x for x in env3['dependencies'] if x not in env1['dependencies'])
with open('merged.yml', 'w') as f:
    yaml.safe_dump(merged, f)
