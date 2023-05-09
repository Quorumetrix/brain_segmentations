import xml.etree.ElementTree as ET
import pandas as pd

def parse_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    series_name = root.get('name')
    target = root.get('target')
    target_resolution = root.get('target-resolution')

    slices = []
    for slice_elem in root.iter('slice'):
        slice_info = {
            'filename': slice_elem.get('filename'),
            'nr': int(slice_elem.get('nr')),
            'width': int(slice_elem.get('width')),
            'height': int(slice_elem.get('height')),
            'anchoring': slice_elem.get('anchoring')
        }
        slices.append(slice_info)

    df_slices = pd.DataFrame(slices)
    return df_slices


def extract_anchoring_data(df_slices):
    anchoring_data = []
    for index, row in df_slices.iterrows():
        if pd.notnull(row['anchoring']):
            params = row['anchoring'].split('&')
            param_dict = {}
            for param in params:
                key, value = param.split('=')
                param_dict[key] = float(value)
            anchoring_data.append({
                'original_index': index,
                'filename': row['filename'],
                'nr': row['nr'],
                'ox': param_dict['ox'],
                'oy': param_dict['oy'],
                'oz': param_dict['oz'],
                'ux': param_dict['ux'],
                'uy': param_dict['uy'],
                'uz': param_dict['uz'],
                'vx': param_dict['vx'],
                'vy': param_dict['vy'],
                'vz': param_dict['vz']
            })
    df_anchoring = pd.DataFrame(anchoring_data)
    return df_anchoring


def create_nr_to_index_mapping(df_anchoring):
    nr_to_index = {}
    for _, row in df_anchoring.iterrows():
        nr_to_index[row['nr']] = row['original_index']
        # Check if the 'nr' value is correctly aligned with the 'filename'
        assert str(row['nr']).zfill(4) in row['filename'], f"nr ({row['nr']}) does not match the filename ({row['filename']})"
    return nr_to_index
