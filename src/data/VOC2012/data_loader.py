import xmltodict
import os
import json
import pandas as pd
from pandas.io.json import json_normalize



def load_XML(directory_in, file_out):
    annotations = []
    for fname in os.listdir(directory_in):
        with open(os.path.join(directory_in, fname)) as f_in:
            annotations.append(xmltodict.parse(f_in.read()))

    with open(file_out, 'w') as f_out:
        f_out.write(json.dumps(annotations))

    return load_json(file_out)


def load_json(file_in):
    df = pd.read_json(file_in, orient='records')
    df = json_normalize(df['annotation'], max_level =0)

    for col in ['size', 'source']:
        add_cols = json_normalize(df[col])
        df = df.join(add_cols)
        df = df.drop(col, axis=1)

    df['object'] = df['object'].apply(dict_to_list_of_dicts)
    df['object_count'] = df['object'].apply(len)

    for col in ['width', 'height', 'depth']:
        df[col] = df[col].astype(int)

    df['img_id'] = df.index
    return df

def dict_to_list_of_dicts(d):
    if isinstance(d, list):
        return d
    else:
        return [d]