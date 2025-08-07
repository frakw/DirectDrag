# usage: python extract_drag_bench.py [drag_bench_data_path]

import sys
import json
import os
import pickle
from PIL import Image
import numpy as np


def main(path):
    categories = os.listdir(path)
    for category in categories:
        category_path = os.path.join(path, category)
        if not os.path.isdir(category_path):
            continue
        samples = os.listdir(category_path)
        for sample in samples:
            sample_path = os.path.join(category_path, sample)
            if not os.path.isdir(sample_path):
                continue
            pkl_path = os.path.join(sample_path, 'meta_data.pkl')
            drag_instruction_json_path = os.path.join(sample_path, 'drag_instruction.json')
            mask_png_path = os.path.join(sample_path, 'mask.png')
            if not os.path.isfile(pkl_path):
                continue
            print(f'Processing {pkl_path}')
            with open(pkl_path, 'rb') as file:
                pkl_data = pickle.load(file)
                #print(pkl_data)
                drag_instruction = {}
                drag_instruction['prompt'] = pkl_data['prompt']
                drag_instruction['points'] = pkl_data['points']
                with open(drag_instruction_json_path, "w") as json_file:
                    json_data = json.dumps(drag_instruction, indent=4)
                    json_file.write(json_data)
                mask = np.array(pkl_data['mask'], dtype=np.uint8) * 255
                mask_img = Image.fromarray(mask, mode='L')
                mask_img.save(mask_png_path)

if __name__ == '__main__':
    path = ""
    if len(sys.argv) > 1:
        path = sys.argv[1]

    if path == "":
        path = './drag_bench_data'
    main(path)
