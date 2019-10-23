#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image

import labelme

import pickle
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    parser.add_argument('--polygons', nargs='+', help='polygons json files (at least one)', required=True)
    parser.add_argument('--local_dataset_dir', help='where the input raw img files are on this machine', required=True)
    parser.add_argument('--merge_classes', nargs='*', help='list classes you want to merge (e.g. make all paths/sidewalks the same color')
    parser.add_argument('--exclude', nargs='*', help='list of filenames you want to exclude from this dataset')
    parser.add_argument('--include', nargs='*', help='list of filenames you want to include from this dataset')
    args = parser.parse_args()

    if args.merge_classes is not None:
        args.output_dir += '_merge'
        for cl in args.merge_classes:
            args.output_dir += '_{}'.format(cl)

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir, '==> deleting.')
        import shutil
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
    os.makedirs(osp.join(args.output_dir, 'SegmentationClass'))
    os.makedirs(osp.join(args.output_dir, 'SegmentationClassPNG'))
    os.makedirs(osp.join(args.output_dir, 'SegmentationClassVisualization'))
    print('Creating dataset:', args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = labelme.utils.label_colormap(255)

    if args.merge_classes is not None:
        for cl in args.merge_classes:
            print("Merging {} into {}".format(cl, args.merge_classes[0]))
            class_name_to_id[cl] = class_name_to_id[args.merge_classes[0]]

    out_colormap_file = osp.join(args.output_dir, "colormap.pkl")
    with open(out_colormap_file, "wb") as f:
        pickle.dump(colormap, f)
    print('Saved colormap:', out_colormap_file)

    label_files = args.polygons
    for label_file in label_files:
        with open(label_file) as f:
            print('Generating dataset from:', label_file)
            for line in tqdm(f.readlines()):
                data = json.loads(line)
                annotation = data
                img_filename = osp.basename(annotation["image_payload"]["image_uri"])
                img_subdir = osp.join(*[str(d) for d in osp.dirname(annotation["image_payload"]["image_uri"]).split("/")[3:]])
                base = osp.splitext(img_filename)[0]
                base = base.replace("(","").replace(")","").replace(" ","_")

                skip = False
                if args.include is not None:
                    for incl in args.include:
                        if incl not in base:
                            skip = True
                if args.exclude is not None:
                    for excl in args.exclude:
                        if excl in base:
                            skip = True
                if skip:
                    continue

                out_img_file = osp.join(
                    args.output_dir, 'JPEGImages', base + '.jpg')
                out_lbl_file = osp.join(
                    args.output_dir, 'SegmentationClass', base + '.npy')
                out_png_file = osp.join(
                    args.output_dir, 'SegmentationClassPNG', base + '.png')
                out_viz_file = osp.join(
                    args.output_dir,
                    'SegmentationClassVisualization',
                    base + '.jpg',
                )

                img_path = osp.join(args.local_dataset_dir, img_subdir)
                img_file = osp.join(img_path, img_filename)
                img = np.asarray(PIL.Image.open(img_file))
                pil_img = PIL.Image.fromarray(img)
                try:
                    pil_img.save(out_img_file)
                except:
                    pil_img = pil_img.convert("RGB")
                    pil_img.save(out_img_file)


                lbl = labelme.utils.google_annotations_to_label(
                    img_shape=img.shape,
                    annotations=data['annotations'],
                    label_name_to_value=class_name_to_id,
                )
                labelme.utils.lblsave(out_png_file, lbl)

                np.save(out_lbl_file, lbl)

                viz = labelme.utils.draw_label(
                    lbl, img, class_names, colormap=colormap)
                PIL.Image.fromarray(viz).save(out_viz_file)

if __name__ == '__main__':
    main()
