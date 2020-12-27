# MIT License
#
# Copyright (c) 2020 Immersive Limit LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# https://github.com/immersive-limit/coco-manager
#
#
import json
from pathlib import Path


class CocoFilter():
    """ Filters the COCO dataset
    """

    def _process_info(self):
        self.info = self.coco['info']

    def _process_licenses(self):
        self.licenses = dict()

        if 'licenses' not in self.coco:
            return

        for license in self.coco['licenses']:
            lic_id = license['id']
            if lic_id not in self.licenses:
                self.licenses[lic_id] = license

    def _process_categories(self):
        self.categories = dict()

        if 'categories' not in self.coco:
            return

        for category in self.coco['categories']:
            cat_id = category['id']
            if cat_id not in self.categories:
                self.categories[cat_id] = category

    def _process_images(self):
        self.images = dict()
        self.images_lics = dict()

        if 'images' not in self.coco:
            return

        for image in self.coco['images']:
            image_id = image['id']
            image_lic_id = image['license']
            if image_id not in self.images:
                self.images[image_id] = image
            if image_lic_id not in self.images_lics:
                self.images_lics[image_lic_id] = 0
            self.images_lics[image_lic_id] += 1

    def _process_annotations(self):
        self.annotations = dict()
        self.annotations_cats = dict()
        self.annotations_imgs = dict()

        if 'annotations' not in self.coco:
            return

        for annotation in self.coco['annotations']:
            annotation_id = annotation['id']
            annotation_cat_id = annotation['category_id']
            annotation_img_id = annotation['image_id']
            if annotation_id not in self.annotations:
                self.annotations[annotation_id] = annotation
            if annotation_cat_id not in self.annotations_cats:
                self.annotations_cats[annotation_cat_id] = dict()
            if annotation_img_id not in self.annotations_imgs:
                self.annotations_imgs[annotation_img_id] = 0
            self.annotations_cats[annotation_cat_id][annotation_id] = annotation
            self.annotations_imgs[annotation_img_id] += 1

    def _count(self):
        self.image_cats = dict()
        for cat_id, category in list(self.categories.items()):
            if cat_id in self.annotations_cats:
                for annotation_id, annotation in self.annotations_cats[cat_id].items():
                    image_id = annotation['image_id']

                    if cat_id not in self.image_cats:
                        self.image_cats[cat_id] = dict()
                    if image_id not in self.image_cats[cat_id]:
                        self.image_cats[cat_id][image_id] = 1

        for cat_id, imgs in self.image_cats.items():
            print("\tCat ID: {}, images: {}".format(cat_id, len(imgs)))
        print("Total annotations: {}".format(len(self.annotations)))

        self.orphan_images = list()
        for image_id in self.images:
            if image_id not in self.annotations_imgs:
                self.orphan_images.append(image_id)
        print("Total images: {} (without annotations: {})".format(len(self.images), len(self.orphan_images)))

        print("Total licenses: {}".format(len(self.licenses)))

    def _filter(self):
        def remove_image(image_id):
            license_id = self.images[image_id]['license']
            self.images_lics[license_id] -= 1

            if self.images_lics[license_id] == 0:
                # Deleting the license image if no more images
                del self.licenses[license_id]
                del self.images_lics[license_id]

            # Deleting the given image if no more annotations
            del self.images[image_id]

        for cat_id, category in list(self.categories.items()):
            if category['name'] not in self.filter_categories:
                if cat_id in self.annotations_cats:
                    # Deleting all annotations of the given category
                    for annotation_id in self.annotations_cats[cat_id].keys():
                        image_id = self.annotations[annotation_id]['image_id']
                        self.annotations_imgs[image_id] -= 1

                        if self.annotations_imgs[image_id] == 0:
                            remove_image(image_id)
                            del self.annotations_imgs[image_id]

                        del self.annotations[annotation_id]

                    del self.annotations_cats[cat_id]

                # Deleting the given category
                del self.categories[cat_id]

        for image_id in self.orphan_images:
            remove_image(image_id)

    def _reindex(self):
        index = 0
        for cat_id, category in list(self.categories.items()):
            index += 1
            category['id'] = index
            if cat_id in self.annotations_cats:
                for _, annotation in self.annotations_cats[cat_id].items():
                    annotation['category_id'] = index

    def main(self, args):
        # Open JSON
        self.input_json_path = Path(args.input_json)
        self.output_json_path = Path(args.output_json)
        self.filter_categories = args.categories

        # Load the JSON
        print('Loading JSON file "%s"' % (self.input_json_path.absolute(),))
        with open(self.input_json_path) as json_file:
            self.coco = json.load(json_file)

        # Process the JSON
        print('Processing input JSON...')
        self._process_info()
        self._process_licenses()
        self._process_categories()
        self._process_images()
        self._process_annotations()
        self._count()

        # Filter to specific categories
        print('Modifying data...')
        self._filter()
        self._reindex()

        # Build new JSON
        print('Building new JSON structure...')
        new_master_json = {
            'info': self.info,
            'licenses': list(self.licenses.values()),
            'images': list(self.images.values()),
            'annotations': list(self.annotations.values()),
            'categories': list(self.categories.values())
        }

        # Write the JSON to a file
        print('Saving new JSON file "%s"' % (self.output_json_path.absolute(),))
        with open(self.output_json_path, 'w+') as output_file:
            json.dump(new_master_json, output_file)

        print('Done')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COCO dataset JSON filter: "
                                                 "Filters a COCO Instances JSON file to only include specified categories. ")

    parser.add_argument("-i", "--input_json", dest="input_json", required=True,
                        help="path to a json file in COCO format")
    parser.add_argument("-o", "--output_json", dest="output_json", required=True,
                        help="path to save the output JSON")
    parser.add_argument("-c", "--categories", nargs='+', dest="categories", required=True,
                        help="List of category names separated by spaces, e.g. -c person dog bicycle")

    args = parser.parse_args()

    cf = CocoFilter()
    cf.main(args)
