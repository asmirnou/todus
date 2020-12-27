import argparse
from pathlib import Path
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils.label_map_util import load_labelmap


def save_labelmap(label_map, path):
    """Saves label map proto.

        Args:
          label_map: label map object
          path: path to StringIntLabelMap proto text file.
        """
    msg = str(text_format.MessageToBytes(label_map, as_utf8=True), 'utf-8')
    with open(path, 'w') as f:
        f.write(msg)


def parse_commandline_arguments():
    parser = argparse.ArgumentParser(description='Filters label text file.')

    parser.add_argument("-i", "--input_pdtxt", dest="input_pdtxt", required=True,
                        help="path to a PBTXT file in protobuf format")
    parser.add_argument("-o", "--output_pdtxt", dest="output_pdtxt", required=True,
                        help="path to save the output PBTXT")
    parser.add_argument("-c", "--categories", nargs='+', dest="categories", required=True,
                        help="List of category names separated by spaces, e.g. -c person dog bicycle")
    parser.add_argument('-m', "--mode", dest='mode',
                        choices=['include', 'exclude'], default='include',
                        help='Whether to include categories or exclude them')

    args = parser.parse_args()
    return args


def main():
    args = parse_commandline_arguments()

    input_pbtxt_path = Path(args.input_pdtxt)
    output_pbtxt_path = Path(args.output_pdtxt)

    print('Loading PBTXT file "%s"' % (input_pbtxt_path.absolute(),))
    label_map = load_labelmap(args.input_pdtxt)

    new_label_map = string_int_label_map_pb2.StringIntLabelMap()
    count = 0
    for item in label_map.item:
        if (item.id == 0 and item.name == 'background') or \
                (args.mode == 'include' and item.display_name in args.categories) or \
                (args.mode == 'exclude' and item.display_name not in args.categories):
            if item.id > 0:
                count += 1
                index = count
            else:
                index = item.id

            new_label_map.item.append(string_int_label_map_pb2.StringIntLabelMapItem(
                id=index,
                name=item.name,
                display_name=item.display_name
            ))

    print('Saving PBTXT file "%s"' % (output_pbtxt_path.absolute(),))
    save_labelmap(new_label_map, output_pbtxt_path)


if __name__ == '__main__':
    main()
