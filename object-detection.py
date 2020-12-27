import os
import json
from urllib.parse import urlparse, unquote
from pip._internal.commands.show import search_packages_info


def get_object_detection_path():
    results = search_packages_info(['object-detection'])
    dist = next(results, None)
    if not dist:
        return ''

    direct_url_json = os.path.join(
        dist.get('location', ''),
        next(filter(lambda file: file.endswith('.dist-info/direct_url.json'),
                    dist['files']), ''))

    with open(direct_url_json) as json_file:
        content = json.load(json_file)
        url = content['url']
        p = urlparse(url)
        return unquote(p.path)


if __name__ == '__main__':
    print(get_object_detection_path())
