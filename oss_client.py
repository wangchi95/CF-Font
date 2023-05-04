import oss2
import cv2
import numpy as np
import urllib.request
import os

from itertools import islice
# import requests

class OSSCTD(object):
    def __init__(self):
        host_name = 'xxx'
        bucket_name = 'xxx'
        auth = oss2.Auth('xxx', 'xxx')
        self.bucket = oss2.Bucket(auth, host_name, bucket_name)
        self.url_prefix = 'xxx'

    def read_file(self, file_path, auth_check=False):
        if auth_check:
            return self.bucket.get_object(file_path).read()
        else:
            url = os.path.join(self.url_prefix, file_path)
            return urllib.request.urlopen(url).read()

    def read_image(self, file_path, mode=cv2.IMREAD_UNCHANGED, auth_check=False):
        img_data = self.read_file(file_path, auth_check=auth_check)
        img_data = np.asarray(bytearray(img_data), dtype='uint8')
        img = cv2.imdecode(img_data, mode)
        return img

    def write_file(self, local_file, remote_file):
        with open(local_file, 'rb') as fin:
            data = fin.read()
            self.bucket.put_object(remote_file, data)

    def fetch_file(self, remote_file, local_file):
        self.bucket.get_object_to_file(remote_file, local_file)

    # def requests_write_file(self, src, dst):
    #     with open(src, 'wb') as f:
    #         url = os.path.join(self.url_prefix, dst)
    #         response = requests.get(url)
    #         f.write(response.content)

    def showFiles(self, bucket):
        print("Show All Files:")
        for b in islice(oss2.ObjectIterator(bucket, prefix='xxx'), None):
            print(b.key)