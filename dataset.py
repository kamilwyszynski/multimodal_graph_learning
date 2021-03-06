import json
import os
from PIL import Image

class VizWiz():

    def __init__(self, path='res/vizwiz', set_option='train', offset=0):
        with open(os.path.join(path, 'annotations', f'{set_option}.json')) as f:
            self.annotations = json.load(f)

        self.images_path = os.path.join(path, set_option)
        self.set_option = set_option
        self.current = offset

    def __iter__(self):
        return self

    def __next__(self):
        img_name = f'VizWiz_{self.set_option}_{self.current:08d}.jpg'
        img = Image.open(os.path.join(self.images_path, img_name))

        cap = ''
        for i in range(5):
            cap += ' ' + self.annotations['annotations'][self.current*5+i]['caption']

        self.current+=1

        return img, cap, img_name

    def get_caption(self, image):
        img_id = int(image[13:-4])

        captions = [self.annotations['annotations'][img_id*5+i]['caption'] for i in range(5)]
        return captions