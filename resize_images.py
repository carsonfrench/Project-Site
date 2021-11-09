from PIL import Image
import os

filepath = 'dataset/movieposter/img_41K/img_41K' 
files = os.listdir(filepath)


def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGB', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

for f in files:
    image = Image.open(f'{filepath}/{f}')
    new_image = make_square(image)
    new_image.save(f'{f}')
