import os
import xml.etree.ElementTree as ET
import shutil


# Total Dataset's Class Count
# Image counts: 853
# with_mask : 3232
# without_mask : 717
# mask_weared_incorrect : 123

with_mask, without_mask, mask_weared_incorrect = 0, 0, 0

os.chdir('annotations')
files = os.listdir()
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    iter_element = root.iter(tag='object')
    for element in iter_element:
        name = element.findtext('name')
        if name=='with_mask':
            with_mask += 1
        elif name=='without_mask':
            without_mask += 1
        elif name=='mask_weared_incorrect':
            mask_weared_incorrect += 1

print('-'*30)
print('Total Dataset\'s Class Count')
print('with_mask :', with_mask)
print('without_mask :', without_mask)
print('mask_weared_incorrect :', mask_weared_incorrect)
print('-'*30)



# There is a class imbalance problem in my custom dataset.
# Thus, I reconstruct my custom dataset.
# If the image has at least one class of 'Without_mask', 'mask_weared_incorrect', I include the image in the reconstructed data.

balanced_item = []
imbalanced_class = ['mask_weared_incorrect']

files = os.listdir()
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    iter_element = root.iter(tag='object')
    for element in iter_element:
        if element.findtext('name') in imbalanced_class:
            balanced_item.append(file.rstrip('.xml'))
balanced_item = list(set(balanced_item))
print('Total images in balance:', len(balanced_item))
print('-'*30)



# I made new folders named 'images_balanced', 'annotations_balanced' which has balanced dataset
# 1. Annotations part
anno_path = '../annotations_balance/'
i = 1
for item in balanced_item:
    anno = item + '.xml'
    shutil.copy(anno, anno_path + 'maskdata' + str(i) + '.xml')
    i += 1

# 2. Image part
os.chdir('../images')
img_path = '../images_balance/'
j = 1
for item in balanced_item:
    img = item + '.png'
    shutil.copy(img,  img_path + 'maskdata' + str(j) + '.png')
    j += 1




# Balanced Data's Class Count
# Image counts: 97
# with_mask : 624
# without_mask : 157
# mask_weared_incorrect : 123
with_mask, without_mask, mask_weared_incorrect = 0, 0, 0

os.chdir('../annotations_balance')
files = os.listdir()
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    iter_element = root.iter(tag='object')
    for element in iter_element:
        name = element.findtext('name')
        if name=='with_mask':
            with_mask += 1
        elif name=='without_mask':
            without_mask += 1
        elif name=='mask_weared_incorrect':
            mask_weared_incorrect += 1

print('<Cunclusion>')
print('Balanced Dataset\'s Class Count')
print('Total images in balance:', len(balanced_item))
print('with_mask :', with_mask)
print('without_mask :', without_mask)
print('mask_weared_incorrect :', mask_weared_incorrect)
print('-'*30)



print()
print('-'*30)
print('Finished')
print('-'*30)