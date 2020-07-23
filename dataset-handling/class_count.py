import os
import xml.etree.ElementTree as ET
import shutil


with_mask, without_mask, mask_weared_incorrect = 0, 0, 0

os.chdir('annotations_balance')
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