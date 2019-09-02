from os import listdir
from os.path import isfile, join
mypath_im = 'training/image_2'
mypath_gt_im = 'training/gt_image_2'
im_name_list = [ f for f in listdir('./' + mypath_im) if isfile(join('./' + mypath_im, f))]
#im_name_list = [mypath_im + '/' + f for f in listdir('./' + mypath_im) if isfile(join('./' + mypath_im, f))]
#gt_im_name_list = [mypath_gt_im + '/' + f for f in listdir('./' + mypath_im) if isfile(join('./' + mypath_im, f))]

with open('train.txt', 'w') as f:
    for item in im_name_list:
        im_item = mypath_im + '/' + item
        gt_im_item = mypath_gt_im + '/' + item
        f.write("%s %s\n" %(im_item , gt_im_item))

