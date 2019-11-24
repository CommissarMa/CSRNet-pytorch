'''
This script is for generating the ground truth density map 
for ShanghaiTech PartB. 
'''
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm


def generate_fixed_kernel_densitymap(image,points,sigma=15):
    '''
    Use fixed size kernel to construct the ground truth density map 
    for ShanghaiTech PartB. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [col,row]. 
    sigma: the sigma of gaussian_kernel to simulate a head. 
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]

    # coordinate of heads in the image
    points_coordinate = points
    # quantity of heads in the image
    points_quantity = len(points_coordinate)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
    for point in points_coordinate:
        c = min(int(round(point[0])),image_w-1)
        r = min(int(round(point[1])),image_h-1)
        # point2density = np.zeros((image_h, image_w), dtype=np.float32)
        # point2density[r,c] = 1
        densitymap[r,c] = 1
    # densitymap += gaussian_filter(point2density, sigma=sigma, mode='constant')
    densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

    densitymap = densitymap / densitymap.sum() * points_quantity
    return densitymap    
    
if __name__ == '__main__':
    phase_list = ['train','test']
    for phase in phase_list:
        if not os.path.exists('./'+phase+'_data/densitymaps'):
            os.mkdir('./'+phase+'_data/densitymaps')
        image_file_list = os.listdir('./'+phase+'_data/images')
        for image_file in tqdm(image_file_list):
            image_path = './'+phase+'_data/images/' + image_file
            mat_path = image_path.replace('images','ground_truth').replace('IMG','GT_IMG').replace('.jpg','.mat')
            image = plt.imread(image_path)
            mat = loadmat(mat_path)
            points = mat['image_info'][0][0][0][0][0]
            # generate densitymap
            densitymap = generate_fixed_kernel_densitymap(image,points,sigma=15)
            np.save(image_path.replace('images','densitymaps').replace('.jpg','.npy'),densitymap)
        print(phase+' density maps have generated.')