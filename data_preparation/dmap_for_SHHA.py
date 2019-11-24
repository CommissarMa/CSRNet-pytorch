#%%
'''
This script is for generating the ground truth density map 
for ShanghaiTech PartA. 
'''
import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM
from tqdm import tqdm


def generate_k_nearest_kernel_densitymap(image,points):
    '''
    Use k nearest kernel to construct the ground truth density map 
    for ShanghaiTech PartA. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [col,row]. 
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
    if points_quantity == 0:
        return densitymap
    else:
        # build kdtree
        tree = scipy.spatial.KDTree(points_coordinate.copy(), leafsize=2048)
        # query kdtree
        distances, locations = tree.query(points_coordinate, k=4)
        for i, pt in enumerate(points_coordinate):
            pt2d = np.zeros((image_h,image_w), dtype=np.float32)
            if int(pt[1])<image_h and int(pt[0])<image_w:
                pt2d[int(pt[1]),int(pt[0])] = 1.
            else:
                continue
            if points_quantity > 3:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            else:
                sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
            densitymap += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        return densitymap


if __name__ == "__main__":
    phase_list = ['train','test']
    for phase in phase_list:
        if not os.path.exists('./'+phase+'_data/densitymaps'):
            os.makedirs('./'+phase+'_data/densitymaps')
        image_file_list = os.listdir('./'+phase+'_data/images')
        for image_file in tqdm(image_file_list):
            image_path = './'+phase+'_data/images/' + image_file
            mat_path = image_path.replace('images','ground_truth').replace('IMG','GT_IMG').replace('.jpg','.mat')
            image = plt.imread(image_path)
            mat = loadmat(mat_path)
            points = mat['image_info'][0][0][0][0][0]
            # generate densitymap
            densitymap = generate_k_nearest_kernel_densitymap(image,points)
            np.save(image_path.replace('images','densitymaps').replace('.jpg','.npy'),densitymap)
        print(phase+' density maps have generated.')


# %%
