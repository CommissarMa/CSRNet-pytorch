#%%
import numpy as np
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def generate_perspective_densitymap(image, points, p_map):
    '''
    Use perspective kernel to generate density map. 
    image: the image. 
    points: n points with [col, row]
    p_map: perspective map. 
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
        point2density = np.zeros((image_h, image_w), dtype=np.float32)
        point2density[r,c] = 1
        sigma = int(15 / p_map[r,c])
        densitymap += gaussian_filter(point2density, sigma=sigma, mode='constant')

    densitymap = densitymap / densitymap.sum() * points_quantity
    return densitymap    


if __name__ == '__main__':
    mat = loadmat('perspective_roi.mat')
    mask = mat['roi']['mask'][0][0]     # mask
    p_map = mat['pMapN']                # perspective map

    mat = loadmat('mall_gt.mat')
    frame = mat['frame'][0]             # all annotated heads

    if not os.path.exists('./densitymaps'):
        os.mkdir('./densitymaps')
    for i, image_name in tqdm(enumerate(os.listdir('./frames'))):
        if '.jpg' not in image_name:
            continue
        image = plt.imread(os.path.join('./frames', image_name))
        plt.imshow(image)
        plt.figure()
        points = frame[i][0][0][0]
        densitymap = generate_perspective_densitymap(image, points, p_map)
        np.save('./densitymaps/'+image_name.replace('.jpg','.npy'),densitymap)
    print('finished.')
    
    
    


# %%
