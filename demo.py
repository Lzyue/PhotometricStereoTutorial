from __future__ import print_function

import numpy as np
import time
from rps import PS
import psutil
import matplotlib.pyplot as plt


DATA_FOLDERNAME = './data/bunny/bunny_lambert/'    # Lambertian diffuse with cast shadow
#DATA_FOLDERNAME = './data/bunny/bunny_lambert_noshadow/'    # Lambertian diffuse without cast shadow

LIGHT_FILENAME = './data/bunny/lights.npy'
MASK_FILENAME = './data/bunny/mask.png'
GT_NORMAL_FILENAME = './data/bunny/gt_normal.npy'


# Photometric Stereo
rps = PS()
rps.load_mask(filename=MASK_FILENAME)    # Load mask image
rps.load_lightnpy(filename=LIGHT_FILENAME)    # Load light matrix
rps.load_npyimages(foldername=DATA_FOLDERNAME)    # Load observations
start = time.time()
rps.solve()    # Compute
elapsed_time = time.time() - start
print("Photometric stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
rps.save_normalmap(filename="./est_normal")    # Save the estimated normal map

# Evaluate the estimate
N_gt = psutil.load_normalmap_from_npy(filename=GT_NORMAL_FILENAME)    # read out the ground truth surface normal
N_gt = np.reshape(N_gt, (rps.height*rps.width, 3))    # reshape as a normal array (p \times 3)
angular_err = psutil.evaluate_angular_error(N_gt, rps.N, rps.background_ind)    # compute angular error
print("Mean angular error [deg]: ", np.mean(angular_err[:]))

# 使用 matplotlib 保存和显示法线图
N = np.reshape(rps.N, (rps.height, rps.width, 3))
N = (N - np.min(N)) / (np.max(N) - np.min(N)) * 255.0
N = N.astype(np.uint8)

# 显示图像
plt.imshow(N)
plt.title('normal map')
plt.show()

# 保存图像
plt.imsave('estimated_normalmap.png', N)

print("done.")
