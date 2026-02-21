import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py as h5
import time
import utils
import parser_ops

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

parser = parser_ops.get_parser()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# .......................Load the Data...........................................
print('\n Loading ' + args.data_opt + ' test dataset...')
kspace_dir, coil_dir, mask_dir, saved_model_dir = utils.get_test_directory(args)

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset
kspace_test = h5.File(kspace_dir, "r")['kspace'][:]
sens_maps_testAll = h5.File(coil_dir, "r")['sens_maps'][:]
original_mask = sio.loadmat(mask_dir)['mask']

print('\n Normalize kspace to 0-1 region')
for ii in range(np.shape(kspace_test)[0]):
    kspace_test[ii, :, :, :] = kspace_test[ii, :, :, :] / np.max(np.abs(kspace_test[ii, :, :, :][:]))

# %% Train and loss masks are kept same as original mask during inference
nSlices, *_ = kspace_test.shape
test_mask = np.complex64(np.tile(original_mask[np.newaxis, :, :], (nSlices, 1, 1)))

print('\n size of kspace: ', kspace_test.shape, ', maps: ', sens_maps_testAll.shape, ', mask: ', test_mask.shape)

# infer global dims from test data so they match exactly
nSlices, nrow_data, ncol_data, ncoil_data = kspace_test.shape
print("Using data shapes for globals in TEST:",
      "nrow_GLOB =", nrow_data, "ncol_GLOB =", ncol_data, "ncoil_GLOB =", ncoil_data)

args.nrow_GLOB  = nrow_data
args.ncol_GLOB  = ncol_data
args.ncoil_GLOB = ncoil_data

# %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency
if args.data_opt == 'Coronal_PD':
    test_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
    test_mask[:, :, 352:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

test_refAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
test_inputAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

print('\n generating the refs and sense1 input images')
for ii in range(nSlices):
    sub_kspace = kspace_test[ii] * np.tile(test_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    test_refAll[ii] = utils.sense1(kspace_test[ii, ...], sens_maps_testAll[ii, ...])
    test_inputAll[ii] = utils.sense1(sub_kspace, sens_maps_testAll[ii, ...])

sens_maps_testAll = np.transpose(sens_maps_testAll, (0, 3, 1, 2))
all_ref_slices, all_input_slices, all_recon_slices = [], [], []

print('\n  loading the saved model ...')
tf.compat.v1.reset_default_graph()
loadChkPoint = tf.compat.v1.train.latest_checkpoint(saved_model_dir)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    new_saver = tf.compat.v1.train.import_meta_graph(saved_model_dir + '/model_test.meta')
    new_saver.restore(sess, loadChkPoint)

    # ..................................................................................................................
    graph = tf.compat.v1.get_default_graph()
    nw_output = graph.get_tensor_by_name('nw_output:0')
    nw_kspace_output = graph.get_tensor_by_name('nw_kspace_output:0')
    mu_param = graph.get_tensor_by_name('mu:0')
    x0_output = graph.get_tensor_by_name('x0:0')
    all_intermediate_outputs = graph.get_tensor_by_name('all_intermediate_outputs:0')

    # ...................................................................................................................
    trn_maskP = graph.get_tensor_by_name('trn_mask:0')
    loss_maskP = graph.get_tensor_by_name('loss_mask:0')
    nw_inputP = graph.get_tensor_by_name('nw_input:0')
    sens_mapsP = graph.get_tensor_by_name('sens_maps:0')
    weights = sess.run(tf.compat.v1.global_variables())

    for ii in range(nSlices):

        ref_image_test = np.copy(test_refAll[ii, :, :])[np.newaxis]
        nw_input_test = np.copy(test_inputAll[ii, :, :])[np.newaxis]
        sens_maps_test = np.copy(sens_maps_testAll[ii, :, :, :])[np.newaxis]
        testMask = np.copy(test_mask[ii, :, :])[np.newaxis]
        ref_image_test, nw_input_test = utils.complex2real(ref_image_test), utils.complex2real(nw_input_test)

        tic = time.time()
        dataDict = {nw_inputP: nw_input_test, trn_maskP: testMask, loss_maskP: testMask, sens_mapsP: sens_maps_test}
        nw_output_ssdu, *_ = sess.run([nw_output, nw_kspace_output, x0_output, all_intermediate_outputs, mu_param], feed_dict=dataDict)
        toc = time.time() - tic
        ref_image_test = utils.real2complex(ref_image_test.squeeze())
        nw_input_test = utils.real2complex(nw_input_test.squeeze())
        nw_output_ssdu = utils.real2complex(nw_output_ssdu.squeeze())

        if args.data_opt == 'Coronal_PD':
            """window levelling in presence of fully-sampled data"""
            factor = np.max(np.abs(ref_image_test[:]))
        else:
            factor = 1

        ref_image_test = np.abs(ref_image_test) / factor
        nw_input_test = np.abs(nw_input_test) / factor
        nw_output_ssdu = np.abs(nw_output_ssdu) / factor

        # ...............................................................................................................
        all_recon_slices.append(nw_output_ssdu)
        all_ref_slices.append(ref_image_test)
        all_input_slices.append(nw_input_test)

        print('\n Iteration: ', ii, 'elapsed time %f seconds' % toc)

# Convert lists to numpy arrays
all_recon_slices = np.asarray(all_recon_slices)
all_ref_slices = np.asarray(all_ref_slices)
all_input_slices = np.asarray(all_input_slices)

# 1. Evaluate Quantitative Metrics
mean_ssim = 0
mean_psnr = 0
for i in range(nSlices):
    # Calculate data range for metrics
    d_range = all_ref_slices[i].max() - all_ref_slices[i].min()
    
    mean_ssim += ssim(all_ref_slices[i], all_recon_slices[i], data_range=d_range)
    mean_psnr += psnr(all_ref_slices[i], all_recon_slices[i], data_range=d_range)

mean_ssim /= nSlices
mean_psnr /= nSlices

print(f"\n" + "="*40)
print(f"       EVALUATION RESULTS (Brain MRI)")
print(f"="*40)
print(f"Average PSNR: {mean_psnr:.2f} dB")
print(f"Average SSIM: {mean_ssim:.4f}")
print(f"="*40 + "\n")

# 2. Save Numerical Data (for MATLAB or Python analysis locally)
output_dict = {
    'reconstruction': all_recon_slices,
    'reference': all_ref_slices,
    'input_undersampled': all_input_slices
}
sio.savemat('ssdu_brain_results.mat', output_dict)
print("-> Saved raw reconstruction arrays to 'ssdu_brain_results.mat'")

# 3. Save Visual Plot (so you can download and view it)
slice_num = min(5, nSlices - 1) # Safely pick a slice (e.g., slice 5)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(all_input_slices[slice_num], cmap='gray')
plt.title('Undersampled Input')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(all_recon_slices[slice_num], cmap='gray')
slice_psnr = psnr(all_ref_slices[slice_num], all_recon_slices[slice_num], data_range=all_ref_slices[slice_num].max()-all_ref_slices[slice_num].min())
plt.title(f'SSDU Recon (PSNR: {slice_psnr:.2f})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(all_ref_slices[slice_num], cmap='gray')
plt.title('Reference (Fully Sampled)')
plt.axis('off')

plt.tight_layout()
plt.savefig('reconstruction_preview.png', bbox_inches='tight', dpi=150)
print("-> Saved preview image to 'reconstruction_preview.png'")
