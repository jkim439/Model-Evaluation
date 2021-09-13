import os
import numpy as np
import shutil
import nibabel as nib


palette_path = '/home/jkim/evaluation/segmentation/palette.npy'
palette = np.load(palette_path)
reshape_pal = palette.reshape(256, 3)

label_dict = {}
for c, label in enumerate(reshape_pal):
    label_dict[tuple(label)] = c

def rgb2parr(img):


    im = list(img.getdata())

    im_nd_tmp = np.asarray(im, dtype=np.uint8)
    nonzero = im_nd_tmp.nonzero()

    bef_buf = 0
    nonzero_mask = []
    mask_img = np.zeros((512 * 512), np.uint8)

    for i in nonzero[0]:
        if bef_buf != i:
            val2 = label_dict[im[i]]
            nonzero_mask.append(tuple((i, int(val2))))
            mask_img[i] = int(val2)
        bef_buf = i

    mask_img = mask_img.reshape(512, 512)
    ####

    return mask_img

def rgb2parr_cls(img, cls_no):


    im = list(img.getdata())

    im_nd_tmp = np.asarray(im, dtype=np.uint8)
    nonzero = im_nd_tmp.nonzero()

    bef_buf = 0
    nonzero_mask = []
    mask_img = np.zeros((512 * 512), np.uint8)

    for i in nonzero[0]:
        if bef_buf != i:
            val2 = label_dict[im[i]]
            if val2 == cls_no:
                nonzero_mask.append(tuple((i, int(val2))))
                mask_img[i] = int(val2)
        bef_buf = i

    mask_img = mask_img.reshape(512, 512)
    ####

    return mask_img

def filt_seg_gth(seg_pil, gth_path):
    np_seg = []
    np_gth = []

    gth = np.array((nib.load(gth_path)).dataobj).transpose((2,1,0))

    for c, (elem_seg, elem_gth) in enumerate(zip(seg_pil, gth)):
        elem_gth_p = np.flipud(elem_gth)
        elem_gth_p = np.fliplr(elem_gth_p)

        elem_seg_p = rgb2parr(elem_seg.convert("RGB"))

        np_seg.append(elem_seg_p)
        np_gth.append(elem_gth_p)


    np_seg = np.array(np_seg).astype(np.uint8)
    np_gth = np.array(np_gth).astype(np.uint8)


    return np_seg, np_gth
def filt_seg_gth2(seg_path, gth_path):
    np_seg = []
    np_gth = []

    gth = np.array((nib.load(gth_path)).dataobj).transpose((2,1,0))
    seg = np.array((nib.load(seg_path)).dataobj).transpose((2,1,0))

    for c, (elem_seg, elem_gth) in enumerate(zip(seg, gth)):
        elem_gth_p = np.flipud(elem_gth)
        elem_gth_p = np.fliplr(elem_gth_p)

        elem_seg_p = np.flipud(elem_seg)
        elem_seg_p = np.fliplr(elem_seg_p)

        np_seg.append(elem_seg_p)
        np_gth.append(elem_gth_p)


    np_seg = np.array(np_seg).astype(np.uint8)
    np_gth = np.array(np_gth).astype(np.uint8)


    return np_seg, np_gth



def save_nii(nii_pno24_path, img_3d, nii_header):
    img_nii = []
    for img in img_3d:
        img = img.convert("RGB")
        img = rgb2parr(img)

        img = np.flipud(img)
        img = np.fliplr(img)

        img_nii.append(img)

    img_nii = np.array(img_nii)
    img_nii = img_nii.transpose((2, 1, 0))

    nii_image = nib.Nifti1Image(img_nii, affine=np.eye(4))

    nii_image.header['pixdim'] = [1, nii_header['pixdim'][1], nii_header['pixdim'][2], nii_header['pixdim'][3], 0, 0, 0, 0]
    nii_image.header['qform_code'] = 1
    # """
    nii_image.header['qoffset_x'] = float(nii_header['qoffset_x'])
    nii_image.header['qoffset_y'] = float(nii_header['qoffset_y'])
    nii_image.header['qoffset_z'] = float(nii_header['qoffset_z'])


    nib.save(nii_image, os.path.join(nii_pno24_path))



def save_seg_nii(pno_24, img_3d, nii_header, nii_save_dir, class_list):
    img_nii = []
    for img in img_3d:
        img = img.convert("RGB")
        img = rgb2parr(img)

        img = np.flipud(img)
        img = np.fliplr(img)

        img = img.astype(np.uint8)
        img_nii.append(img)

    img_nii = np.array(img_nii)
    img_nii = img_nii.transpose((2, 1, 0))

    nii_image = nib.Nifti1Image(img_nii, affine=np.eye(4))

    nii_image.header['pixdim'] = [1, nii_header['pixdim'][1], nii_header['pixdim'][2], nii_header['pixdim'][3], 0, 0, 0, 0]
    nii_image.header['qform_code'] = 1
    # """
    nii_image.header['qoffset_x'] = float(nii_header['qoffset_x'])
    nii_image.header['qoffset_y'] = float(nii_header['qoffset_y'])
    nii_image.header['qoffset_z'] = float(nii_header['qoffset_z'])


    nib.save(nii_image, os.path.join(nii_save_dir, 'seg'+pno_24+'.nii.gz'))

def save_gth_nii(pno_24, pno_gtnii_path, nii_header, nii_save_dir, class_list):
    img_3d = np.array((nib.load(pno_gtnii_path)).dataobj).transpose((2,1,0))


    img_nii = []
    for img in img_3d:
        img = img.astype(np.uint8)
        img_nii.append(img)

    img_nii = np.array(img_nii)
    img_nii = img_nii.transpose((2, 1, 0))

    nii_image = nib.Nifti1Image(img_nii, affine=np.eye(4))

    nii_image.header['pixdim'] = [1, nii_header['pixdim'][1], nii_header['pixdim'][2], nii_header['pixdim'][3], 0, 0, 0, 0]
    nii_image.header['qform_code'] = 1
    # """
    nii_image.header['qoffset_x'] = float(nii_header['qoffset_x'])
    nii_image.header['qoffset_y'] = float(nii_header['qoffset_y'])
    nii_image.header['qoffset_z'] = float(nii_header['qoffset_z'])


    nib.save(nii_image, os.path.join(nii_save_dir, 'gt'+pno_24+'.nii.gz'))



def save_seg_cls_nii(pno_24, img_3d, nii_header, nii_save_dir, class_list):
    nii_pno24_dir = os.path.join(nii_save_dir, pno_24)

    try:
        os.mkdir(nii_pno24_dir)
    except:
        pass


    img_nii = []
    for cnt, class_name in enumerate(class_list):
        if cnt:
            prefix = 'seg'+str(cnt)

            img_nii = []
            for img in img_3d:
                img = img.convert("RGB")
                img = rgb2parr_cls(img, cnt)

                img = np.flipud(img)
                img = np.fliplr(img)

                img_nii.append(img)

            img_nii = np.array(img_nii)
            img_nii = img_nii.transpose((2, 1, 0))

            nii_image = nib.Nifti1Image(img_nii, affine=np.eye(4))

            nii_image.header['pixdim'] = [1, nii_header['pixdim'][1], nii_header['pixdim'][2], nii_header['pixdim'][3], 0, 0, 0, 0]
            nii_image.header['qform_code'] = 1
            # """
            nii_image.header['qoffset_x'] = float(nii_header['qoffset_x'])
            nii_image.header['qoffset_y'] = float(nii_header['qoffset_y'])
            nii_image.header['qoffset_z'] = float(nii_header['qoffset_z'])


            nib.save(nii_image, os.path.join(nii_pno24_dir, prefix + pno_24 + '.nii.gz'))





def save_gt_cls_nii(pno_24, gt_path, nii_header, nii_save_dir, class_list):
    nii_pno24_dir = os.path.join(nii_save_dir, pno_24)
    nii_gth = np.array((nib.load(gt_path)).dataobj).transpose((2,1,0))

    try:
        os.mkdir(nii_pno24_dir)
    except:
        pass


    img_nii = []
    for cnt, class_name in enumerate(class_list):
        if cnt:
            prefix = 'gt'+str(cnt)

            img_nii = []
            for img_gt in nii_gth:
                img = (img_gt == cnt)*cnt
                img = img.astype('uint8')
                img_nii.append(img)

            img_nii = np.array(img_nii)
            img_nii = img_nii.transpose((2, 1, 0))

            nii_image = nib.Nifti1Image(img_nii, affine=np.eye(4))

            nii_image.header['pixdim'] = [1, nii_header['pixdim'][1], nii_header['pixdim'][2], nii_header['pixdim'][3], 0, 0, 0, 0]
            nii_image.header['qform_code'] = 1
            # """
            nii_image.header['qoffset_x'] = float(nii_header['qoffset_x'])
            nii_image.header['qoffset_y'] = float(nii_header['qoffset_y'])
            nii_image.header['qoffset_z'] = float(nii_header['qoffset_z'])


            nib.save(nii_image, os.path.join(nii_pno24_dir, prefix + pno_24 + '.nii.gz'))





def save_seg_1cls_nii(pno_24, img_3d, nii_header, nii_save_dir, class_list):
    nii_pno24_dir = os.path.join(nii_save_dir, pno_24)

    try:
        os.mkdir(nii_pno24_dir)
    except:
        pass


    img_nii = []

    prefix = 'segone'

    img_nii = []
    for img in img_3d:
        img = img.convert("RGB")
        img = rgb2parr(img)
        img = (img != 0)*1
        img = img.astype('uint8')

        img = np.flipud(img)
        img = np.fliplr(img)

        img_nii.append(img)

    img_nii = np.array(img_nii)
    img_nii = img_nii.transpose((2, 1, 0))

    nii_image = nib.Nifti1Image(img_nii, affine=np.eye(4))

    nii_image.header['pixdim'] = [1, nii_header['pixdim'][1], nii_header['pixdim'][2], nii_header['pixdim'][3], 0, 0, 0, 0]
    nii_image.header['qform_code'] = 1
    # """
    nii_image.header['qoffset_x'] = float(nii_header['qoffset_x'])
    nii_image.header['qoffset_y'] = float(nii_header['qoffset_y'])
    nii_image.header['qoffset_z'] = float(nii_header['qoffset_z'])


    nib.save(nii_image, os.path.join(nii_pno24_dir, prefix + pno_24 + '.nii.gz'))

def save_gt_1cls_nii(pno_24, gt_path, nii_header, nii_save_dir, class_list):
    nii_pno24_dir = os.path.join(nii_save_dir, pno_24)
    nii_gth = np.array((nib.load(gt_path)).dataobj).transpose((2,1,0))

    try:
        os.mkdir(nii_pno24_dir)
    except:
        pass


    img_nii = []

    prefix = 'gtone'

    img_nii = []
    for img_gt in nii_gth:
        img = (img_gt != 0)*1
        img = img.astype('uint8')
        img_nii.append(img)

    img_nii = np.array(img_nii)
    img_nii = img_nii.transpose((2, 1, 0))

    nii_image = nib.Nifti1Image(img_nii, affine=np.eye(4))

    nii_image.header['pixdim'] = [1, nii_header['pixdim'][1], nii_header['pixdim'][2], nii_header['pixdim'][3], 0, 0, 0, 0]
    nii_image.header['qform_code'] = 1
    # """
    nii_image.header['qoffset_x'] = float(nii_header['qoffset_x'])
    nii_image.header['qoffset_y'] = float(nii_header['qoffset_y'])
    nii_image.header['qoffset_z'] = float(nii_header['qoffset_z'])


    nib.save(nii_image, os.path.join(nii_pno24_dir, prefix + pno_24 + '.nii.gz'))
def print_log(log_name, log_list, pno_24, overlay_save_dir):

    txt_path = os.path.join(overlay_save_dir, pno_24, log_name+'.txt')
    file = open(txt_path, 'a')

    for line in log_list:
        file.writelines(str(line)+'\n')
