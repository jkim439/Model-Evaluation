__author__ = 'Junghwan Kim'
__copyright__ = 'Copyright 2016-2019 Junghwan Kim. All Rights Reserved.'
__version__ = '1.0.0'

import os
import numpy as np
import glob
from scipy import ndimage

import shutil

import nibabel as nib
import sys
sys.path.append('/home/jkim/evaluation/segmentation')
import util
palette_path = '/home/jkim/evaluation/segmentation/palette.npy'
palette = np.load(palette_path)
reshape_pal = palette.reshape(256, 3)

label_dict = {}
for c, label in enumerate(reshape_pal):
    label_dict[tuple(label)] = c



class MorphologyOps(object):
    """
    Class that performs the morphological operations needed to get notably
    connected component. To be used in the evaluation
    """

    def __init__(self, binary_img, neigh):
        assert len(binary_img.shape) == 3, 'currently supports 3d inputs only'
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.neigh = neigh

    def border_map(self):
        """
        Creates the border for a 3D image
        :return:
        """
        west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
        east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
        north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
        south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
        top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
        bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
        cumulative = west + east + north + south + top + bottom
        border = ((cumulative < 6) * self.binary_map) == 1
        return border

    def foreground_component(self):
        return ndimage.label(self.binary_map)


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

def hausdorff_dist(seg, ref, class_no):

    debug = 0

    seg = (seg==class_no)
    ref = (ref==class_no)

    seg = seg.reshape(list(seg.shape)+[1,1])
    ref = ref.reshape(list(ref.shape)+[1,1])



    debug = 0

    def borders(seg, ref, neigh=8):
        """
        This function determines the points that lie on the border of the
        inferred and reference segmentations
        :param seg: numpy array with binary mask from inferred segmentation
        :param ref: numpy array with binary mask from reference segmentation
        :param neigh: connectivity 4 or 8
        :return: numpy arrays of reference and inferred segmentation borders
        """
        border_ref = MorphologyOps(ref[:, :, :, 0, 0], neigh).border_map()
        border_seg = MorphologyOps(seg[:, :, :, 0, 0], neigh).border_map()
        return border_ref, border_seg

    def border_distance(seg, ref, neigh=8):
        """
        This functions determines the distance at each seg border point to the
        nearest ref border point and vice versa
        :param seg: numpy array with binary mask from inferred segmentation
        :param ref: numpy array with binary mask from reference segmentation
        :param neigh: connectivity 4 or 8
        :return: numpy arrays for distance_from_ref_border, distance_from
        seg_border
        """
        border_ref, border_seg = borders(seg, ref, neigh)
        distance_ref = ndimage.distance_transform_edt(1 - border_ref)
        distance_seg = ndimage.distance_transform_edt(1 - border_seg)
        distance_border_seg = border_ref * distance_seg
        distance_border_ref = border_seg * distance_ref
        return distance_border_ref, distance_border_seg


    ref_border_dist, seg_border_dist = border_distance(seg, ref, 8)
    return np.max([np.max(ref_border_dist), np.max(seg_border_dist)])



def eval_seg(input_pred, input_gth, classes, dim_cfg=2):
    debug = 0

    nii_gth = np.array((nib.load(input_gth)).dataobj).transpose((2,1,0))

    dice_value=[]
    jacc_value=[]
    for elem_pred, elem_gth in zip(input_pred, nii_gth):
        elem_gth = np.flipud(elem_gth)
        elem_gth = np.fliplr(elem_gth)

        elem_pred_p = rgb2parr(elem_pred.convert("RGB"))

        elem_dice_name = []
        elem_dice_value = []
        elem_jacc_value = []

        for cnt, class_name in enumerate(classes):
            class_name = class_name[:-1]


            if cnt:
                a_l = np.sum((elem_pred_p == cnt))
                a_g = np.sum((elem_gth == cnt))

                a_lg = np.sum((elem_pred_p == cnt) * (elem_gth == cnt))
                s_lg = np.sum((elem_pred_p == cnt) + (elem_gth == cnt))
                if (a_l + a_g) == 0:
                    dice = '-'
                    jacc = '-'
                elif a_g == 0:
                    dice = 'noGTH'
                    jacc = 'noGTH'
                else:
                    dice = 2. * a_lg / (a_l + a_g)
                    jacc = 1. * a_lg / s_lg

                elem_dice_value.append(dice)
                elem_jacc_value.append(jacc)


        dice_value.append(elem_dice_value)
        jacc_value.append(elem_jacc_value)

        #print elem_dice_value

    return dice_value, jacc_value

def eval_seg_3d(np_seg, np_gth, classes, dim_cfg=2):
    debug = 0

    dice_value=[]
    jacc_value=[]


    cum_TP = 0
    cum_FP = 0
    cum_TN = 0
    cum_FN = 0

    cum_a_l = 0
    cum_a_g = 0

    cnt_value_1 = 0
    dice_value_1=0
    jacc_value_1=0

    cnt_value_2 = 0
    dice_value_2=0
    jacc_value_2=0

    dice_tot = [0,0]
    jacc_tot = [0,0]
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    cum_cf_m = np.zeros((len(classes),len(classes)))
    for c, (elem_seg, elem_gth) in enumerate(zip(np_seg, np_gth)):
        elem_dice_name = []
        elem_dice_value = []
        elem_jacc_value = []

        cf_m = np.array(confusion_matrix(elem_gth, elem_seg, classes))
        cum_cf_m = cum_cf_m + cf_m

        dice_value.append(elem_dice_value)
        jacc_value.append(elem_jacc_value)

        #print elem_dice_value


    res = get_res_from_mat(cum_cf_m, gth_option=0)

    #calc_eval(cum_cf_m)
    #calc_eval2(res)
    debug = 0

    return res

def confusion_matrix(gth, seg, class_list):

    M = []
    dewbug = 0
    for act_cnt, act_name in enumerate(class_list):

        m = []
        gth_list = [1,2]
        for pred_cnt, pred_name in enumerate(class_list):
            if pred_cnt == act_cnt:
                a_tp = np.sum((seg == pred_cnt) * (gth == act_cnt))
                # print a_tp
                m.append(a_tp)
            else:
                a_fn = np.sum((gth == act_cnt) * (seg == pred_cnt))
                # print a_fp
                m.append(a_fn)
        M.append(m)


    return M

def get_res_from_mat(M, gth_option=0):
    res = []
    for class_cnt, class_name in enumerate(M):
        tp = M[class_cnt][class_cnt]
        fp = 0
        fn = 0
        sum_M_fn = 0

        if class_cnt:
            M_fn = np.delete(M, class_cnt, 0)
            M_fn = np.delete(M_fn, 0, 0)

            sum_M_fn = np.sum(M_fn)


        for class_e_cnt, class_e_name in enumerate(M):
            if class_e_cnt != class_cnt:
                e_fp = M[class_e_cnt][class_cnt]
                e_fn = M[class_cnt][class_e_cnt]

                fp = fp + e_fp
                fn = fn + e_fn

        #"""
        if gth_option:
            fn = fn+sum_M_fn
            tn = np.sum(M) - tp - fp - fn
            res_arr = [tp, fp, fn, tn]

            if not class_cnt:
                res_arr = [fp, tp, tn, fn]


        else:
            tn = np.sum(M) - tp - fp - fn
            res_arr = [tp, fp, fn, tn]
        #"""




        res.append(res_arr)

        #print(str(class_cnt) + "\t" + "tp=" + str(tp) + " fp=" + str(fp) + " fn=" + str(fn) + " tn=" + str(tn))

    return res


def calc_eval(cnf_matrix):
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    debug = 0
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    print ("TPR = "+str(TPR))
    print ("TNR = "+str(TNR))
    print ("PPV = "+str(PPV))
    print ("NPV = "+str(NPV))
    print ("FPR = "+str(FPR))
    print ("FNR = "+str(FNR))
    print ("FDR = "+str(FDR))
    print ("ACC = "+str(ACC))


def calc_eval2(res):

    TP=[]
    FP=[]
    FN=[]
    TN=[]

    for elem_res in res:
        TP.append(elem_res[0])
        FP.append(elem_res[1])
        FN.append(elem_res[2])
        TN.append(elem_res[3])

    TP=np.array(TP).astype(float)
    FP=np.array(FP).astype(float)
    FN=np.array(FN).astype(float)
    TN=np.array(TN).astype(float)


    debug = 0
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    print ("TPR = "+str(TPR))
    print ("TNR = "+str(TNR))
    print ("PPV = "+str(PPV))
    print ("NPV = "+str(NPV))
    print ("FPR = "+str(FPR))
    print ("FNR = "+str(FNR))
    print ("FDR = "+str(FDR))
    print ("ACC = "+str(ACC))


def main():

    input_path = "/home/jkim/evaluation/segmentation/testlist.txt"
    model_path = "/home/jkim/evaluation/segmentation/model/1"

    modelName = glob.glob(os.path.join(model_path, '*.caffemodel'))[0]
    modelProt = os.path.join(model_path, 'deploy.prototxt')
    modelClass = os.path.join(model_path, 'classes.txt')
    modelMean = os.path.join(model_path, 'mean.binaryproto')

    # Net = testnet.caftestnet(True,modelName,modelProt,modelClass,modelMean,gpuno=0)

    NAS_USERID = "lp-002"

    file = open(input_path, "r")
    pno_dir_list = file.readlines()

    file_class = open(modelClass, "r")
    class_list = file_class.readlines()
    # class_list=['BGD', 'CX']

    dice_tot_1 = 0
    dice_tot_2 = 0

    cum_res = np.zeros((len(class_list), 4))
    cnt = 0
    cum_np_seg = None
    cum_np_gth = None

    for pno_dir_path in pno_dir_list:
        pno_dir_path = pno_dir_path[:-1]

        pno_24 = pno_dir_path.split('/')[-4] + \
                 pno_dir_path.split('/')[-3] + \
                 pno_dir_path.split('/')[-2] + \
                 pno_dir_path.split('/')[-1]

        # nii_fill = vol_inference.run_vol_inference(Net,pno_dir_path,None, map_no = 0)

        pno_gtnii_path = os.path.join(pno_dir_path, 'nifti', 'gt' + pno_24 + '.nii.gz')
        pno_segnii_path = os.path.join(pno_dir_path, 'nifti', 'seg' + pno_24 + '.nii.gz')

        pno_dcmnii_path = os.path.join(pno_dir_path, 'nifti', pno_24 + '.nii.gz')
        nii_header = nib.load(pno_dcmnii_path).header
        # np_seg, np_gth = util.filt_seg_gth2(tmp_seg, tmp_gth)

        # np_seg, np_gth = util.filt_seg_gth(nii_fill, pno_gtnii_path)
        np_seg, np_gth = util.filt_seg_gth2(pno_segnii_path, pno_gtnii_path)

        if cum_np_gth is None and cum_np_gth is None:
            cum_np_seg = np_seg
            cum_np_gth = np_gth
        else:
            cum_np_seg = np.append(cum_np_seg, np_seg, axis=0)
            cum_np_gth = np.append(cum_np_gth, np_gth, axis=0)

        # res = eval_seg_3d(np_seg, np_gth, class_list, dim_cfg=2)
        # cum_res = cum_res+np.array(res)

    cum_res = eval_seg_3d(cum_np_seg, cum_np_gth, class_list, dim_cfg=2)

    hdrf_list = []
    for cls_no, cls_name in enumerate(class_list):
        print("Get hausdorff_dist")
        hdrf = hausdorff_dist(cum_np_seg, cum_np_gth, cls_no)
        hdrf_list.append(hdrf)

    ### RESULT ###

    print("cls\t" + str("precision").ljust(18, ' ') + "\t"
          + str("recall").ljust(18, ' ') + "\t"
          + str("f1score").ljust(18, ' ') + "\t"
          + str("gthvol").ljust(18, ' ') + "\t"
          + str("segvol").ljust(18, ' ') + "\t"
          + str("hdrf_dist").ljust(18, ' ') + "\t"

          + str("dice").ljust(18, ' '))

    for cls_no, res_elem in enumerate(cum_res):
        precision = res_elem[0] / (res_elem[0] + res_elem[1])
        recall = res_elem[0] / (res_elem[0] + res_elem[2])
        f1score = 2. * (precision * recall) / (precision + recall)
        gthvol = res_elem[0] + res_elem[2]
        segvol = res_elem[0] + res_elem[1]
        dice = 2. * res_elem[0] / (2. * res_elem[0] + res_elem[1] + res_elem[2])

        print(str(cls_no)
              + "\t" + str(precision).ljust(18, ' ')
              + "\t" + str(recall).ljust(18, ' ')
              + "\t" + str(f1score).ljust(18, ' ')
              + "\t" + str(gthvol).ljust(18, ' ')
              + "\t" + str(segvol).ljust(18, ' ')
              + "\t" + str(hdrf_list[cls_no]).ljust(18, ' ')

              + "\t" + str(dice).ljust(18, ' ')
              )
    cnt = cnt + 1


if __name__ == '__main__':
    main()