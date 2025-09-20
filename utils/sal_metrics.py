import torch
import cv2
import numpy as np
from audtorch.metrics.functional import pearsonr
import torch.distributed as dist
from enum import Enum

def CC(pred: torch.Tensor, gt: torch.Tensor, eps=1e-7):
    a = pearsonr(pred.flatten(), gt.flatten())
    a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    a = a.mean()
    pearson = float(a)
    return pearson


def KLDivergence(pred: torch.Tensor, gt: torch.Tensor, eps=1e-7):
    P = pred
    P = P / (eps + torch.sum(P, dim=[1, 2, 3], keepdim=True))
    Q = gt
    Q = Q / (eps + torch.sum(Q, dim=[1, 2, 3], keepdim=True))

    R = Q * torch.log(eps + Q / (eps + P))
    R = R.sum()
    kld = float(R)
    return kld

def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    h = s_map.size(1)
    w = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, h, w)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, h, w)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map

def SIM(s_map, gt):
    ''' For single image metric
        Size of Image - WxH or 1xWxH
        gt is ground truth saliency map
    '''
    s_map = s_map.squeeze(1)
    gt = gt.squeeze(1)
    batch_size = s_map.size(0)
    h = s_map.size(1)
    w = s_map.size(2)

    s_map_norm = normalize_map(s_map)
    gt_norm = normalize_map(gt)

    sum_s_map = torch.sum(s_map_norm.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, h, w)

    assert expand_s_map.size() == s_map_norm.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, h, w)

    s_map_norm = s_map_norm / (expand_s_map * 1.0)
    gt_norm = gt / (expand_gt * 1.0)

    s_map_norm = s_map_norm.view(batch_size, -1)
    gt_norm = gt_norm.view(batch_size, -1)
    # return torch.mean(torch.sum(torch.min(s_map, gt), 1))
    return torch.sum(torch.min(s_map_norm, gt_norm), 1)


def discretize_gt(gt, threshold=0.7):
    gt = gt.astype(np.float32)
    epsilon = 1e-6
    binary_gt = np.where(gt >= threshold - epsilon, 1.0, 0.0)
    assert np.isin(binary_gt, [0, 1]).all(), "discretize error"
    return binary_gt


def AUC_J(s_map, gt):
    s_map = normalize_map(s_map.squeeze(1))
    s_map = s_map[0].cpu().detach().numpy()
    gt = normalize_map(gt.squeeze(1))
    gt = gt[0].cpu().detach().numpy()
    # gt = gt[0].squeeze(0).cpu().detach().numpy()
    # ground truth is discrete, s_map is continous and normalized
    gt = discretize_gt(gt)
    # thresholds are calculated from the salience map, only at places where fixations are present
    thresholds = []
    for i in range(0, gt.shape[0]):
        for k in range(0, gt.shape[1]):
            if gt[i][k] > 0:
                thresholds.append(s_map[i][k])

    num_fixations = np.sum(gt)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    # fp_list = []
    # tp_list = []
    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map, keep only those pixels with values above threshold
        temp = np.zeros(s_map.shape)
        temp[s_map >= thresh] = 1.0
        assert np.max(gt) == 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
        assert np.max(s_map) == 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
        num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / ((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)

        area.append((round(tp, 4), round(fp, 4)))
    # tp_list.append(tp)
    # fp_list.append(fp)

    # tp_list.reverse()
    # fp_list.reverse()
    area.append((1.0, 1.0))
    # tp_list.append(1.0)
    # fp_list.append(1.0)
    # print tp_list
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))


def AUC_B(s_map, gt, splits=100, stepsize=0.1):
    s_map = normalize_map(s_map.squeeze(1))
    s_map = s_map[0].cpu().detach().numpy()
    gt = normalize_map(gt.squeeze(1))
    gt = gt[0].cpu().detach().numpy()
    # gt = gt[0].squeeze(0).cpu().detach().numpy()
    gt = discretize_gt(gt)
    num_fixations = np.sum(gt)

    num_pixels = s_map.shape[0] * s_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        for k in range(0, int(num_fixations)):
            temp_list.append(np.random.randint(num_pixels))
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k % s_map.shape[0] - 1, k // s_map.shape[0]])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


def AUC_S(s_map, gt, other_map, splits=100, stepsize=0.1):
    # gt = discretize_gt(gt)
    # other_map = discretize_gt(other_map)

    num_fixations = np.sum(gt)

    x, y = np.where(other_map == 1)
    other_map_fixs = []
    for j in zip(x, y):
        other_map_fixs.append(j[0] * other_map.shape[0] + j[1])
    ind = len(other_map_fixs)
    assert ind == np.sum(other_map), 'something is wrong in auc shuffle'

    num_fixations_other = min(ind, num_fixations)

    num_pixels = s_map.shape[0] * s_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        t1 = np.random.permutation(ind)
        for k in t1:
            temp_list.append(other_map_fixs[k])
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k % s_map.shape[0] - 1, k / s_map.shape[0]])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


def NSS(s_map, gt):
    s_map = s_map[0].squeeze(0).cpu().detach().numpy()
    gt = gt[0].squeeze(0).cpu().detach().numpy()
    gt_ = discretize_gt(gt)
    s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)

    x, y = np.where(gt_ == 1)
    temp = []
    for i in zip(x, y):
        temp.append(s_map_norm[i[0], i[1]])
    # if np.isnan(np.mean(temp)):
    #     print('Warning: NaN in NSS')
    return np.mean(temp)
    # MAP = (s_map - s_map.mean()) / (s_map.std())
    # mask = gt.astype(np.bool_)
    #
    # score =  MAP[mask].mean()
    # return score



class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        # dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)