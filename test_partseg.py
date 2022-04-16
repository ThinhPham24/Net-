"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import json
import h5py


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Orchid': [41, 42, 43],
#               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
 #              'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
  #             'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_classes = {'Orchid': [41, 42, 43]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
savepath ="/home/airlab/Desktop/net++/output"

#change code
with open('./part_color_mapping.json', 'r') as f:
    color = json.load(f)

for c in color:
    c[0] = int(c[0]*255)
    c[1] = int(c[1]*255)
    c[2] = int(c[2]*255)


def write_ply(savepath, nameoutput, point1, label):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    with open(os.path.join(savepath + '/' + '{}.ply'.format(nameoutput)), "w") as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment 2048\n')
        f.write('element vertex %d\n' % 2048)
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float nx\n')
        f.write('property float ny\n')
        f.write('property float nz\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property uchar label\n')
        f.write('end_header\n')

        for i in range(len(point1[0])):
            if (label[:][i] == 41):
                f.write("".join(
                    '{} {} {} {} {} {} {} {} {} {}'.format(point1[0][i][0], point1[0][i][1], point1[0][i][2],
                                                               point1[0][i][3], point1[0][i][4], point1[0][i][5], 255, 0, 0,
                                                               label[:][i])) + "\n")
            elif (label[:][i] == 42):
                f.write("".join(
                    '{} {} {} {} {} {} {} {} {} {}'.format(point1[0][i][0], point1[0][i][1], point1[0][i][2],
                                                               point1[0][i][3], point1[0][i][4], point1[0][i][5], 0,
                                                               255, 0, label[:][i])) + "\n")
            else:
                f.write("".join(
                    '{} {} {} {} {} {} {} {} {} {}'.format(point1[0][i][0], point1[0][i][1], point1[0][i][2],
                                                               point1[0][i][3], point1[0][i][4], point1[0][i][5], 0, 0, 255,
                                                               label[:][i])) + "\n")
        f.close()
def write_ply1(savepath, nameoutput, point1, label):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    with open(os.path.join(savepath + '/' + '{}.ply'.format(nameoutput)), "w") as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment 2048\n')
        f.write('element vertex %d\n' % 2048)
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float nx\n')
        f.write('property float ny\n')
        f.write('property float nz\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property uchar label\n')
        f.write('end_header\n')
        #label1=label.transpose()
        #print(label[4])
        #print(point1.shape)
        for i in range(len(point1)):
            if (label[i] == 41):
                f.write("".join(
                    '{} {} {} {} {} {} {} {} {} {}'.format(point1[i][0], point1[i][1], point1[i][2],
                                                               point1[i][3], point1[i][4], point1[i][5], 255, 0, 0,
                                                               label[i])) + "\n")
            elif (label[i] == 42):
                f.write("".join(
                    '{} {} {} {} {} {} {} {} {} {}'.format(point1[i][0], point1[i][1], point1[i][2],
                                                               point1[i][3], point1[i][4], point1[i][5], 0,
                                                               255, 0, label[i])) + "\n")
            else:
                f.write("".join(
                    '{} {} {} {} {} {} {} {} {} {}'.format(point1[i][0], point1[i][1], point1[i][2],
                                                               point1[i][3], point1[i][4], point1[i][5], 0, 0, 255,
                                                               label[i])) + "\n")
        f.close()

for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals') #fix
    parser.add_argument('--num_votes', type=int, default=1, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = '/home/airlab/Desktop/net++/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    #print('test',TEST_DATASET.__getitem__(0))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=1) #fix here original is num_work=4
    #print(testDataLoader)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 16# fix here 16
    num_part = 50 #fix here

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            #print('points', points)
            #print('batch',batch_id)
            #print(len(target))
            point1 = np.array(points)
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            #print('pointor', points)
            points = points.transpose(2, 1)
            #print(point1.shape)
            #print('trans',point1[0]) #fix here
            #print('pointor', points)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred
            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            #print(cur_pred_val[:,1])
            cur_pred_val_logits = cur_pred_val
            #print('cur_pred_val_logits',cur_pred_val_logits)
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()
            targetbuff = target.transpose()
            #print(targetbuff)
            #print(target)

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            #print(cur_pred_val)
            label_pred = cur_pred_val.transpose()
            #print(label_pred.shape)
            correct = np.sum(cur_pred_val == target)
            print("Points is correct:", correct)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)
            #print(cur_batch_size)
            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))
        # change code to write  Groud-truth ply files:
        #write_ply(savepath, "GT_pc3", point1, targetbuff)

        # change code to write  predict ply files:
       # write_ply(savepath, "Pred_pc3", point1, label_pred)
        #fix here
            #print(point1[0])
            #print(label_pred[:,0])
            temp = batch_id*4
            if len(target)==1:
                write_ply(savepath, "GT_pc{}".format(batch_id), point1, targetbuff)
                write_ply(savepath, "Pred_pc{}".format(batch_id), point1, label_pred)
            if len(target)!=1:
                for li in range(len(target)):
                    write_ply1(savepath, "GT_pc{}".format(temp), point1[li], targetbuff[:,li])
                    write_ply1(savepath, "Pred_pc{}".format(temp), point1[li], label_pred[:,li])
                    temp = temp + li


        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
