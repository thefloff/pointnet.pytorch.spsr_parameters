from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import json
from pointnet.dataset import ShapeNetDataset, ModelNetDataset, ABCDataset
from pointnet.model import PointNetParam, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
from parameter_optimization.src.parameter_space_optimization.targets.poisson_reconstruction import PoissonReconstruction
from parameter_optimization.src.parameter_space_optimization.parameter_space_optimization import test_single
from visualize import visualize


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'abc':
    dataset = ABCDataset(
        root=opt.dataset,
        split='train')
    test_dataset = ABCDataset(
        root=opt.dataset,
        split='test')
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print("train: ", len(dataset), "  test: ", len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetParam()

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.000001, betas=(0.95, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

default_config = {
    "cgDepth": 0,
    "confidence": False,
    "depth": 5,
    "fullDepth": 5,
    "iters": 8,
    "pointWeight": 4.0,
    "preClean": False,
    "samplesPerNode": 1.5,
    "scale": 1.1,
    "visibleLayer": False,
    "id": "default"
}

loss = torch.nn.MSELoss() 
for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target, name = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        output = loss(pred, target)
        if opt.feature_transform:
            output += feature_transform_regularizer(trans_feat) * 0.001 
        output.backward()
        optimizer.step()
        # pred_choice = pred.data.max(1)[1]
        # correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, output.item()))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target, name = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = torch.nn.MSELoss()
            output = loss(pred, target)
            # pred_choice = pred.data.max(1)[1]
            # correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), output.item()))

    scheduler.step()
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

print()

all_results = []

improvements = []
off_from_targets = []
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target, name = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)

    for i in range(target.shape[0]):
        cfg_target = target[i].tolist()
        cfg_target = ABCDataset.un_norm(cfg_target)
        target_config = {
            "cgDepth": round(cfg_target[0]),
            "confidence": False,
            "depth": round(cfg_target[1]),
            "fullDepth": round(cfg_target[2]),
            "iters": round(cfg_target[3]),
            "pointWeight": cfg_target[4],
            "preClean": False,
            "samplesPerNode": cfg_target[5],
            "scale": cfg_target[6],
            "visibleLayer": False,
            "id": name[i]
        }
        cfg_pred = pred[i].tolist()
        cfg_pred = ABCDataset.un_norm(cfg_pred)
        pred_config = {
            "cgDepth": round(cfg_pred[0]),
            "confidence": False,
            "depth": round(cfg_pred[1]),
            "fullDepth": round(cfg_pred[2]),
            "iters": round(cfg_pred[3]),
            "pointWeight": cfg_pred[4],
            "preClean": False,
            "samplesPerNode": cfg_pred[5],
            "scale": cfg_pred[6],
            "visibleLayer": False,
            "id": name[i]
        }
        default_config["id"] = name[i]

        all_results.append({
            'gt': target_config,
            'pred': pred_config
        })

        # spsr = PoissonReconstruction(
        #     base_dir = opt.dataset,
        #     dataset_dir = "abc",
        #     ref_mesh_dir = '03_meshes',
        #     dir_in_pointcloud = '04_pts',
        #     normals_dir = '06_normals',
        #     pts_dir = '06_normals/pts',
        #     recon_mesh_dir = '06_poisson_rec_gt_normals',
        #     num_processes = int(opt.workers),
        #     dataset = name[i]
        # )

        # result_default = test_single(spsr, default_config)
        # result_target = test_single(spsr, target_config)
        # result_pred = test_single(spsr, pred_config)
        # improvement = (1 - float(result_pred[0]["value"]) / float(result_default[0]["value"])) * 100
        # off_from_target = abs(float(result_pred[0]["value"]) / float(result_target[0]["value"]) - 1) * 100

        # print('[{:d} {}] improvement over default: {:.1f}%, off from target by {:.1f}% - default: {:.2f} ground truth: {:.2f} from NN: {:.2f}'.format(i, name[i], improvement, off_from_target, float(result_default[0]["value"]), float(result_target[0]["value"]), float(result_pred[0]["value"])))

        # improvements.append(improvement)
        # off_from_targets.append(off_from_target)


with open("all_results.json", "w") as outfile:
    json.dump(all_results, outfile)

visualize(all_results)

# print("avg improvement over default: {:.1f}%".format(sum(improvements) / len(improvements)))
# print("avg distance to gt: {:.1f}%".format(sum(off_from_targets) / len(off_from_targets)))