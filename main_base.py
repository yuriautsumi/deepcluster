# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import shutil
import pickle
import time

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA #Principal Component Analysis
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
plt.rcParams["legend.loc"] = "upper right"

import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler
from cartoon_dataset import CartoonDataset
from time_cartoon_dataset import TimeCartoonDataset

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--experiment', default="baseline", help='type of experiment, choose from: baseline (default), trivial_time')
    parser.add_argument('--weight_time', type=float, default=1.0,
                        help='weight on using time info in clustering')
    parser.add_argument('--labels', default=True, type=bool, help='True if labels can be parsed from data directory. False otherwise.')
    parser.add_argument('--path_file', metavar='PATH_FILE', help='path to file')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16'], default='alexnet',
                        help='CNN architecture (default: alexnet)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    return parser.parse_args()

def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # define results folder
    data_id = args.data.split('/')[-1]
    DATA_RESULTS_DIR = os.path.join(args.exp, data_id)
    RESULTS_DIR = os.path.join(DATA_RESULTS_DIR, f'{args.experiment}_{args.arch}_{args.clustering}{args.nmb_cluster}_epoch{args.epochs}_lr{args.lr}')
    if args.experiment == "trivial_time":
        RESULTS_DIR = RESULTS_DIR + f"_wt{args.weight_time}"
    if os.path.isdir(RESULTS_DIR):
        i = 1
        while os.path.isdir(f"{RESULTS_DIR}_v{i}"):
            i += 1
        RESULTS_DIR = f"{RESULTS_DIR}_v{i}"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    IMAGES_DIR = os.path.join(RESULTS_DIR, 'clustered_images')
    os.makedirs(IMAGES_DIR, exist_ok=True)

    PCA_PLOT_DIR = os.path.join(RESULTS_DIR, 'pca_plots')
    os.makedirs(PCA_PLOT_DIR, exist_ok=True)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=args.sobel)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(RESULTS_DIR, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(RESULTS_DIR, 'clusters'))

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    if args.experiment == "baseline":
        dataset = CartoonDataset(args.data, args.path_file, transform=transforms.Compose(tra))
    elif args.experiment == "trivial_time":
        dataset = TimeCartoonDataset(args.data, args.path_file, transform=transforms.Compose(tra))
        max_frame_num = dataset.max_frame_num
        max_shot_num = dataset.max_shot_num
    else:
        print(f"##### Experiment type {args.experiment} not supported! Defaulting to baseline. #####")
        dataset = CartoonDataset(args.data, args.path_file, transform=transforms.Compose(tra))

    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    # training convnet with DeepCluster
    METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(METRICS_FILE, 'a') as f:
        f.write("epoch, clustering_loss, convnet_loss, silhouette_score, nmi_previous, nmi_gt\n")
    clustering_losses, convnet_losses, nmi_prevs, nmi_gts, silhouette_scores = [], [], [], [], []
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset))
        """
        if args.experiment == "baseline":
            features = compute_features(dataloader, model, len(dataset))
        elif args.experiment == "trivial_time":
            features = compute_features(dataloader, model, len(dataset), max_shot_num, max_frame_num)
        else: # default to baseline features
            features = compute_features(dataloader, model, len(dataset))
        """

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                  dataset.imgs)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, criterion, optimizer, epoch, RESULTS_DIR)

        # get NMI metric between pseudo and gt labels
        def _get_scene_num(ds_img_tuple):
            path = ds_img_tuple[0]
            shot_scene = path.split('/')[-2]
            scene_str = shot_scene.split('_')[-1]
            scene_id = int(scene_str[5:])
            return scene_id

        if args.labels: # do computation only if labels are provided
            gt_labels = list(map(_get_scene_num, dataset.imgs)) # gt_labels <=> dataset.imgs (filepaths)

            cluster_labels = np.zeros(len(gt_labels))
            for cluster_id, img_list in enumerate(deepcluster.images_lists):
                for img_idx in img_list:
                    cluster_labels[img_idx] = cluster_id

        # compute silhouette score
        sorted_features, sorted_cluster_labels = None, None
        for cluster, images in enumerate(deepcluster.images_lists):
            current_features = features[images,:]
            current_labels = np.ones(len(images))*cluster
            if sorted_features is None:
                sorted_features = current_features
                sorted_cluster_labels = current_labels
            else:
                sorted_features = np.vstack((sorted_features, current_features))
                sorted_cluster_labels = np.concatenate((sorted_cluster_labels, current_labels))
        sil_score = silhouette_score(sorted_features, sorted_cluster_labels)

        # print log
        nmi, labels_nmi = None, None
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}\n'
                  'Silhouette score: {4:.3f}'
                  .format(epoch, time.time() - end, clustering_loss, loss, sil_score))
            try:
                # compute NMI against previous assignment
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.5f}'.format(nmi))

                # compute NMI against ground truth (if provided)
                if args.labels:
                    labels_nmi = normalized_mutual_info_score(cluster_labels, gt_labels)
                    print("NMI against ground truth: {0:.5f}".format(labels_nmi))

            except IndexError:
                pass
            print('####################### \n')

        # save metrics 
        with open(METRICS_FILE, 'a') as f:
            f.write(f"{epoch}, {clustering_loss}, {loss}, {sil_score}, {nmi}, {labels_nmi}\n")
        clustering_losses.append(clustering_loss)
        convnet_losses.append(loss)
        silhouette_scores.append(sil_score)
        nmi_prevs.append(nmi)
        nmi_gts.append(labels_nmi)

        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                    os.path.join(RESULTS_DIR, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

        # Save clustered images 
        def save_cluster_images(epoch):
            CLUSTERED_IMAGES_DIR = os.path.join(IMAGES_DIR, f'epoch={epoch}')
            os.makedirs(CLUSTERED_IMAGES_DIR, exist_ok=True)
            for clusts in range((args.nmb_cluster + 1)):
                CLUSTS_DIR = os.path.join(CLUSTERED_IMAGES_DIR, f"{clusts}")
                os.makedirs(CLUSTS_DIR, exist_ok=True)
            for clust_lbl, gt_path in zip(cluster_labels, dataset.imgs):
                gt_split = gt_path[0].split('/')
                img_name = "_".join(gt_split[-3:])
                CLUSTS_DIR = os.path.join(CLUSTERED_IMAGES_DIR, f"{int(clust_lbl)}")
                shutil.copy(gt_path[0], os.path.join(CLUSTS_DIR, img_name))

                with open(os.path.join(CLUSTS_DIR, "meta.txt"), 'a') as f:
                    f.write(f'{gt_path[0]}\n')

        # only save for first, last, and every 50 epochs
        if (epoch == 0 or (epoch+1)%50 == 0) or (epoch+1 == args.epochs):
            save_cluster_images(epoch)

        # Scatter plot showing clusterings 
        # get colors set up 
        NUM_COLORS = args.nmb_cluster
        pca_2d = PCA(n_components=2)

        def plot_pseudo_clusters(epoch, pca_2d):
            cm = plt.get_cmap('gist_rainbow')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

            pseudolabels = []
            image_indexes = []
            cluster_pcas = {}
            for cluster, images in enumerate(deepcluster.images_lists):
                image_indexes.extend(images)
                pseudolabels.extend([cluster] * len(images))

                # compute PCA
                try:
                    PCs_2d = pd.DataFrame(pca_2d.fit_transform(features[images,:]))
                    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
                    cluster_pcas[cluster] = PCs_2d

                    ax.scatter(x=PCs_2d["PC1_2d"], y=PCs_2d["PC2_2d"], label=f"Cluster {cluster}")
                except:
                    continue
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.title(f"Visualizing Pseudolabels with 2D PCA, K={args.nmb_cluster}, epoch={epoch}")
            PLOT_PATH = os.path.join(PCA_PLOT_DIR, f"pseudolabels_epoch={epoch}.png")
            plt.savefig(PLOT_PATH, dpi=300)

            plt.clf()

            return cluster_pcas

        def plot_groundtruth_clusters(epoch, pca_2d, cluster_pcas):
            cm = plt.get_cmap('gist_rainbow')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

            NUM_SCENES = max(gt_labels)
            ax.set_prop_cycle(color=[cm(1.*i/NUM_SCENES) for i in range(NUM_SCENES)])
            gt_pcas_list = [[] for _ in range(NUM_SCENES)]
            gt_images_list = [[] for _ in range(NUM_SCENES)]
            cluster_to_pc_df_dict = {(i+1): None for i in range(NUM_SCENES)}

            pseudolabels = []
            image_indexes = []
            for cluster, images in enumerate(deepcluster.images_lists):
                try:
                    PCs_2d = cluster_pcas[cluster]
                    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
                    
                    for ii in range(len(images)):
                        label_ii = gt_labels[images[ii]]
                        gt_images_list[label_ii-1].append(images[ii])
                        gt_pcas_list[label_ii-1].append(PCs_2d.iloc[ii])
                except:
                    pass

            # compute PCA
            for scene_idx in range(NUM_SCENES):
                current_PCs = pd.concat(gt_pcas_list[scene_idx], join="inner")
                current_PCs.columns = ["PC1_2d", "PC2_2d"]
                ax.scatter(x=current_PCs["PC1_2d"], y=current_PCs["PC2_2d"], 
                           label=f"Scene {scene_idx+1}")

            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.title(f"Visualizing Ground Truth Labels with 2D PCA, K={args.nmb_cluster}, epoch={epoch}")
            PLOT_PATH = os.path.join(PCA_PLOT_DIR, f"ground_truth_epoch={epoch}.png")
            plt.savefig(PLOT_PATH, dpi=300)
            plt.clf()

        if (epoch == 0 or (epoch+1)%10 == 0) or (epoch+1 == args.epochs):
            cluster_pcas = plot_pseudo_clusters(epoch, pca_2d)
            plot_groundtruth_clusters(epoch, pca_2d, cluster_pcas)

    # after training for all epochs, plot the metrics
    titles = ["Clustering Losses", "ConvNet Losses", "Silhouette Scores", "NMI against Previous Label", "NMI against Ground Truth Label"]
    ylabels = ["loss", "loss", "silhouette coefficient", "NMI", "NMI"]
    filenames = ["clustering_loss.png", "convnet_loss.png", "silhouette.png", "nmi_prev.png", "nmi_gt.png"]
    data_lists = [clustering_losses, convnet_losses, silhouette_scores, nmi_prevs, nmi_gts]

    for i in range(5):
        data = data_lists[i]
        ylabel = ylabels[i]
        epochs_x = list(range(1, args.epochs+1)) 

        if ylabels == "NMI":
            epochs_x = list(range(2, args.epochs+1))

            # if NMI scores don't exist, don't plot them 
            data_arr = np.array(data) 
            if np.any(data_arr == None):
                continue
        
        plt.plot(epochs_x, data)
        plt.xlabel("epochs")
        plt.ylabel(ylabel)
        plt.title(f"{titles[i]}, K={args.nmb_cluster}")

        PLOT_PATH = os.path.join(RESULTS_DIR, f"{filenames[i]}.png")
        plt.savefig(PLOT_PATH, dpi=300)
        plt.clf()

def train(loader, model, crit, opt, epoch, RESULTS_DIR):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10**args.wd,
    )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                RESULTS_DIR,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)

        target = target.cuda(non_blocking=True)
        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.data, input_tensor.size(0))
        #losses.update(loss.data[0], input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return losses.avg

def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, input_dict in enumerate(dataloader):
        input_tensor = input_dict['image']
        # frame_number = input_dict['frame_num']
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            if args.experiment == "trivial_time":
                features = np.zeros((N, aux.shape[1] + 2), dtype='float32')
            else:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        shot_num = input_dict['shot_num'] * args.weight_time
        frame_num = input_dict['frame_num'] * args.weight_time
        if args.experiment == "trivial_time":
            aux = np.concatenate((aux, np.expand_dims(shot_num, axis=1), np.expand_dims(frame_num, axis=1)), axis=1)

        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features


'''
def compute_features(dataloader, model, N, max_shot_num=0, max_frame_num=0):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    
    # discard the label information in the dataloader
    for i, input_dict in enumerate(dataloader):
        print("time", i)
        input_tensor = input_dict['image']
        # frame_number = input_dict['frame_num']
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)

        # aux: features, (256, 4096)
        aux = model(input_var).data.cpu().numpy()

        # initialize feature array to store aux (and time info)
        if i == 0:
            if args.experiment == "trivial_time":
                features = np.zeros((N, aux.shape[1]+max_shot_num+max_frame_num+2), dtype='float32')
            else:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        # aux: (args.batch, feature_length)
        print("Length of dataloader", len(dataloader))
        if len(dataloader) < 256:
            if args.experiment == "trivial_time" and i < len(dataloader) - 1:
                shot_num = input_dict['shot_num']
                frame_num = input_dict['frame_num']
                shot_array = np.zeros((args.batch, max_shot_num))
                frame_array = np.zeros((args.batch, max_frame_num))
                print("batch_size", args.batch)
                print("shot_array", shot_array.shape)
                print("max_shot_num", max_shot_num)
                shot_array[np.arange(max_shot_num), shot_num] = 1
                frame_array[np.arange(max_frame_num), frame_num] = 1
                shot_array = shot_array.astype('float32')
                frame_array = frame_array.astype('float32')
                aux = np.concatenate((aux, shot_array, frame_array), axis=1)
            if args.experiment == "trivial_time" and i == len(dataloader) - 1:
                shot_num = input_dict['shot_num']
                frame_num = input_dict['frame_num']
                left_num = N - args.batch * i
                print("left", left_num)
                shot_array = np.zeros((left_num, max_shot_num+1))
                frame_array = np.zeros((left_num, max_frame_num+1))
                print("batch_size", args.batch)
                print("shot_array", shot_array.shape)
                print("max_shot_num", max_shot_num) 
                shot_array[np.arange(len(shot_num)), shot_num] = 1
                frame_array[np.arange(len(frame_num)), frame_num] = 1
                shot_array = shot_array.astype('float32')
                frame_array = frame_array.astype('float32')
                aux = np.concatenate((aux, shot_array, frame_array), axis=1)
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * (args.batch):] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features
'''

# def compute_features(dataloader, model, N):
#     if args.verbose:
#         print('Compute features')
#     batch_time = AverageMeter()
#     end = time.time()
#     model.eval()
#     # discard the label information in the dataloader
#     for i, input_dict in enumerate(dataloader):
#         input_tensor = input_dict['image']
#         # frame_number = input_dict['frame_num']
#         input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
#         aux = model(input_var).data.cpu().numpy()

#         if i == 0:
#             features = np.zeros((N, aux.shape[1]), dtype='float32')

#         aux = aux.astype('float32')
#         if i < len(dataloader) - 1:
#             features[i * args.batch: (i + 1) * args.batch] = aux
#         else:
#             # special treatment for final batch
#             features[i * args.batch:] = aux

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if args.verbose and (i % 200) == 0:
#             print('{0} / {1}\t'
#                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
#                   .format(i, len(dataloader), batch_time=batch_time))
#     return features


if __name__ == '__main__':
    args = parse_args()
    main(args)
