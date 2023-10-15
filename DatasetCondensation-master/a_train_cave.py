import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
# from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

from utils_baseline import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import warnings
import argparse
from a_cvae import  CVAE
import torch.utils.data
from torch import optim
warnings.filterwarnings("ignore")

# watch -n 1 nvidia-smi
import os

def main(args):
    print("Test_mode:",args.test_mode)

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")


    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)


    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans


    args.distributed = torch.cuda.device_count() > 1

    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    def get_images_init(c, n,exp): # get random n images from class c
        # start_idx = i  # 指定起始索引 i
        # end_idx = i + n  # 计算结束索引（不包括结束索引）

        # 从指定的起始索引到结束索引获取元素
        idx_shuffle  = indices_class[c][args.ipc*exp:args.ipc*exp + n]

        # idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle],idx_shuffle
    
    # def get_images(c, n):  # get random n images from class c
    #         idx_shuffle = np.random.permutation(indices_class[c])[:n]
    #         return images_all[idx_shuffle].detach().data,idx_shuffle
  
    if args.pairsnum!=100:
        pairs_real = []
        indexs_real = []
        for exp in range(100 - args.pairsnum):
            # 80-99

            exp = exp + args.pairsnum
        
            for c in range(num_classes):
                reals,index = get_images_init(c, args.ipc,exp)
                reals = reals.detach().data
                pairs_real.append(reals)
                indexs_real.append(index)
        img_real_test= torch.cat(pairs_real, dim=0)
        label_real_test = []
        
        
        new_lst = np.array(indexs_real).flatten()

        label_real_test = labels_all[new_lst]

    device = args.device

    # get img_real and img_syn for training CVAE
    img_syn = []
    label_syn = []
    img_real_train = []
    label_real_train = []
    indexs_real=[]
    # /home/ssd7T/ztl_dm/indexs_real_20.pt 0-39 50 -89 90 70 50 30 10
    for i in range(args.pairsnum):
        try:

            
            img_syn_ = torch.load(f'/home/ssd7T/ztl_ftd/img_syn_{i}.pt')
            label_syn_ = torch.load(f'/home/ssd7T/ztl_ftd/label_syn_{i}.pt')
            # pairs_real_=torch.load(f'/home/ssd7T/ztl_ftd/pairs_real_{i}.pt')
            indexs_real_=torch.load(f'/home/ssd7T/ztl_ftd/indexs_real_{i}.pt')
            indexs_real.append(indexs_real_)

            img_syn.append(img_syn_)
            label_syn.append(label_syn_)
                
        except:
            pass


    img_syn = torch.cat(img_syn, dim=0).to(device)
    label_syn = torch.cat(label_syn, dim=0).to(device)
    # img_real_train = torch.cat(img_real_train, dim=0).to(device)
    # label_real_train = torch.cat(label_real_train, dim=0).to(device)
    
    new_lst = np.array(indexs_real).flatten()
    img_real_train = torch.randn(size=(num_classes * args.pairsnum * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
    img_real_train.data = images_all[new_lst]
    # print("test_image_syn: ",test_image_syn.shape)

    label_real_train = labels_all[new_lst]
    # images_all[new_lst].shape

    print("syn set shape:",img_syn.shape)
    print("img_real_train shape:",img_real_train.shape)
    print("label_real_train shape:",label_real_train.shape)
    if args.pairsnum!=100:
        print("img_real_test shape:",img_real_test.shape)
    

    accs = []
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    import copy
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    model_eval= model_eval_pool[0]

    for it_eval in range(args.num_eval):
        # net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

        if args.test_mode == 0:
                
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(device) # get a random model
            # image_syn_eval, label_syn_eval = copy.deepcopy(images_all), copy.deepcopy(labels_all) # avoid any unaware modification
            image_syn_eval, label_syn_eval = copy.deepcopy(img_real_train.to(device)), copy.deepcopy(label_real_train.to(device))
            # print("Final test shape:",image_syn_eval.shape)
            print("Final test shape:",label_syn_eval.shape)
            
            # image_syn_eval, label_syn_eval = copy.deepcopy(torch.cat((output[0].to(device),img_real_test_concat.to(device)), dim = 0)),
            # copy.deepcopy(torch.cat((label_real_test.to(device),label_real_test_concat.to(device)), dim = 0)) # avoid any unaware modification
            
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
            accs.append(acc_test)
        # 0 no cvae, just test all train images(T) and labels
            
        else: 

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)


            lr = args.lr_cvae
            k = args.k 
            hidden = args.hidden 
            num_channels = 3 # CIFA
            batch = args.batch_size
            
        # 1 train cvae with all the train images(T), obtain model A, then input the train images for test(T2), using output for test
        # 2 train cvae with all the train images(T), obtain model A, then input the train images for test(T2), using combination of output and T for test

        # 3 train cvae with N batchs of the sample pairs(T1 - S1), obtain model B, then input the train images for test(T2), using output for test
        # 4 train cvae with N batchs of the sample pairs(T1 - S1), obtain model B, then input the train images for test(T2), using combination of output and T for test
            
            if args.test_mode == 1:
                
                if os.path.exists("cvae_A.pt"):
                    model = torch.load('cvae_A.pt').to(device)
                    
                else:
                    model = CVAE(d = hidden, k=k, num_channels=num_channels).to(device) 

                    # def __init__(self, d, kl_coef=0.1, **kwargs):

                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args.dataset == 'imagenet' else 30, 0.5,)
                    # A
                    epochs = args.epochs_cave
                    for epoch in range(epochs):

                        random_indices = np.random.choice(len(img_real_train), batch, replace=False)

                    # 使用随机选择的索引来获取数据并改变形状
                        batch_img = images_all[random_indices].reshape((batch, 3, 32, 32)).to(device)
                        # batch_syn = img_syn[random_indices].reshape((batch, 3, 32, 32)).to(device) 
                        # batch_img_y = label_real_train[c*batch:(c+1)*batch].to(device) 
                        
                        outputs = model(batch_img)
                        loss = model.loss_function(batch_img, *outputs)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()
                        if epoch % 10 == 0 or epoch == epochs -1:
                            print("====> Epoch: {} || {}".format(epoch, loss.item()))
                    
                    torch.save(model,'cvae_A.pt') 
                        
                with torch.no_grad():
                    output = model(images_all.to(device))
                data_save = []
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(device) # get a random model
                image_syn_eval, label_syn_eval = copy.deepcopy(output[0]), copy.deepcopy(labels_all) # avoid any unaware modification
                print("Final test shape:",image_syn_eval.shape)
                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                accs.append(acc_test)
                
            elif args.test_mode == 2:
                
                model = torch.load('cvae_A.pt').to(device)  
                        
                with torch.no_grad():
                    output = model(img_real_test.to(device))
        
                data_save = []
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(device) # get a random model

                image_syn_eval, label_syn_eval = copy.deepcopy(torch.cat((output[0].to(device),img_real_train.to(device)), dim = 0)),copy.deepcopy(torch.cat((label_real_test.to(device),label_real_train.to(device)), dim = 0)) # avoid any unaware modification
                print("Final test shape:",image_syn_eval.shape)
                
                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                accs.append(acc_test)
                
            elif args.test_mode == 3:
                if os.path.exists("cvae_B3.pt"):
                    model = torch.load('cvae_B3.pt').to(device)
                else:
                    try : 
                        model = torch.load('cvae_A.pt').to(device)   
                    except:
                        print("Set args.test_mode == 1 to get model A first! ")

                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args.dataset == 'imagenet' else 30, 0.5,)

                    epochs = args.epochs_cave
                    for epoch in range(epochs):
                        
                        # 随机选择batch个索引
                        random_indices = np.random.choice(len(img_real_train), batch, replace=False)

                        # 使用随机选择的索引来获取数据并改变形状
                        batch_img = img_real_train[random_indices].reshape((batch, 3, 32, 32)).to(device)
                        batch_syn = img_syn[random_indices].reshape((batch, 3, 32, 32)).to(device) 

                        # batch_img_y = label_real_train[c*batch:(c+1)*batch].to(device) 
                        
                        outputs = model(batch_img)
                        
                        loss = model.loss_function(batch_syn, *outputs)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # logs['loss'].append(loss.item())
                        # scheduler.step()

                        if epoch % 10 == 0 or epoch == epochs -1:
                            # print("Epoch {:02d}/{:02d} , Loss {:9.4f}".format(
                            #     epoch, epochs,  loss.item()))
                            print("====> Epoch: {} || {}".format(epoch, loss.item()))
                    torch.save(model,'cvae_B3.pt') 
                
                with torch.no_grad():
                    output = model(images_all.to(device))
                data_save = []
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(device) # get a random model
                image_syn_eval, label_syn_eval = copy.deepcopy(output[0]), copy.deepcopy(labels_all) # avoid any unaware modification
                print("Final test shape:",image_syn_eval.shape)
                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                accs.append(acc_test)
                
            elif args.test_mode == 4:
                if os.path.exists("cvae_B3.pt"):
                    model = torch.load('cvae_B3.pt').to(device)
                    
                else:
                    print("Set args.test_mode == 1 or 2 to get model A first! ") 
                        
                with torch.no_grad():
                    output = model(img_real_test.to(device))
                data_save = []
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(device) # get a random model
                
                image_syn_eval, label_syn_eval = copy.deepcopy(torch.cat((output[0].to(device),img_real_train.to(device)), dim = 0)),copy.deepcopy(torch.cat((label_real_test.to(device),label_real_train.to(device)), dim = 0)) # avoid any unaware modification
                print("Final test shape:",image_syn_eval.shape)
                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                accs.append(acc_test)

    print('Test mode %d Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(args.test_mode,len(accs), model_eval, np.mean(accs), np.std(accs)))
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Parameter Processing')


    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--zca', default='True', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')



    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=2000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='/home/ssd7T/ZTL_gcond/data_cv', help='dataset path')
    parser.add_argument('--save_path', type=str, default='/home/ssd7T/ztl_dm/gen', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('-k', '--dict-size',  default=10,type=int, dest='k', metavar='K',
                                help='number of atoms in dictionary')
    parser.add_argument('--lr', type=float, default=2e-4,
                                help='learning rate')
    
    parser.add_argument('--lr_cvae', type=float, default=2e-4,
                                help='learning rate')
    
    parser.add_argument('--vq_coef', type=float, default=None,
                                help='vq coefficient in loss')
    parser.add_argument('--commit_coef', type=float, default=None,
                                help='commitment coefficient in loss')
    parser.add_argument('--kl_coef', type=float, default=None,
                                help='kl-divergence coefficient in loss')

    parser.add_argument('--num_exp', type=int, default=10, help='the batchs of test data')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--pairsnum', type=int, default=90, help='image real for test')
    parser.add_argument('--test_mode', type=int, default=1, help='test mode')
    # 0 no cvae, just test all train images(T) and labels  90 70 50 30 10

    # 1 train cvae with all the train images(T), obtain model A, then input the train images for test(T2), using output for test
    # 2 train cvae with all the train images(T), obtain model A, then input the train images for test(T2), using combination of output and T for test

    # 3 train cvae with N batchs of the sample pairs(T1 - S1), obtain model B, then input the train images for test(T2), using output for test
    # 4 train cvae with N batchs of the sample pairs(T1 - S1), obtain model B, then input the train images for test(T2), using combination of output and T for test
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                                help='input batch size for training (default: 128)')
    parser.add_argument('--hidden', type=int, default=256, metavar='N',
                                help='number of hidden channels')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                                    help='random seed (default: 1)')
    parser.add_argument('--epochs_cave', type=int, default=3000,
                                help='train cvae')
    parser.add_argument('--gpu_id', type=str, default="0",
                                help='gpu')
    args = parser.parse_args()
    # --seed
    # 显示第 0 和第 1 个 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    main(args)

# cd /home/wangkai/ztl_project/difussion-dd/DatasetCondensation-master
# conda activate ztl

# python a_train_cave.py --test_mode 0 --pairsnum 80 --gpu_id 0
# python a_train_cave.py --test_mode 1 --pairsnum 80 --gpu_id 1
# python a_train_cave.py --test_mode 2 --pairsnum 80 --gpu_id 4

# python a_train_cave.py --test_mode 3 --pairsnum 80 --gpu_id 5

# python a_train_cave.py --test_mode 4 --pairsnum 80 --gpu_id 6

# watch -n 1 nvidia-smi

# git init
# git add .
# git config --global user.email "2537643186@qq.com"
# git config --global user.name "ztlmememe"
# git commit -m "CVAE"
# git remote add origin git@github.com:ztlmememe/VAE-GAN.git

# git push -u origin master

# scp -r /home/wangkai/ztl_project/difussion-dd/FTD-distillation-main kwang@10.11.65.8:/home/kwang/ztl/difussion-dd/FTD-distillation-main