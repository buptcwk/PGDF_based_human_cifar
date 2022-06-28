from warnings import filterwarnings

filterwarnings("ignore")

import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture

from torch.utils.data import DataLoader, Dataset
from dataloader_cifar import cifar_dataset
from dataset_his import his_dataset
import dataloader_cifar as dataloader
import dataloader_easy 
from PreResNet import *
from preset_parser import *

import pickle


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == "__main__":
    args = parse_args("./presets.json")

    logs = open(os.path.join(args.checkpoint_path, "saved", "metrics.log"), "a")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    noise_file_human10 = torch.load('./data/CIFAR-10_human.pt')
    noise_file_human100 = torch.load('./data/CIFAR-100_human.pt')
    if args.noise_type == 'aggre':
        noise_file = noise_file_human10['aggre_label'].tolist()
        noise_filename = "./aggre_label.json"
        detect_file = "./aggre_detection.npy"
        with open(noise_filename,"w") as f:
            json.dump(noise_file,f)
    elif args.noise_type == 'worst':
        noise_file = noise_file_human10['worse_label'].tolist()
        noise_filename = "./worse_label.json"
        detect_file = "./worse_detection.npy"
        with open(noise_filename,"w") as f:
            json.dump(noise_file,f)
    elif args.noise_type == 'rand1':
        noise_file = noise_file_human10['random_label1'].tolist()
        noise_filename = "./random_label1.json"
        detect_file = "./rand1_detection.npy"
        with open(noise_filename,"w") as f:
            json.dump(noise_file,f)
    elif args.noise_type == 'noisy100':
        noise_file = noise_file_human100['noisy_label'].tolist()
        noise_filename = "./noisy_label.json"
        detect_file = "./noisy100_detection.npy"
        with open(noise_filename,"w") as f:
            json.dump(noise_file,f)



    def record_history(index, output, target, recorder):
        pred = F.softmax(output, dim=1).cpu().data
        for i, ind in enumerate(index):
            recorder[ind].append(pred[i][target.cpu()[i]].numpy().tolist())

        return

   

    def warmup(epoch, net, optimizer, dataloader, recorder):
        net.train()
        num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            record_history(path, outputs, labels, recorder)
            loss = CEloss(outputs, labels)
            if (
                args.noise_mode == "asym"
            ):  # penalize confident prediction for asymmetric noise
                penalty = conf_penalty(outputs)
                L = loss + penalty
            elif args.noise_mode == "sym":
                L = loss
            L.backward()
            optimizer.step()

            # sys.stdout.write("\r")
            # sys.stdout.write(
            #     "%s: %.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f"
            #     % (
            #         args.dataset,
            #         args.r,
            #         args.noise_mode,
            #         epoch,
            #         args.num_epochs - 1,
            #         batch_idx + 1,
            #         num_iter,
            #         loss.item(),
            #     )
            # )
            # sys.stdout.flush()

    class NegEntropy(object):
        def __call__(self, outputs):
            probs = torch.softmax(outputs, dim=1)
            return torch.mean(torch.sum(probs.log() * probs, dim=1))

    def create_model(devices=[0]):
        model = ResNet18(num_classes=args.num_class)
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=devices).cuda()
        return model

    loader = dataloader.cifar_dataloader(
        dataset=args.dataset,
        r=args.r,
        noise_mode=args.noise_mode,
        batch_size=args.batch_size,
        warmup_batch_size=args.warmup_batch_size,
        num_workers=args.num_workers,
        root_dir=args.data_path,
        noise_file=noise_filename,
        augmentation_strategy=args,
    )

    

    print("| Building net")
    devices = range(torch.cuda.device_count())
    net1 = create_model(devices)
    cudnn.benchmark = True

    optimizer1 = optim.SGD(
        net1.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )

    all_loss = [[], []]  # save the history of losses from two networks
    epoch = 0

    CE = nn.CrossEntropyLoss(reduction="none")
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode == "asym":
        conf_penalty = NegEntropy()
    warmup_trainloader = loader.run("warmup")
    recorder1 = [[] for i in range(len(warmup_trainloader.dataset))]# recorder for training history of all samples
    
    while epoch < args.num_epochs:
        lr = args.learning_rate
        if epoch >= args.lr_switch_epoch:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr
        print("epoch:%d" % epoch)
        warmup(epoch, net1, optimizer1, warmup_trainloader, recorder1)
        epoch += 1

    with open(f"{args.checkpoint_path}/saved/recorder1.p","wb") as f1:
        pickle.dump(recorder1,f1)
    
    #filter easy samples
    record = np.array(recorder1)
    r_mean = np.mean(record,axis=1)
    sort_mean = np.sort(r_mean)
    sort_index = np.argsort(r_mean)
    train_dataset = cifar_dataset(dataset=args.dataset,
                            r=args.r,
                            noise_mode="sym",
                            root_dir=args.data_path,
                            noise_file=noise_filename,
                            transform="",
                            mode="all")
    easy_noisylabel = np.array(train_dataset.noise_label)

    easy_class_index = [[] for i in range(args.num_class)]
    easy_class_mean = [[] for i in range(args.num_class)]
    easy_class_sortind = [[] for i in range(args.num_class)]
    easy_class_sortind_all = [[] for i in range(args.num_class)]
    easy = []
    mix = []
    class_ii = []

    for ii in range(args.num_class):
        easy_class_index[ii] = np.argwhere(easy_noisylabel == ii)[:,0]
        easy_class_mean[ii] = r_mean[easy_class_index[ii]]
        easy_class_sortind[ii] = np.argsort(easy_class_mean[ii])
        easy_class_sortind_all[ii] = easy_class_index[ii][easy_class_sortind[ii]]
        class_ii.append(easy_class_sortind_all[ii][int(len(easy_class_sortind_all[ii])*(0.5+args.r*0.5)):])
        easy = easy + easy_class_sortind_all[ii][int(len(easy_class_sortind_all[ii])*(0.5+args.r*0.5)):].tolist()
        mix = mix + easy_class_sortind_all[ii][int(len(easy_class_sortind_all[ii])*(args.r*0.5)):int(len(easy_class_sortind_all[ii])*(0.5+args.r*0.5))].tolist()
    easy = np.array(easy)
    mix = np.array(mix)
    with open(f"{args.checkpoint_path}/saved/class_test_index7.p","wb") as f:
        pickle.dump(class_ii,f)

    
    """
    record = np.array(recorder1)
    r_mean = np.mean(record,axis=1)
    sort_mean = np.sort(r_mean)
    sort_index = np.argsort(r_mean)
    train_dataset = cifar_dataset(dataset=args.dataset,
                            r=args.r,
                            noise_mode="sym",
                            root_dir=args.data_path,
                            noise_file=f"{args.checkpoint_path}/saved/labels.json",
                            transform="",
                            mode="all")
    easy = sort_index[int(train_dataset.__len__()*(0.5+args.r*0.5)):]
    """
    train_data_easy = train_dataset.train_data[easy]
    train_label_easy = np.array(train_dataset.noise_label)[easy]
    with open(f"{args.checkpoint_path}/saved/train_data_easy.p","wb") as f1:
        pickle.dump(train_data_easy,f1)
    with open(f"{args.checkpoint_path}/saved/train_label_easy.p","wb") as f1:
        pickle.dump(train_label_easy,f1)



    #inject noise to D_e

    easy_loader = dataloader_easy.easy_dataloader(
        dataset=args.dataset,
        r=args.r,
        noise_mode=args.noise_mode,
        batch_size=args.batch_size,
        warmup_batch_size=args.warmup_batch_size,
        num_workers=args.num_workers,
        root_dir=args.data_path,
        noise_file=f"{args.checkpoint_path}/saved/easy_labels.p",
        augmentation_strategy=args,
    )

    print("| Building net")
    devices = range(torch.cuda.device_count())
    net_easy = create_model(devices)
    cudnn.benchmark = True

    optimizer_easy = optim.SGD(
        net_easy.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )

    all_loss = [[], []]  
    epoch = 0


        
    warmup_easyloader = easy_loader.run("warmup",train_data_easy,train_label_easy) # D_a

    
    recorder_easy = [[] for i in range(len(warmup_easyloader.dataset))]


    while epoch < args.num_epochs:
        lr = args.learning_rate
        if epoch >= args.lr_switch_epoch:
            lr /= 10
        for param_group in optimizer_easy.param_groups:
            param_group["lr"] = lr



        print("epoch:%d" % epoch)
        warmup(epoch, net_easy, optimizer_easy, warmup_easyloader, recorder_easy) # recorder for training history of easy samples


    
        epoch += 1
    
    with open(f"{args.checkpoint_path}/saved/recorder_easy.p","wb") as f1:
        pickle.dump(recorder_easy,f1)

    
    #training 1D-CNN classifier 
    record_easy = np.array(recorder_easy)
    r_mean_easy = np.mean(record_easy,axis=1)
    sort_mean_easy = np.sort(r_mean_easy)
    sort_index_easy = np.argsort(r_mean_easy)
    train_label_e = train_label_easy
    with open(f"{args.checkpoint_path}/saved/easy_labels.p","rb") as f1:
        noise_label_e = pickle.load(f1)
        
    noise_rate=args.r

    ##record index for every class
    noise_label_arr = np.array(noise_label_e)
    class_index_easy = [[] for i in range(args.num_class)]
    class_sort_easy = [[] for i in range(args.num_class)]
    class_sortind_easy = [[] for i in range(args.num_class)]
    class_sortind_all_easy = [[] for i in range(args.num_class)]
    class_train_index = [[] for i in range(args.num_class)]
    class_train_record = [[] for i in range(args.num_class)]
    class_train_rlabel = [[] for i in range(args.num_class)]

    noisy_labels_test = np.array(train_dataset.noise_label)
    class_index = [[] for i in range(args.num_class)]
    class_sort = [[] for i in range(args.num_class)]
    class_sortind = [[] for i in range(args.num_class)]
    class_sortind_all = [[] for i in range(args.num_class)]
    class_test_index = [[] for i in range(args.num_class)]
    class_test_record = [[] for i in range(args.num_class)]
    class_test_rlabel = [[] for i in range(args.num_class)]
    

    pred = [False for j in range(train_dataset.__len__())]
    prob = [0 for j in range(train_dataset.__len__())]
    pred2 = [False for j in range(train_dataset.__len__())]
    prob2 = [0 for j in range(train_dataset.__len__())]

    class oneD_CNN(nn.Module):
        def __init__(self):
            super(oneD_CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(1, 16, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(16, 32, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.conv3 = nn.Sequential(
                nn.Conv1d(32, 32, 3, stride=1, padding=1),
                nn.ReLU()
            )

            self.output = nn.Linear(in_features=32*args.num_epochs, out_features=2)
        def forward(self,x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.size(0),-1)
            output = self.output(x)
            return output
    
    for kk in range(args.num_class):
        print(f"training 1D-CNN classifier for class{kk}")
        class_index_easy[kk] = np.argwhere(noise_label_arr == kk)[:,0]
        
        class_sort_easy[kk] = r_mean_easy[class_index_easy[kk]]
        
        class_sortind_easy[kk] = np.argsort(class_sort_easy[kk])
       
        class_sortind_all_easy[kk] = class_index_easy[kk][class_sortind_easy[kk]]
        class_sort_easy[kk] = np.sort(class_sort_easy[kk])
        class_train_index[kk] = class_sortind_all_easy[kk][int(class_sortind_all_easy[kk].shape[0]*noise_rate*0.5):int(class_sortind_all_easy[kk].shape[0]*(0.5+noise_rate*0.5))]
        
        class_train_record[kk] = record_easy[class_train_index[kk]]
        

        for index in class_train_index[kk]:
            if train_label_e[index]==noise_label_e[index]:
                class_train_rlabel[kk].append(1)#hard samples
            else:
                class_train_rlabel[kk].append(0)#noisy samples
        class_train_record[kk] = torch.Tensor(class_train_record[kk])
        class_train_record[kk] = torch.unsqueeze(class_train_record[kk],1)

        class_train_rlabel[kk] = torch.Tensor(class_train_rlabel[kk]).type(torch.LongTensor)

        class_index[kk] = np.argwhere(noisy_labels_test == kk)[:,0]
        
        class_sort[kk] = r_mean[class_index[kk]]
        
        class_sortind[kk] = np.argsort(class_sort[kk])
        
        class_sortind_all[kk] = class_index[kk][class_sortind[kk]]
        class_sort[kk] = np.sort(class_sort[kk])
        class_test_index[kk] = class_sortind_all[kk][int(class_sortind_all[kk].shape[0]*noise_rate*0.5):int(class_sortind_all[kk].shape[0]*(0.5+noise_rate*0.5))]
        
        class_test_record[kk] = record[class_test_index[kk]]

        

        for index in class_test_index[kk]:
            if train_dataset.noise_label[index] == train_dataset.train_label[index]:
                class_test_rlabel[kk].append(1)#hard samples
            else:
                class_test_rlabel[kk].append(0)#noisy samples
        
        
        class_test_record[kk] = torch.Tensor(class_test_record[kk])
        class_test_record[kk] = torch.unsqueeze(class_test_record[kk],1)
        class_test_rlabel[kk] = torch.Tensor(class_test_rlabel[kk]).type(torch.LongTensor)

    

        batch_size = 256
        train_dataset_ehn = his_dataset(class_train_record[kk], class_train_rlabel[kk])
        test_dataset_ehn = his_dataset(class_test_record[kk], class_test_rlabel[kk])
        train_dataloader_ehn = DataLoader(dataset=train_dataset_ehn, batch_size = batch_size, num_workers = 16, shuffle = False)
        test_dataloader_ehn = DataLoader(dataset=test_dataset_ehn, batch_size = 50000, num_workers = 16)

        # one dimension CNN classifier
        
    

        net = oneD_CNN()    

        #training
        net = net.cuda()
        optimizer = torch.optim.Adadelta(net.parameters(), lr=1)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        loss_func = torch.nn.CrossEntropyLoss()
        if args.num_class == 10:
            max_epoch = 50 #hyper-parameters
        elif args.num_class == 100:
            max_epoch = 100
        for epoch in range(max_epoch):
            net.train()
            loss_sigma = 0.0  #
            correct = 0.0
            total = 0.0
            for i,(train_data,train_label) in enumerate(train_dataloader_ehn):
                train_data,train_label = Variable(train_data).cuda(),Variable(train_label).cuda()
                out = net(train_data)  

                loss = loss_func(out, train_label)  

                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step()

                _, predicted = torch.max(out.data, 1)
                total += train_label.size(0)
                correct += (predicted == train_label).squeeze().sum().cpu().numpy()
                loss_sigma += loss.item()
            
            scheduler.step()
            print("Training: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, loss_sigma, correct / total))
            
            if epoch % 2 == 0:
                net.eval()
                conf_matrix = np.zeros((2,2))
                with torch.no_grad():
                    for it,(test_data,test_label) in enumerate(test_dataloader_ehn):
                        test_data,test_label = Variable(test_data).cuda(),Variable(test_label).cuda()
                        test_out = net(test_data)
                        _, predicted = torch.max(test_out.data, 1)
                        for i in range(predicted.shape[0]):
                            conf_matrix[test_label[i],predicted[i]]+=1
                        
                    
                print(conf_matrix)
                acc = np.diag(conf_matrix).sum()/np.sum(conf_matrix)
                print(acc)

        test_out = F.softmax(test_out)
        
        for i in range(test_out.shape[0]):
            p = float(test_out[i][1])
            if p > 0.5:#hard
                prob[class_test_index[kk][i]]=p
                #print("class_test_index",class_test_index[kk][i])
            else :# noisy
                prob[class_test_index[kk][i]]=p
                #print("class_test_index",class_test_index[kk][i])
        for ind in class_sortind_all[kk][int(class_sortind_all[kk].shape[0]*(0.5+noise_rate*0.5)):]:
            prob[ind]=1 # easy
            #print("ind",ind)
        

        net2 = oneD_CNN()    

        #training
        net2 = net2.cuda()
        optimizer2 = torch.optim.Adadelta(net2.parameters(), lr=1)
        scheduler2 = StepLR(optimizer2, step_size=1, gamma=0.99)
        loss_func2 = torch.nn.CrossEntropyLoss()
        if args.num_class == 10:
            max_epoch2 = 50 #hyper-parameters
        elif args.num_class == 100:
            max_epoch2 = 100
       
        for epoch in range(max_epoch2):
            net2.train()
            loss_sigma2 = 0.0  #
            correct2 = 0.0
            total2 = 0.0
            for i,(train_data,train_label) in enumerate(train_dataloader_ehn):
                train_data,train_label = Variable(train_data).cuda(),Variable(train_label).cuda()
                out2 = net2(train_data)  

                loss2 = loss_func2(out2, train_label)  

                optimizer2.zero_grad() 
                loss2.backward() 
                optimizer2.step()

                _, predicted2 = torch.max(out2.data, 1)
                total2 += train_label.size(0)
                correct2 += (predicted2 == train_label).squeeze().sum().cpu().numpy()
                loss_sigma2 += loss2.item()
            
            scheduler2.step()
            print("Training: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch2, loss_sigma2, correct2 / total2))
            
            if epoch % 2 == 0:
                net2.eval()
                conf_matrix2 = np.zeros((2,2))
                with torch.no_grad():
                    for it,(test_data,test_label) in enumerate(test_dataloader_ehn):
                        test_data,test_label = Variable(test_data).cuda(),Variable(test_label).cuda()
                        test_out2 = net2(test_data)
                        _, predicted2 = torch.max(test_out2.data, 1)
                        for i in range(predicted2.shape[0]):
                            conf_matrix2[test_label[i],predicted2[i]]+=1
                        
                    
                print(conf_matrix2)
                acc2 = np.diag(conf_matrix2).sum()/np.sum(conf_matrix2)
                print(acc2)

        test_out2 = F.softmax(test_out2)
        
        for i in range(test_out2.shape[0]):
            p2 = float(test_out2[i][1])
            if p2 > 0.5:#hard
                prob2[class_test_index[kk][i]]=p2
                #print("class_test_index",class_test_index[kk][i])
            else :# noisy
                prob2[class_test_index[kk][i]]=p2
                #print("class_test_index",class_test_index[kk][i])
        for ind in class_sortind_all[kk][int(class_sortind_all[kk].shape[0]*(0.5+noise_rate*0.5)):]:
            prob2[ind]=1 # easy

    
    prob = np.array(prob)
    with open(f"{args.checkpoint_path}/saved/prob1_ehn.p","wb") as f:
        pickle.dump(prob,f) 
    prob2 = np.array(prob2)
    noise_or_not_pred = prob2 <= 0.5
    np.save(detect_file,noise_or_not_pred)
    with open(f"{args.checkpoint_path}/saved/prob2_ehn.p","wb") as f2:
        pickle.dump(prob2,f2)



