import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import pandas as pd
import cv2
from tqdm import tqdm

class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, 'images')  # image path
        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
        self.cls_num = len(self.data)

        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array(
            [self.img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:9]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

class CASMECombinedDataset(Dataset):
    def __init__(self,path = '.',
                 img_sz = 128,
                 calculate_strain = False,
                 initialized_df = None):
        
        if initialized_df is None:
            print('Initializing CASME Combined Dataset...')
            self.df = pd.read_csv(path + '/' + 'combined_3class.csv')
            self.df['OpticalFlow'] = None
            self.path = path
            self.img_sz = img_sz
            for idx,row in tqdm(self.df.iterrows(),ascii = '='):
                prefix = self.__get_prefix(row)
                onset = self.__read_img(
                    prefix + str(row['Onset']) + '.jpg'
                )
                apex = self.__read_img(
                    prefix + str(row['Apex']) + '.jpg'
                )
                self.df.at[idx,'OpticalFlow'] = self.__calc_optical_flow(onset,apex)
                if calculate_strain:
                    self.df.at[idx,'OpticalFlow'] = self.__append_optical_strain(self.df.at[idx,'OpticalFlow'])
                self.df.at[idx,'Class'] = {'negative':0,'positive':1,'surprise':'2'}[row['Class']]  
        else:
            self.df = initialized_df
    
    def __get_prefix(self,row):
        sub_sample = row['Subject'] + '/' + row['Sample'] + '/'
        if row['Dataset'] == 'casme1':
            return self.path + '/casme1_cropped/' + sub_sample + 'reg_' + row['Sample'] + '-'
        elif row['Dataset'] == 'casme2':
            return self.path + '/casme2_cropped/' + sub_sample + 'reg_img'
        elif row['Dataset'] == 'casme^2':
            return self.path + '/casme^2_cropped/' + sub_sample + 'img'
            
    def __read_img(self,name):
        return cv2.cvtColor(
            cv2.resize(
                cv2.imread(name,cv2.IMREAD_COLOR),
                (self.img_sz,self.img_sz),
                interpolation = cv2.INTER_CUBIC
            ),
            cv2.COLOR_BGR2GRAY
        )
    
    def __calc_optical_flow(self,onset,apex):
        return np.array(
            cv2.optflow.DualTVL1OpticalFlow_create().calc(onset,apex,None)
        ).transpose((2,0,1))
    
    def __append_optical_strain(self,flow):
        ux = cv2.Sobel(flow[0],cv2.CV_32F,1,0)
        uy = cv2.Sobel(flow[0],cv2.CV_32F,0,1)
        vx = cv2.Sobel(flow[1],cv2.CV_32F,1,0)
        vy = cv2.Sobel(flow[1],cv2.CV_32F,0,1)
        strain = np.sqrt(ux * ux + uy * uy + 0.5 * (vx + uy) * (vx + uy))
        return np.concatenate((flow,strain.reshape(1,self.img_sz,self.img_sz)),axis = 0)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        return self.df.at[idx,'OpticalFlow'],int(self.df.at[idx,'Class'])
        
class LOSOGenerator():
    def __init__(self,dataset):
        self.data = dataset
        self.subjects = self.data.df[['Dataset','Subject']].drop_duplicates().reset_index()
        self.idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx == len(self.subjects):
            raise StopIteration
        ds,sub = self.subjects.at[self.idx,'Dataset'],self.subjects.at[self.idx,'Subject']
        self.idx += 1
        train_df = self.data.df[(self.data.df.Dataset != ds) | (self.data.df.Subject != sub)] \
            .reset_index(drop=True)
        test_df = self.data.df[(self.data.df.Dataset == ds) & (self.data.df.Subject == sub)] \
            .reset_index(drop=True)
        return CASMECombinedDataset(initialized_df = train_df),CASMECombinedDataset(initialized_df = test_df)