import csv

import numpy as np
from torch.utils.data import Dataset
class MyData(Dataset):
    def __init__(self, root_dir):
        self.rootdir = "./data/rating.csv"
        traindata = {}  # 用于存储读取的数据
        testdata = {}
        userid_map = {}
        itemid_map = {}
        with open(self.rootdir, "r") as file:
            reader = csv.reader(file)
            userid = 0
            for row in reader:
                string_list = row[0].split()
                if string_list[0] not in userid_map:
                    userid_map[string_list[0]] = userid
                    traindata[userid] = [[string_list[1], string_list[3]]]
                    userid += 1
                else:
                    traindata[userid_map[string_list[0]]].append([string_list[1], string_list[3]])

        # print(userid_map)
        # print(traindata[0])
        item_number = 0
        for i in range(userid):
            testdata_list = traindata[i].pop()
            testdata[i] = testdata_list[0]
            if int(testdata[i]) > item_number:
                item_number = int(testdata[i])
        # print(testdata[0])
        # print(traindata[0])

        for i in range(userid):
            traindata[i] = sorted(traindata[i], key=lambda x: x[1])
            tmp_list = []
            for list in traindata[i]:
                tmp_list.append(list[0])
                if int(list[0])> item_number:
                    item_number= int(list[0])
            traindata[i] = tmp_list
        self.train_data=traindata
        self.test_date=testdata
        self.user_num=userid
        self.item_num=item_number+1

    def __getitem__(self, idx):
        userid = idx
        pos_itemid = self.train_data[userid][np.random.randint(0, len(self.train_data[userid]))]
        neg_itemid = np.random.randint(0, self.item_num)
        while neg_itemid in self.train_data[userid]:
            neg_itemid = np.random.randint(0, self.item_num)

        return userid, int(pos_itemid), int(neg_itemid)



    def __len__(self):
        return self.user_num