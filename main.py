from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from data import *
from Bpr import *
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def auc(model, dataset):
    # 获取数据集的大小
    dataset_size = len(dataset)

    # 创建一个SubsetRandomSampler对象，用于随机取样
    sampler = SubsetRandomSampler(range(dataset_size))

    # 使用random_split函数将数据集划分为两个子集，一个用于随机取样，另一个用于剩余的样本
    subset_dataset, _ = random_split(dataset, [100, dataset_size - 100])

    # 创建一个DataLoader对象，用于加载随机取样的子集
    dataloader = DataLoader(subset_dataset, batch_size=1)

    # 遍历DataLoader，获取随机取样的100个样本
    right_sample = 0
    for (user, item_pos, item_neg) in dataloader:
        # print("user")
        # print(user)
        user = torch.Tensor(user).long().to(device)
        item_pos = torch.Tensor(item_pos).long().to(device)
        item_neg = torch.Tensor(item_neg).long().to(device)
        if model.predict(user, item_pos,item_neg) > 0:
            right_sample += 1

    return right_sample/100


if __name__ == '__main__':
    dataset = MyData("./data/rating.csv")
    dataloader = DataLoader(dataset,256)
    print("device =%s"%device)
    Bprmodel = Bpr(dataset.user_num,dataset.item_num,32).to(device)
    for para in Bprmodel.parameters():
        print(para.shape)
        print(para.type())
    opt = torch.optim.Adam(Bprmodel.parameters(), lr=0.001)

    for epoch in range(200):
        epoch_loss=0
        Bprmodel = Bprmodel.train()
        for (users, items_pos, items_neg) in dataloader:
            # print(users)
            # print(items_pos)
            users = torch.Tensor(users).long().to(device)
            items_pos = torch.Tensor(items_pos).long().to(device)
            items_neg = torch.Tensor(items_neg).long().to(device)
            opt.zero_grad()
            loss=Bprmodel.forward(users,items_pos,items_neg)
            epoch_loss+=loss
            loss.backward()
            opt.step()
        auc1=auc(Bprmodel,dataset)

        print("Epoch %d: loss = %f, AUC = %f" % (epoch + 1, epoch_loss, auc1))