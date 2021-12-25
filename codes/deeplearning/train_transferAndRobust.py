import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # ------------------------------------------------------------#
    # 定义用于数据增强的transform
    # ------------------------------------------------------------#
    transList = [transforms.RandomHorizontalFlip(p=0.5),  # 水平镜像
                 transforms.RandomVerticalFlip(p=0.5),  # 垂直镜像
                 transforms.RandomApply(
                     transforms.RandomRotation((-90, 90), resample=False, expand=True, center=None),
                     p=0.5
                 ),  # 旋转角度
                 transforms.RandomApply(
                     transforms.ColorJitter(brightness=0.5),
                     p=0.5
                 ),  # 调整明亮度
                 transforms.RandomApply(
                     transforms.ColorJitter(contrast=0.5),
                     p=0.5
                 ),  # 调整对比度
                 transforms.RandomApply(
                     transforms.ColorJitter(saturation=0.5),
                     p=0.5
                 ),  # 调整渗透率
                 transforms.RandomApply(
                     transforms.ColorJitter(hue=0.5),
                     p=0.3
                 ),  # 调整色调
                 transforms.RandomApply(
                     transforms.Grayscale(num_output_channels=3),
                     p=0.1
                 ),  # 改为灰度图
                 transforms.RandomApply(
                     transforms.RandomAffine(degrees=0, shear=(0, 0, 0, 45)),
                     p=0.2
                 )  # 仿射变换
                 ]

    data_transform = {
        "train": transforms.Compose([transforms.CenterCrop(256),
                                     transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "train_robust": transforms.Compose([transforms.RandomOrder(transList),
                                            transforms.CenterCrop(256),
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.CenterCrop(256),
                                   transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # ------------------------------------------------------------#
    # 导入数据
    # ------------------------------------------------------------#
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../archive"))
    image_path = os.path.join(data_root, "all")
    image_path_robust = os.path.join(data_root, "robust_cracked")  # flower data set path

    # dataset 的大小 [张数,2,3,224,224] 即 [图片张数， 图片和类, RGB三通道，处理后的分辨率，处理后的分辨率]
    dataset_original = datasets.ImageFolder(image_path, transform=data_transform["train"])
    dataset_robust = datasets.ImageFolder(image_path_robust, transform=data_transform["train"])

    dataset = dataset_original.__add__(dataset_robust)

    classIdx_list = dataset_original.class_to_idx
    cla_dict = dict((val, key) for key, val in classIdx_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_size = len(validate_dataset)
    train_size = len(train_dataset)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    net = resnet34()
    # load pretrain weights

    # ------------------------------------------------------------#
    # 迁移学习
    # ------------------------------------------------------------#
    model_weight_path = "../../resnet34-333f7ec4.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=5e-6)

    epochs = 10
    best_acc = 0.0
    save_path = './resNet34_transAndRobust.pth'
    train_steps = len(train_loader)
    val_steps = len(validate_loader)

    # tensorboardX画图准备
    writer = SummaryWriter(log_dir='../../logs/transAndRobust', flush_secs=60)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # 将loss写入tensorboard
            writer.add_scalar("Train_loss", loss, (epoch * (int(train_size / batch_size)) + step))
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] ".format(epoch + 1, epochs)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]

                val_labels_t2n = val_labels.numpy()
                cracked_idx = val_labels_t2n == 0
                nonCracked_idx = val_labels_t2n == 1

                TP += (predict_y[cracked_idx] == 0).sum().item()
                TN += (predict_y[nonCracked_idx] == 1).sum().item()
                FP += (predict_y[nonCracked_idx] == 0).sum().item()
                FN += (predict_y[cracked_idx] == 1).sum().item()

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()


                # 将loss写入tensorboard
                writer.add_scalar("Val_loss", loss, (epoch * (int(val_size / batch_size)) + step))
                # print statistics
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        if (TP + FP != 0):
            precision_cracked = TP / (TP + FP)
            writer.add_scalar("Val_precision_cracked", precision_cracked, epoch)
        else:
            precision_cracked = np.nan

        if( TN + FN != 0):
            precision_nonCracked = TN / (TN + FN)
            writer.add_scalar("Val_precision_nonCracked", precision_nonCracked, epoch)
        else:
            precision_nonCracked = np.nan

        if(TP + FN != 0):
            recall_cracked = TP / (TP + FN)
            writer.add_scalar("Val_recall_cracked", recall_cracked, epoch)
        else:
            recall_cracked = np.nan

        if(TP + FN != 0):
            recall_nonCracked = TN / (TN + FP)
            writer.add_scalar("Val_recall_nonCracked", recall_nonCracked,epoch)
        else:
            recall_nonCracked = np.nan

        val_accurate = acc / val_size
        # 将 val_accuracy写入tensorboard
        writer.add_scalar("Val_acc_epoch", val_accurate, epoch)

        print(
            '[epoch %d] \n train_loss: %.3f  val_loss: %.3f  \n val_accuracy: %.3f \n precision_cracked: %.3f  precision_nonCracked: %.3f \n recall_cracked:%.3f  recall_nonCracked:%.3f  ' %
            (epoch + 1, running_loss / train_steps, val_loss / val_steps, val_accurate, precision_cracked, precision_nonCracked,
             recall_cracked, recall_nonCracked))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    writer.close()
    print('Finished Training')

if __name__ == '__main__':
    main()