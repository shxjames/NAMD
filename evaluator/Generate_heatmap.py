import torch
from vit import ViT
#from creat_dataloaders import Finetune_TrainSet,Finetune_ValSet,Finetune_TestSet
from creat_dataloaders import Lung_DM
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_curve, auc
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### ViT ###

model = ViT(
    image_size = 64,
    patch_size = 8,
    num_classes = 1,
    dim = 1024,
    depth = 6,
    heads = 32,
    mlp_dim = 2048,
    dropout = 0.5,
    emb_dropout = 0.5
).to(device)
###

checkpoint_path = 'checkpoint/vit/'

model.load_state_dict(torch.load(checkpoint_path + str(36) + '.pth'))
model.to(device)
#summary(model, (1,64, 64))


dm = Lung_DM(ch=1)
dm.setup('fit')
#data_loader_train = dm.train_dataloader()
#data_loader_val = dm.val_dataloader()
data_loader_test = dm.test_seperate_dataloader()

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
pos_weight = torch.tensor([1.0]).to(device)
criterion_score = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)

#epoch = 50
### Train ###
#for epi in range(epoch):

model.load_state_dict(torch.load(checkpoint_path + str(42) + '.pth'))
model.to(device)

train_loss = 0
model.eval()

Result_A = []
Result_B = []
Label = []

for index, (image_A, image_B, label) in enumerate(data_loader_test):

    image_A = image_A.to(device)
    image_B = image_B.to(device)

    label = label.to(device)

    output_A, attention_map = model(image_A, return_map = True)
    output_B, attention_map = model(image_B, return_map = True)
    cls_attn = attention_map[0, :, 0, 1:]
    cls_attn_mean = cls_attn.mean(0)

    num_patches = cls_attn_mean.shape[0]
    grid_size = int(math.sqrt(num_patches))  # e.g. 8
    attn_map = cls_attn_mean.reshape(grid_size, grid_size)  # (8,8)

    heatmap = attn_map.unsqueeze(0).unsqueeze(0)  # (1,1,14,14)
    heatmap = F.interpolate(heatmap, size=(64,64), mode="bilinear").squeeze()


    Result_A.append(output_A.detach().cpu().numpy())
    Result_B.append(output_B.detach().cpu().numpy())
    Label.append(label.detach().cpu().numpy())


    image = image_A
    image = image[0,0,:]
    img_np = image.detach().cpu().numpy()
    heatmap = heatmap.detach().cpu()

    img_min = heatmap.min()
    img_max = heatmap.max()
    heatmap = (heatmap - img_min) / (img_max - img_min)

    image = (image + 1)/2

    concatenated = torch.cat([image.detach().cpu(), heatmap], dim=1)

    #plt.imshow(concatenated.squeeze().numpy(), cmap='gray')
    #plt.axis('off')
    #plt.show()

    save_image = transforms.ToPILImage()(concatenated)


    path2 = "Attention_map_test/"
    if label == 0:
        filename = 'B_' + str(index) + '.png'
    else:
        filename = 'M_' + str(index) + '.png'

    #save_image.save(os.path.join(path2, filename))

        
Result_A = np.array(Result_A).reshape(-1)
Result_B = np.array(Result_B).reshape(-1)
Label = np.array(Label).reshape(-1)

fpr, tpr, thresholds = metrics.roc_curve(Label, Result_A, pos_label=1)
pr_auc = metrics.auc(fpr, tpr)
print('Test_auc_A', pr_auc)

fpr, tpr, thresholds = metrics.roc_curve(Label, Result_B, pos_label=1)
pr_auc = metrics.auc(fpr, tpr)
print('Test_auc_B', pr_auc)


