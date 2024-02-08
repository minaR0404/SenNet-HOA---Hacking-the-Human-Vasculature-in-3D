
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel 
from torch.cuda.amp import autocast

from config.setting import CFG
from utils.data import load_data, Kaggle_Dataset
from utils.function import norm_with_clip, add_noise
from utils.loss import DiceLoss, dice_coef, loss_func
from utils.model import build_model


#-----------------------------------------------------------------------------------------------
### main
#-----------------------------------------------------------------------------------------------
if __name__ == '__main__':

    if CFG.DataMode == 'kidney_1':
        #-----------------------------------------------------------------------------------------------
        ### Loader
        #-----------------------------------------------------------------------------------------------
        train_x = []
        train_y = []

        root = "./data/blood-vessel-segmentation"
        paths = [f"{root}/train/kidney_1_dense", ]

        for i, path in enumerate(paths):
            if path == f"{root}/train/kidney_3_dense":
                continue

            glob_images = glob(f"{path}/images/*")
            glob_images = [glob_images[i].replace('\\', '/') for i in range(len(glob_images))]
            glob_labels = glob(f"{path}/labels/*")
            glob_labels = [glob_labels[i].replace('\\', '/') for i in range(len(glob_labels))]

            x = load_data(glob_images, is_label=False)
            y = load_data(glob_labels, is_label=True)
            
            train_x.append(x)
            train_y.append(y)
            
            train_x.append(x.permute(1,2,0))
            train_y.append(y.permute(1,2,0))
            train_x.append(x.permute(2,0,1))
            train_y.append(y.permute(2,0,1))
            
        path_sp = f"{root}/train/kidney_3_sparse"
        path_de = f"{root}/train/kidney_3_dense"

        path_de_y = glob(f"{path_de}/labels/*")
        path_de_x = [x.replace("labels", "images").replace("dense", "sparse") for x in path_de_y]

        valid_x = load_data(path_de_x, is_label=False)
        valid_y = load_data(path_de_y, is_label=True)



        #-----------------------------------------------------------------------------------------------
        ### Training
        #-----------------------------------------------------------------------------------------------
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        train_dataset = Kaggle_Dataset(train_x, train_y, arg=True)
        train_dataset = DataLoader(train_dataset, batch_size=CFG.train_batch_size, num_workers=2, shuffle=True, pin_memory=True)

        valid_dataset = Kaggle_Dataset([valid_x], [valid_y])
        valid_dataset = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, num_workers=2, shuffle=False, pin_memory=True)

        model = build_model()
        model = DataParallel(model)

        #loss_func = DiceLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = CFG.lr,
                                                    steps_per_epoch = len(train_dataset),
                                                    epochs = CFG.epochs + 1,
                                                    pct_start = 0.1,)
        best_score = 0
        
        for epoch in range(CFG.epochs):
            
            model.train()
            
            time = tqdm(range(len(train_dataset)))
            losses = 0
            scores = 0
            
            for i, (x,y) in enumerate(train_dataset):
                x = x.cuda().to(torch.float32)
                y = y.cuda().to(torch.float32)
                
                x = norm_with_clip(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
                x = add_noise(x, max_randn_rate=0.5, x_already_normed=True)
                
                with autocast():
                    pred = model(x)
                    loss = loss_func(pred, y)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad()
                scheduler.step()
                
                score = dice_coef(pred.detach(), y)
                scores = (scores * i + score) / (i + 1)
                losses = (losses * i + loss.item()) / (i + 1)
                
                time.set_description(f"epoch:{epoch}, loss:{losses:.4f}, score:{scores:.4f}, lr:{optimizer.param_groups[0]['lr']:.4e}")
                time.update()
                
                del loss, pred
                
            time.close()
            
            
            model.eval()
            
            time = tqdm(range(len(valid_dataset)))
            valid_losses = 0
            valid_scores = 0
            
            for i, (x,y) in enumerate(valid_dataset):
                x = x.cuda().to(torch.float32)
                y = y.cuda().to(torch.float32)
                
                x = norm_with_clip(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
                
                with autocast():
                    with torch.no_grad():
                        pred = model(x)
                        loss = loss_func(pred, y)
                        
                score = dice_coef(pred.detach(), y)
                valid_scores = (valid_scores * i + score) / (i + 1)
                valid_losses = (valid_losses * i + loss.item()) / (i + 1)
                
                time.set_description(f"val-->loss:{valid_losses:.4f}, score:{valid_scores:.4f}")
                time.update()

            if valid_scores > best_score:
                torch.save(model.module.state_dict(),f"./pth/{CFG.backbone}_3ch_model_real_best.pt")
                best_score = valid_scores
                
            time.close()
            
        torch.save(model.module.state_dict(), f"./pth/{CFG.backbone}_epoch{epoch+1}_loss{losses:.2f}_score{scores:.2f}_valid_loss{valid_losses:.2f}_valid_score{valid_scores:.2f}.pt")

        time.close()

    elif CFG.DataMode == 'kidney_1_3':
        #-----------------------------------------------------------------------------------------------
        ### Loader
        #-----------------------------------------------------------------------------------------------
        train_x=[]
        train_y=[]

        root = "./data/blood-vessel-segmentation"
        paths = [f"{root}/train/kidney_1_dense", ]

        for i, path in enumerate(paths):
            glob_images = glob(f"{path}/images/*")
            glob_images = [glob_images[i].replace('\\', '/') for i in range(len(glob_images))]
            glob_labels = glob(f"{path}/labels/*")
            glob_labels = [glob_labels[i].replace('\\', '/') for i in range(len(glob_labels))]

            x = load_data(glob_images, is_label=False)
            y = load_data(glob_labels, is_label=True)
            
            train_x.append(x)
            train_y.append(y)
            
            train_x.append(x.permute(1,2,0))
            train_y.append(y.permute(1,2,0))
            train_x.append(x.permute(2,0,1))
            train_y.append(y.permute(2,0,1))


        train_x3 = []
        train_y3 = []

        p_lab85 = sorted(glob( f"{root}/train/kidney_3_sparse/labels/*"))
        p_lab85 = [p_lab85[i].replace('\\', '/') for i in range(len(p_lab85))]
        p_lab = sorted(glob( f"{root}/train/kidney_3_dense/labels/*"))
        p_lab = [p_lab[i].replace('\\', '/') for i in range(len(p_lab))]
        p_lab_concatenated = p_lab85[0:496] + p_lab + p_lab85[997:]
            
        p_img = sorted(glob( f"{root}/train/kidney_3_sparse/images/*" ))
        p_img = [p_img[i].replace('\\', '/') for i in range(len(p_img))]

        x = load_data(p_img, is_label=False)
        y = load_data(p_lab_concatenated, is_label=True) 

        train_x3.append(x)
        train_y3.append(y)

        train_x3.append(x.permute(1,2,0))
        train_y3.append(y.permute(1,2,0))
        train_x3.append(x.permute(2,0,1))
        train_y3.append(y.permute(2,0,1))


        path_de = f"{root}/train/kidney_3_dense"

        path_de_y = glob(f"{path_de}/labels/*")
        path_de_x = [x.replace("labels", "images").replace("dense", "sparse") for x in path_de_y]

        valid_x = load_data(path_de_x, is_label=False)
        valid_y = load_data(path_de_y, is_label=True)
        


        #-----------------------------------------------------------------------------------------------
        ### Training
        #-----------------------------------------------------------------------------------------------
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        train_dataset1 = Kaggle_Dataset(train_x, train_y, arg=True)
        train_dataset1 = DataLoader(train_dataset1, batch_size=CFG.train_batch_size, num_workers=2, shuffle=True, pin_memory=True)

        train_dataset3 = Kaggle_Dataset(train_x3, train_y3, arg=True)
        train_dataset3 = DataLoader(train_dataset3, batch_size=CFG.train_batch_size, num_workers=2, shuffle=True, pin_memory=True)

        valid_dataset = Kaggle_Dataset([valid_x], [valid_y])
        valid_dataset = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, num_workers=2, shuffle=False, pin_memory=True)

        len_train = len(train_dataset1) + len(train_dataset3)

        model=build_model()
        model=DataParallel(model)

        loss_fc=DiceLoss()
        #loss_fn=nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(),lr=CFG.lr)
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = CFG.lr,
                                                        steps_per_epoch = len_train , epochs = CFG.epochs+1,
                                                        pct_start = 0.1,)
        best_score = 0

        for epoch in range(CFG.epochs):
            model.train()
            time=tqdm(range(len_train))
            losss=0
            scores=0
            
            for train_dataset in [train_dataset1, train_dataset3]:
                for i,(x,y) in enumerate(train_dataset):
                
                    x=x.cuda().to(torch.float32)
                    y=y.cuda().to(torch.float32)
                    x=norm_with_clip(x.reshape(-1,*x.shape[2:])).reshape(x.shape)
                    x=add_noise(x,max_randn_rate=0.5,x_already_normed=True)
                
                    with autocast():
                        pred=model(x)
                        loss=loss_fc(pred,y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    score=dice_coef(pred.detach(),y)
                    losss=(losss*i+loss.item())/(i+1)
                    scores=(scores*i+score)/(i+1)
                    time.set_description(f"epoch:{epoch},loss:{losss:.4f},score:{scores:.4f},lr{optimizer.param_groups[0]['lr']:.4e}")
                    time.update()
                    del loss,pred
            time.close()
            
            model.eval()
            time=tqdm(range(len(valid_dataset)))
            val_losss=0
            val_scores=0
            for i,(x,y) in enumerate(valid_dataset):
                x=x.cuda().to(torch.float32)
                y=y.cuda().to(torch.float32)
                x=norm_with_clip(x.reshape(-1,*x.shape[2:])).reshape(x.shape)

                with autocast():
                    with torch.no_grad():
                        pred=model(x)
                        loss=loss_fc(pred,y)
                score=dice_coef(pred.detach(),y)
                val_losss=(val_losss*i+loss.item())/(i+1)
                val_scores=(val_scores*i+score)/(i+1)
                time.set_description(f"val-->loss:{val_losss:.4f},score:{val_scores:.4f}")
                time.update()
                
                if val_scores > best_score:
                    torch.save(model.module.state_dict(),f"{CFG.backbone}_model_real_best.pt")
                    best_score = val_scores

            time.close()
        torch.save(model.module.state_dict(),f"./{CFG.backbone}_epoch{epoch}_loss{losss:.2f}_score{scores:.2f}_val_loss{val_losss:.2f}_val_score{val_scores:.2f}_midd_rot002.pt")

        time.close()