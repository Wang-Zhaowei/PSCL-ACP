import os
import pickle
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader

from CCI import CCI
from SSCR import SSCR
from SHTL import SHTL
from util import *

str_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

MODEL_CONFIG = {
    'SSCR': {'class': SSCR, 'index': 1, 'lr_scale': 5},
    'SHTL': {'class': SHTL, 'index': 2, 'lr_scale': 5},
    'CCI': {'class': CCI, 'index': 3, 'lr_scale': 1},
}
MODEL_ALIASES = {
    'MODEL1': 'SSCR',
    'MODEL2': 'SHTL',
    'MODEL3': 'CCI',
}


def normalize_model_name(model_name):
    name = str(model_name).upper()
    name = MODEL_ALIASES.get(name, name)
    if name not in MODEL_CONFIG:
        choices = ', '.join(MODEL_CONFIG)
        raise ValueError(f'Unknown model {model_name!r}; choose one of: {choices}.')
    return name


def checkpoint_path(data_type, model_index, round_index):
    return f'{tmp_dir}/{data_type}/model{model_index}_{round_index}'


class Runner:
    def __init__(self, model, lr=1e-4, lambda_balance=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Run on device: %s' % self.device)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
        self.lambda_balance = lambda_balance
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)


    def _unpack_forward_result(self, result):
        if isinstance(result, tuple) and len(result) == 3:
            outputs, internal, aux_loss = result
        elif isinstance(result, tuple) and len(result) == 2:
            outputs, internal = result
            aux_loss = outputs.new_tensor(0.0)
        else:
            raise ValueError('Model forward should return (outputs, internal) or (outputs, internal, aux_loss).')

        if not torch.is_tensor(aux_loss):
            aux_loss = outputs.new_tensor(float(aux_loss))
        else:
            aux_loss = aux_loss.to(outputs.device)
        return outputs, internal, aux_loss



    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        internal_list = []
        outputs_list, labels_list = [], []
        for batch in train_loader:
            batch = batch.to(self.device)
            
            x1 = batch.embed_features
            
            seqlen = batch.seqlen
            mask = batch.mask
            labels = batch.y
            graph_data = batch

            self.optimizer.zero_grad()
            
            # 根据模型类别自动匹配不同的 forward 传参维度
            if isinstance(self.model, SSCR):
                result = self.model(x1, seqlen, mask)
            elif isinstance(self.model, SHTL):
                result = self.model(graph_data)
            else:
                result = self.model(x1, seqlen, mask, graph_data)

            outputs, internal, aux_loss = self._unpack_forward_result(result)
            
            internal_list.append(internal.detach().cpu())
            cls_loss = self.criterion(outputs, labels)
            loss = cls_loss + self.lambda_balance * aux_loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            outputs_list.append(outputs.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
        train_loss /= len(train_loader)
        outputs_list = np.vstack(outputs_list)
        labels_list = np.vstack(labels_list)
        return train_loss, outputs_list, labels_list, internal_list


    def val(self, val_loader):
        self.model.eval()
        val_loss = 0
        outputs_list, labels_list = [], []
        internal_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                x1 = batch.embed_features
                seqlen = batch.seqlen
                mask = batch.mask
                labels = batch.y
                graph_data = batch
                    
                # 根据模型类别自动匹配不同的 forward 传参维度
                if isinstance(self.model, SSCR):
                    result = self.model(x1, seqlen, mask)
                elif isinstance(self.model, SHTL):
                    result = self.model(graph_data)
                else:
                    result = self.model(x1, seqlen, mask, graph_data)

                outputs, internal, aux_loss = self._unpack_forward_result(result)
                    
                internal_list.append(internal.cpu())
                cls_loss = self.criterion(outputs, labels)
                loss = cls_loss + self.lambda_balance * aux_loss
                val_loss += loss.item()
                outputs_list.append(outputs.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())
        val_loss /= len(val_loader)
        outputs_list = np.vstack(outputs_list)
        labels_list = np.vstack(labels_list)
        return val_loss, outputs_list, labels_list, internal_list

def train(train_dataset, val_dataset, model, lr, data_type, model_idx, round_idx, stop_sign = 'valid', lambda_balance=1):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
    
    best_loss, best_idx = 100, -1
    counter = 0
    train_loss_list, val_loss_list, acc_list, auc_list = [], [], [], []
    total_epoch, patience = 300, 40

    runner = Runner(model, lr, lambda_balance=lambda_balance)
    os.makedirs(f'{tmp_dir}/{data_type}', exist_ok=True)
    save_dir = checkpoint_path(data_type, model_idx, round_idx)
    with tqdm(total=total_epoch) as pbar:
        for epoch in range(total_epoch):
            train_loss, train_outputs, train_labels, train_internal = runner.train(train_loader)
            train_loss_list.append(train_loss)

            val_loss, test_outputs, test_labels, test_internal = runner.val(val_loader)
            val_loss_list.append(val_loss)
            
            acc = compute_acc(test_labels, test_outputs, data_type)
            acc_list.append(acc)
            auc = compute_auc(test_labels, test_outputs, data_type)
            auc_list.append(auc)


            # 保存最佳模型
            if stop_sign == 'train':
                loss = train_loss
            elif stop_sign == 'valid':
                loss = val_loss
                
            if loss < best_loss:
                best_loss = loss
                best_idx = epoch
                torch.save(runner.model.state_dict(), save_dir)
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    break

            pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
            pbar.update()
            if epoch % 5 == 4:
                draw_loss([train_loss_list, val_loss_list], ['train_loss', 'val_loss'])
                draw_auc_epoch([auc_list, acc_list], ['AUC', 'ACC'])
                draw_PCA([train_internal, test_internal], [train_labels[:, 1], test_labels[:, 1]], epoch + 1)
            if epoch == 10:
                show_PCA1(torch.vstack(train_internal), train_labels[:, 1], '10', f'{tmp_dir}/PCA{model_idx}_1.png')
    show_PCA1(torch.vstack(train_internal), train_labels[:, 1], 'Final', f'{tmp_dir}/PCA{model_idx}_2.png')
    print(f"Best model is from epoch {best_idx}: {round(best_loss,3)}")
    return save_dir

def predict(test_dataset, model, data_type, name, idx):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model_path = f'{tmp_dir}/{data_type}/{name}_{idx}' 
    print(f'Loaded model from {model_path}')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    runner = Runner(model)
    val_loss, outputs, labels, internal_list = runner.val(test_loader)
    os.makedirs(f'{result_dir}/{data_type}/{str_time}/{name}', exist_ok=True)
    pkl_dir = f'{result_dir}/{data_type}/{str_time}/{name}/{idx}.pkl'
    with open(pkl_dir, 'wb') as fw:
        print(f'Result saved to {pkl_dir}')
        pickle.dump([outputs, labels], fw)
    return pkl_dir


def train_model(train_dataset, val_dataset, data_type, idx, base_lr, model_name):
    name = normalize_model_name(model_name)
    config = MODEL_CONFIG[name]
    model = config['class'](data_type)

    if name == 'CCI':
        sscr_path = checkpoint_path(data_type, 1, idx)
        shtl_path = checkpoint_path(data_type, 2, idx)
        missing = [path for path in (sscr_path, shtl_path) if not os.path.exists(path)]
        if missing:
            missing_text = ', '.join(missing)
            raise FileNotFoundError(
                'CCI requires pretrained SSCR and SHTL checkpoints. '
                f'Missing: {missing_text}'
            )
        model.load_sscr(sscr_path)
        model.load_shtl(shtl_path)

    return train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        lr=config['lr_scale'] * base_lr,
        data_type=data_type,
        model_idx=config['index'],
        round_idx=idx,
        stop_sign='valid',
    )


def predict_model(test_dataset, data_type, model_name, idx=0):
    name = normalize_model_name(model_name)
    config = MODEL_CONFIG[name]
    return predict(
        test_dataset=test_dataset,
        model=config['class'](data_type),
        data_type=data_type,
        name=f"model{config['index']}",
        idx=idx,
    )
