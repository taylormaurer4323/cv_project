# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:25:12 2022

@author: taylo
"""
from mobilestereonet.models import __models__, model_loss

from mobilestereonet.utils import *
import time
import torch
import gc
import matplotlib.pyplot as plt

def train(model, optimizer, TrainImgLoader, logger, logdir, start_epoch = 0, epochs = 1, lrepochs= "200:10", summary_freq = 10, 
    learning_rate = .001, train_limit = 100000, max_disp = 192, save_freq = 1, TestImgLoader=0):
    
    best_checkpoint_loss = 100
    for epoch_idx in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch_idx, learning_rate, lrepochs)
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % summary_freq == 0
            
            loss, scalar_outputs, image_outputs = train_sample(model, optimizer, sample, compute_metrics=do_summary, max_disp=max_disp)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, epochs,
                                                                                        batch_idx,
                                                                                        len(TrainImgLoader), loss,
                                                                                        time.time() - start_time))
            if batch_idx > train_limit:
                break  
        #Saving checkopoints
        if (epoch_idx + 1)  % save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(logdir, epoch_idx))

        gc.collect()
        
        #Test and possibly resave...
        if TestImgLoader != 0:
            avg_test_scalars = AverageMeterDict()
            for batch_idx, sample in enumerate(TestImgLoader):
                global_step = len(TestImgLoader) * epoch_idx + batch_idx
                start_time = time.time()
                do_summary = global_step % 10 == 0
                loss, scalar_outputs, image_outputs = test_sample(model, sample, max_disp, compute_metrics=do_summary)
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)
                    save_images(logger, 'test', image_outputs, global_step)
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs
                print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, epochs,
                                                                                         batch_idx,
                                                                                         len(TestImgLoader), loss,
                                                                                         time.time() - start_time))
            avg_test_scalars = avg_test_scalars.mean()
            if avg_test_scalars['loss'] < best_checkpoint_loss:
                best_checkpoint_loss = avg_test_scalars['loss']
                print("Overwriting best checkpoint")
                checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/best.ckpt".format(logdir))


# train one sample
def train_sample(model, optimizer, sample, max_disp = 192, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']

    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()
    disp_ests = model(imgL, imgR)
    mask = (disp_gt < max_disp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs
@make_nograd_func
def test_sample(model, sample, maxdisp, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs
if __name__ == '__main__':
    train()
    