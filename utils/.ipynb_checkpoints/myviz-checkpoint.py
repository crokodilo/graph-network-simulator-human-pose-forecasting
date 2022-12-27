#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import h36motion3d as datasets
from utils.loss_funcs import mpjpe_error
from utils.data_utils import define_actions
import pickle


def create_pose(ax,plots,vals,pred=True,update=False):

            
    
    # h36m 32 joints(full)
    connect = [
            (1, 2), (2, 3), (3, 4), (4, 5),
            (6, 7), (7, 8), (8, 9), (9, 10),
            (0, 1), (0, 6),
            (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
            (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (24, 25), (24, 17),
            (24, 14), (14, 15)
    ]
    LR = [
            False, True, True, True, True,
            True, False, False, False, False,
            False, True, True, True, True,
            True, True, False, False, False,
            False, False, False, False, True,
            False, True, True, True, True,
            True, True
    ]  


# Start and endpoints of our representation
    I   = np.array([touple[0] for touple in connect])
    J   = np.array([touple[1] for touple in connect])
# Left / right indicator
    LR  = np.array([LR[a] or LR[b] for a,b in connect])
    if pred:
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
    else:
        lcolor = "#8e8e8e"
        rcolor = "#383838"

    for i in np.arange( len(I) ):
        x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        if not update:

            if i ==0:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--' ,c=lcolor if LR[i] else rcolor,label=['GT' if not pred else 'Pred']))
            else:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--', c=lcolor if LR[i] else rcolor))

        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)
    
    return plots
   # ax.legend(loc='lower left')




def rollout(simulator, initial_positions, nsteps):
    """Rolls out a trajectory by applying the model in sequence.

    Args:
    simulator: Learned simulator.
    features: Torch tensor features.
    nsteps: Number of steps.
    """

    current_positions = initial_positions
    predictions = []

    for step in range(nsteps):
        # Get next position with shape (nnodes, dim)
        next_position = simulator.predict_positions(current_positions)

        predictions.append(next_position)

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)

    # Predictions with shape (nnodes, time, dim)

    return torch.stack(predictions, dim=1)


def render(input_n,output_n,visualize_from,path,modello,device,n_viz,skip_rate,actions):
    input_n += 1
    actions=define_actions(actions)
    
    for action in actions:
    
        if visualize_from=='train':
            loader=datasets.Datasets(path,input_n,output_n,skip_rate, device, split=0,actions=[action])
        elif visualize_from=='validation':
            loader=datasets.Datasets(path,input_n,output_n,skip_rate, device, split=1,actions=[action])
        elif visualize_from=='test':
            loader=datasets.Datasets(path,input_n,output_n,skip_rate, device, split=2,actions=[action])
            
        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                        46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                        75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
                        
      # joints at same loc ----> joints at index_to_ignore are equal to joints at index_to_equal (this is my guess)
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
            
            
        loader = DataLoader(
                            loader,
                            batch_size=1,
                            shuffle = False, # for comparable visualizations with other models
                            num_workers=0)       
        
    
        for cnt, batch in enumerate(loader): 
            batch = batch.to(device) 
            
            batch_dim = batch.shape[0]
            
            all_joints_seq=batch.clone()[:, input_n:input_n+output_n,:]
            
            
            sequences_train=batch[:, 0: input_n, dim_used].view(-1, input_n,len(dim_used)//3,3).permute(0,2,1,3)

            sequences_gt=batch[:,  input_n: input_n+ output_n, :].view(-1, output_n,32,3).permute(0,2,1,3)

            next_position = batch[:,  input_n: input_n+ output_n, dim_used].view(-1, output_n,len(dim_used)//3,3).permute(0,2,1,3)
            
            pred_vel, target_vel = modello.predict_velocities(
                    next_positions=next_position.reshape(-1,  output_n, 3)[:,0].to(device),
                    position_sequence=sequences_train.reshape(-1,  input_n, 3).to(device))
            
            loss = (pred_vel - target_vel) ** 2
            loss = loss.mean()

            pred_sequence = rollout(modello, sequences_train.reshape(-1,  input_n, 3),  output_n).reshape(batch_dim, len(dim_used)//3,  output_n, 3)
            
            all_joints_seq[:,:,dim_used] = pred_sequence.permute(0,2,1,3).reshape(batch_dim,  output_n, len(dim_used))

            all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]

            all_joints_seq = all_joints_seq.reshape(batch_dim,  output_n, 32, 3)
            sequences_gt = sequences_gt.permute(0,2,1,3)
            
            mpjperror = mpjpe_error(all_joints_seq, sequences_gt)
    
            data_pred=torch.squeeze(all_joints_seq,0).cpu().data.numpy()/1000 # in meters
            data_gt=torch.squeeze(sequences_gt,0).cpu().data.numpy()/1000
    
            with open(f'rollouts/rollout_pred_{action}_out_{output_n}.pkl', 'wb') as f:
                pickle.dump(data_pred, f)

            with open(f'rollouts/rollout_gt_{action}_out_{output_n}.pkl', 'wb') as f:
                pickle.dump(data_gt, f)
                
            with open(f'rollouts/loss_{action}_out_{output_n}.pkl', 'wb') as f:
                pickle.dump(mpjperror, f)
                
            if cnt == n_viz-1:
                break


def mycreate_pose(ax,plots,vals,pred=True,update=False):
    
            
    
    # h36m 32 joints(full)
    connect = [
            (1, 2), (2, 3), (3, 4), (4, 5),
            (6, 7), (7, 8), (8, 9), (9, 10),
            (0, 1), (0, 6),
            (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
            (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (24, 25), (24, 17),
            (24, 14), (14, 15)
    ]
    LR = [
            False, True, True, True, True,
            True, False, False, False, False,
            False, True, True, True, True,
            True, True, False, False, False,
            False, False, False, False, True,
            False, True, True, True, True,
            True, True
    ]  


# Start and endpoints of our representation
    I   = np.array([touple[0] for touple in connect])
    J   = np.array([touple[1] for touple in connect])
# Left / right indicator
    LR  = np.array([LR[a] or LR[b] for a,b in connect])
    if pred:
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
    else:
        lcolor = "#8e8e8e"
        rcolor = "#383838"

    for i in np.arange( len(I) ):
        x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        if not update:

            if i ==0:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--' ,c=lcolor if LR[i] else rcolor,label=['GT' if not pred else 'Pred']))
            else:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--', c=lcolor if LR[i] else rcolor))

        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)
    
    return plots
   # ax.legend(loc='lower left')


# In[11]:


def myupdate(num,data_gt,data_pred,plots_gt,plots_pred,fig,ax, ax1):
    
    gt_vals=data_gt[num]
    pred_vals=data_pred[num]
    plots_gt=create_pose(ax,plots_gt,gt_vals,pred=False,update=True)
    plots_pred=create_pose(ax1,plots_pred,pred_vals,pred=True,update=True)
    
    
    r = 0.5
    xroot, zroot, yroot = gt_vals[0,0], gt_vals[0,1], gt_vals[0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    ax.set_zlim3d([-r+zroot, r+zroot])
    
    
    ax1.set_xlim3d([-r+xroot, r+xroot])
    ax1.set_ylim3d([-r+yroot, r+yroot])
    ax1.set_zlim3d([-r+zroot, r+zroot])
    
    #ax.set_title('pose at time frame: '+str(num))
    #ax.set_aspect('equal')
 
    return plots_gt,plots_pred

def myvisualize(output_n, actions):
    
    actions=define_actions(actions)
    
    for action in actions:
    
        with open(f'rollouts/rollout_pred_{action}_out_{output_n}.pkl', "rb") as file:
            data_pred = pickle.load(file)
        
        with open(f'rollouts/rollout_gt_{action}_out_{output_n}.pkl', "rb") as file:
            data_gt = pickle.load(file)
            
        with open(f'rollouts/loss_{action}_out_{output_n}.pkl', "rb") as file:
            loss = pickle.load(file) 
            
        rounded_loss = round(loss.item(), 3)            
        fig = plt.figure()
        fig.suptitle(f'Prediction for action {action} for {output_n} frames (loss in mm is {rounded_loss})', fontsize=12, y=0.9)
        
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.view_init(elev=20, azim=-40)
        vals = np.zeros((32, 3)) # or joints_to_consider
        gt_plots=[]

        gt_plots=mycreate_pose(ax,gt_plots,vals,pred=False,update=False)

        ax.set_title('Ground truth', fontsize = 10, y=-0.2)
        
        ax.set_axis_off()
        
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.view_init(elev=20, azim=-40)
        vals = np.zeros((32, 3)) # or joints_to_consider

        pred_plots=[]
        pred_plots=mycreate_pose(ax1,pred_plots,vals,pred=True,update=False)
        
        
        ax1.set_axis_off()

        ax1.set_title('Prediction', fontsize = 10, y=-0.2)
        
        
        line_anim = animation.FuncAnimation(fig, myupdate, output_n, fargs=(data_gt,data_pred,gt_plots,pred_plots,
                                                                    fig, ax, ax1), interval=70, blit=False)
        #plt.show()
        
        line_anim.save(f'gifs/{action}.gif', writer='pillow')