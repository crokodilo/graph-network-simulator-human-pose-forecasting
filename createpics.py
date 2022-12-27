from utils import h36motion3d as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from utils.parser import args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])



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



def render(input_n,output_n,path,device,skip_rate):
    
    action = "walking"
    
    loader=datasets.Datasets(path,input_n,output_n,skip_rate, device, split=2,actions=[action])
        
    loader = DataLoader(
                        loader,
                        batch_size=1,
                        shuffle = False, # for comparable visualizations with other models
                        num_workers=0)       
    

    for _, batch in enumerate(loader): 
        batch = batch.to(device) 
                                
        sequences_gt=batch[:,  input_n: input_n+ output_n, :].view(-1, output_n,32,3).permute(0,2,1,3)
        
        sequences_gt = sequences_gt.permute(0,2,1,3)
            
        data_gt=torch.squeeze(sequences_gt,0).cpu().data.numpy()/1000

        return data_gt


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

lcolor = "#8e8e8e"
rcolor = "#383838"

def mycreate_pose(ax,vals):


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

    lcolor = "#8e8e8e"
    rcolor = "#383838"

    
    for i in np.arange( len(I) ):
        x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        ax.plot(x, y, z, lw=2, linestyle='--', c=lcolor if LR[i] else rcolor)


    r = 0.5
    xroot, zroot, yroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    ax.set_zlim3d([-r+zroot, r+zroot])


def input_sequence(data_gt):
                    
    fig = plt.figure(figsize = (15,7))

    for i in range(5):
        ax = fig.add_subplot(1, 5, i+1, projection='3d')
        ax.view_init(elev=20, azim=-40)
        
        mycreate_pose(ax, data_gt[int(i*3),:])

        ax.set_axis_off()
    
    #fig.suptitle('Input Pose History', fontsize = 25)

    plt.tight_layout()
    plt.savefig('sequence.png', dpi=500)



def mycreate_graph(ax,vals):

    lcolor = rcolor
    
    for i in np.arange( len(I) ):
        x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        ax.plot(x, y, z, lw=2, marker = 'o', markerfacecolor= lcolor if LR[i] else rcolor, linestyle='--', c=lcolor if LR[i] else rcolor)


    r = 0.5
    xroot, zroot, yroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    ax.set_zlim3d([-r+zroot, r+zroot])


def mycreate_fully_connected_graph(ax,vals):

    connect = [
            (1, 2), (2, 3), (3, 4), (4, 5),
            (6, 7), (7, 8), (8, 9), (9, 10),
            (0, 1), (0, 6),
            (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
            (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (24, 25), (24, 17),
            (24, 14), (14, 15)
    ]

    nodes = np.unique([i for i,j in connect] + [j for i,j in connect] )
    
    connect = [(i,j) for i in nodes for j in nodes if i!=j]
    # Start and endpoints of our representation
    I   = np.array([touple[0] for touple in connect])
    J   = np.array([touple[1] for touple in connect])

    color = "#383838"


    for i in np.arange( len(I) ):
        x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        ax.plot(x, y, z, lw=0.2, marker = 'o', markerfacecolor= color, c=color)


    r = 0.5
    xroot, zroot, yroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    ax.set_zlim3d([-r+zroot, r+zroot])

def graph_encoder(data_gt):

    elev = 0
    azim = 90
                    
    fig = plt.figure(figsize = (15,7))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.view_init(elev=20, azim=-40)
    ax.set_axis_off()
    mycreate_pose(ax, data_gt[int(3*3),:])

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()    
    mycreate_graph(ax, data_gt[int(3*3),:])


    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    mycreate_fully_connected_graph(ax, data_gt[int(3*3),:])


    
    fig.suptitle('Encoder', fontsize = 25)

    plt.tight_layout()
    plt.savefig('encoder.png', dpi=500)


def graph_decoder(data_gt):
    
    elev = 0
    azim = 90
                    
    fig = plt.figure(figsize = (15,7))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    mycreate_fully_connected_graph(ax, data_gt[int(3*3),:])



    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    mycreate_graph(ax, data_gt[int(4*3),:])


    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.view_init(elev=20, azim=-40)
    ax.set_axis_off()
    mycreate_pose(ax, data_gt[int(4*3),:])
    
    fig.suptitle('Decoder', fontsize = 25)

    plt.tight_layout()
    plt.savefig('decoder.png', dpi=500)

if __name__ == '__main__':
    data = render(args.input_n,args.output_n,args.data_dir,device, args.skip_rate)
    input_sequence(data)
    graph_encoder(data)
    graph_decoder(data)






