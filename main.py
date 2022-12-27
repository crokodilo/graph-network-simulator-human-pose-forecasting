import os
from re import X
from utils import h36motion3d as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from utils.loss_funcs import *
from utils.data_utils import define_actions
from utils.myviz import render, myvisualize
from utils.parser import args
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

torch.manual_seed(args.seed)


dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

def collate_fn(data):
    positions_list = []
    for positions in data:
        positions_list.append(positions[:, dim_used].reshape(-1,len(dim_used)//3,3))
    return torch.tensor(np.concatenate(positions_list, axis= 1)).to(torch.float32).contiguous().permute(1,0,2)


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


    

def train(mps=10, cr=np.Inf, nl=2, hp_tuning=0):
    from gnn import simulator
    
    model = simulator.get_simulator(args.input_n, device, nmessage_passing_steps=mps, nmlp_layers=nl, connectivity_radius=cr).to(device)

    print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if hp_tuning!=0:
      print('number of message passing steps: '+str(mps))
      print('number of mlp layers: '+str(nl))
      print('connectivity radius: '+str(cr), ' (mm)')

    if hp_tuning==0:
        model_name=f'autoregressive_{args.input_n}_ckpt'
    else:
        model_name=f'autoregressive_input_{args.input_n}_mps_{mps}_cr_{cr}_nl_{nl}_ckpt'

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)
    
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    train_loss = []
    val_loss = []
    val_mpjperror = []

    dataset = datasets.Datasets(args.data_dir,args.input_n, args.output_n,args.skip_rate, device, split=0)
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn = collate_fn)

    vald_dataset = datasets.Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate, device, split=1)
    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
    vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn = collate_fn)

    results = {}

    results['training_loss'] = []
    results['validation_loss'] = []
    results['train_mpjperror'] = []
    results['val_mpjperror'] = []

    for epoch in range(args.n_epochs):
        running_loss=0
        n=0
        model.train()
        for cnt, batch in enumerate(data_loader):
            batch=batch.to(device)
            batch_dim=batch.shape[0]//(len(dim_used)//3)
            n+=batch_dim
            
            sequences_train=batch[:, 0:args.input_n, :]
            sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :]

            optimizer.zero_grad()

            pred_vel, target_vel = model.predict_velocities(
                    next_positions=sequences_gt[:,0].view(-1,3).to(device),
                    position_sequence=sequences_train.to(device))
            
            loss = (pred_vel - target_vel) ** 2
            loss = loss.mean()

            if cnt % 200 == 0:
                print('[%d, %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item())) 

            loss.backward()  
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

            optimizer.step()
            running_loss += loss*batch_dim
        
        train_loss.append(running_loss.detach().cpu().item()/n)


        model.eval()
        with torch.no_grad():
            running_mjperror = 0
            running_loss=0
            n=0
            for cnt,batch in enumerate(vald_loader):
                batch=batch.to(device)
                batch_dim=batch.shape[0]//(len(dim_used)//3)
                n+=batch_dim
                
                sequences_train=batch[:, 0:args.input_n, :]
                sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :]

                pred_vel, target_vel = model.predict_velocities(
                        next_positions=sequences_gt[:,0].view(-1,3).to(device),
                        position_sequence=sequences_train.to(device))
                
                loss = (pred_vel - target_vel) ** 2
                loss = loss.mean()

                if cnt % 200 == 0:
                    print('[%d, %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item())) 
                
                pred_sequence = rollout(model, sequences_train, args.output_n)
                mpjperror = mpjpe_error(pred_sequence, sequences_gt)
                
                running_mjperror += mpjperror * batch_dim
                running_loss += loss * batch_dim
            
            val_loss.append(running_loss.detach().cpu().item() / n)
            val_mpjperror.append(running_mjperror.detach().cpu().item() / n)

        results['training_loss'] = train_loss
        results['validation_loss'] = val_loss
        results['val_mpjperror'] = val_mpjperror
    
        if hp_tuning==0:
            with open(os.path.join(args.model_path, f'loss_{args.input_n}.json'), 'w') as f:
                json.dump(results, f)
        else:
            with open(os.path.join(args.model_path, f'loss_input_{args.input_n}_mps_{mps}_cr_{cr}_nl_{nl}.json'), 'w') as f:
                json.dump(results, f)

        if args.use_scheduler:
            scheduler.step()

        if (epoch+1)%10==0:
            print('----saving model-----')
            torch.save(model.state_dict(), os.path.join(args.model_path, model_name+'_epoch'+str(epoch+1)))

    plt.figure()
    plt.plot(train_loss, 'r', label='Train loss')
    plt.plot(val_loss, 'g', label='Val loss')
    plt.legend()
    
    if hp_tuning==0:
        plt.savefig(os.path.join(args.model_path, f'loss_{args.input_n}.png'))
    else:
        plt.savefig(os.path.join(args.model_path, f'loss_input_{args.input_n}_mps_{mps}_cr_{cr}_nl_{nl}.png'))

    plt.figure()
    plt.plot(val_mpjperror, 'g', label='Val MPJPE')
    plt.legend()
    if hp_tuning==0:
        plt.savefig(os.path.join(args.model_path, f'val_mpjpe_{args.input_n}.png'))
    else:
        plt.savefig(os.path.join(args.model_path, f'val_mpjpe_input_{args.input_n}_mps_{mps}_cr_{cr}_nl_{nl}.png'))

    return  results['val_mpjperror'][-1]


def test(model, model_name):  
    
    actions = define_actions(args.actions_to_consider)
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
    
    print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
    model.eval()

    for lookahead in [2, 4, 8, 10, 14, 18, 22, 25]:
        args.output_n = lookahead
        
        accum_loss=0
        n_batches=0 # number of batches for all the sequences
        results = {}

        for action in actions:
            running_loss=0
            n=0
            dataset_test = datasets.Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate, device, split=2,actions=[action])
            print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))

            test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
            for cnt, batch in enumerate(test_loader):
                with torch.no_grad():

                    batch=batch.to(device)
                    batch_dim=batch.shape[0]
                    n+=batch_dim
                    
                    all_joints_seq=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]

                    sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,2,1,3)

                    sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :].view(-1,args.output_n,32,3).permute(0,2,1,3)

                    next_position = batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//3,3).permute(0,2,1,3)
                    
                    pred_vel, target_vel = model.predict_velocities(
                            next_positions=next_position.reshape(-1, args.output_n, 3)[:,0].to(device),
                            position_sequence=sequences_train.reshape(-1, args.input_n, 3).to(device))
                    
                    loss = (pred_vel - target_vel) ** 2
                    loss = loss.mean()

                    pred_sequence = rollout(model, sequences_train.reshape(-1, args.input_n, 3), args.output_n).reshape(batch_dim, len(dim_used)//3, args.output_n, 3)
                    
                    all_joints_seq[:,:,dim_used] = pred_sequence.permute(0,2,1,3).reshape(batch_dim, args.output_n, len(dim_used))

                    all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]

                    mpjperror = mpjpe_error(all_joints_seq.reshape(batch_dim, args.output_n, 32, 3), sequences_gt.permute(0,2,1,3))

                    running_loss+=mpjperror*batch_dim
                    accum_loss+=mpjperror*batch_dim
                
                print('loss at test subject for action : '+str(action)+ ' is: '+ str(running_loss/n))
                results[action] = running_loss.cpu().detach().item()/n
            n_batches+=n
        print('overall average loss in mm is: '+str(accum_loss/n_batches))
        results['overall'] = accum_loss.cpu().detach().item()/n_batches

        with open(os.path.join(args.output_path, f'dict_in{args.input_n}_out{args.output_n}.json'), 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    if args.mode == 'train':
        val_mpjpe_final = train(args.n_mps, args.cradius, args.nmlp_layers, 0)
        
    elif args.mode == 'hptuning':
        val_min = np.inf
        d = {}
        nmessage_passing_steps = [1, 5, 10, 12]
        connectivity_radius = [0, 500, 1000, np.inf]
        nmlp_layers = [2, 3]
        for mps in nmessage_passing_steps:
            for cr in connectivity_radius:
                for nl in nmlp_layers:
                    val_mpjpe_final = train(mps, cr, nl, 1)
                    d[f'mps_{mps}_cr_{cr}_nl_{nl}'] = val_mpjpe_final
                    if val_mpjpe_final < val_min:
                        val_min = val_mpjpe_final
                        mps_min=mps
                        cr_min=cr
                        nl_min=nl
        with open(os.path.join(args.model_path, 'hpt_dict'), 'w') as f:
            json.dump(d, f)
        print(f'Best model parameters: mps= {mps_min}, cr= {cr_min}, nl={nl_min}')
        

    elif args.mode == 'test':
        from gnn import simulator
        
        model = simulator.get_simulator(args.input_n, device, nmessage_passing_steps=args.n_mps, nmlp_layers=args.nmlp_layers, connectivity_radius=args.cradius).to(device)
        
        model_name = f'autoregressive_input_{args.input_n}_mps_{args.n_mps}_cr_{args.cradius}_nl_{args.nmlp_layers}_ckpt_epoch{args.epoch_test}'
        
        test(model, model_name)
    
    elif args.mode=='render':
        from gnn import simulator
          
        model = simulator.get_simulator(args.input_n, device, nmessage_passing_steps=args.n_mps, nmlp_layers=args.nmlp_layers, connectivity_radius=args.cradius).to(device)
        model_name = f'autoregressive_input_{args.input_n}_mps_{args.n_mps}_cr_{args.cradius}_nl_{args.nmlp_layers}_ckpt_epoch{args.epoch_test}'
        print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
        model.eval()
        render(args.input_n,args.output_n,args.visualize_from,args.data_dir,model,device,args.n_viz,args.skip_rate,args.actions_to_consider)
       
    elif args.mode=='viz':
        myvisualize(args.output_n, args.actions_to_consider)



