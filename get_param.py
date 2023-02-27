import torch
ckpt_path = "./ckpt/DTDB/clouds/Stage2_DTDB_Date-2023-1-12-8-29-22_1/checkpoint_best_val.pth"
save_path = "./ckpt/DTDB/clouds/Stage2_DTDB_Date-2023-1-12-8-29-22_1/cINN.pth"

net = torch.load(ckpt_path)
new_state_dict = {}
for k,v in net['state_dict'].items():
    if 'flow' in k:
        new_state_dict[k.lstrip("flow.")] = v

print(new_state_dict)
net['state_dict'] = new_state_dict
torch.save(net, save_path)
