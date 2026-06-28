import torch

import argparse

def get_autoencoder_state_dict(state_dict):

    new_state_dict = {}

    for k, v in state_dict.items():
        if 'first_stage_model' in k:
            new_k = k.replace('first_stage_model.', '')
            new_state_dict[new_k] = v
            print(new_k)
    
    return new_state_dict

def get_ldm_state_dict(state_dict):

    for k, v in state_dict.items():
        if 'first_stage_model' in k:
            state_dict.pop(k)

    return state_dict

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--type', type=str, choices=['autoencoder', 'ldm'], required=True)

    args = parser.parse_args()

    state = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    state_dict = state['state_dict']

    if args.type == 'autoencoder':
        state_dict = get_autoencoder_state_dict(state_dict)
    elif args.type == 'ldm':
        state_dict = get_ldm_state_dict(state_dict)

    state_dict = {
        'state_dict': state_dict,
    }
    torch.save(state_dict, args.output_path)

if __name__ == "__main__":
    main()