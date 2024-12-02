import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import matplotlib.pyplot as plt
import time

import torch

from src.stable_diffusion.inverse_stable_diffusion import InversableStableDiffusionPipeline 
from diffusers import DPMSolverMultistepScheduler
from src.utils.optim_utils import *
from src.utils.io_utils import *
import torch

# optim_utils include "get_watermarking_pattern" that performs similar job
# This is modified version to be applicable on watermark detection
def get_watermarking_patterns(pipe, args, device, shape=None, option=None):
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    if 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch_tmp = copy.deepcopy(gt_patch)
        
        keys = torch.Tensor(
            [[1.1, 0.8, 1.2, 0.9, 0.6, 1.4],
             [1.1, 0.6, 1.1, 1.2, 0.8, 1.2],
             [0.7, 1.2, 1.3, 0.8, 0.9, 1.1],
             [1.2, 0.8, 1.2, 1.1, 0.6, 1.1],
            ]

        )
        keys = (keys - 1)*2 + 1
        const = 80

        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                if option is None:
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
                else:
                    gt_patch[:, j, tmp_mask] = torch.complex(const*keys[option][i-1], torch.Tensor([0]))

    return gt_patch

def main(args):
    if args.with_tracking:
        wandb.init(project='watermark_detection', name=args.run_name)
        wandb.config.update(args)
        table = wandb.Table(columns=['prompt', 'gen_w1', 'gen_w2', 'gen_w3', 'w_metric11', 'w_metric12', 'w_metric13', 'w_metric21', 'w_metric22', 'w_metric23', 
        'w_metric31', 'w_metric32', 'w_metric33'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler(
        beta_end = 0.012,
        beta_schedule = 'scaled_linear',
        beta_start = 0.00085,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        trained_betas = None,
        solver_order = args.solver_order,
        #steps_offset = 1, # depends on diffusers version
        )

    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        )
    pipe = pipe.to(device)

    # dataset
    dataset, prompt_key = get_dataset(args.dataset)

    # Create patchs
    gt_patches = []
    for i in range(args.target_num):
        gt_patches.append(get_watermarking_patterns(pipe, args, device, option=i))

    w_metrics = [[[], [], []],[[], [], []],[[], [], []]]
    no_w_metrics = [[[], [], []],[[], [], []],[[], [], []]]

    # Indexing process
    for ii in tqdm(range(args.start, args.end)):

        seed = ii + args.gen_seed
        current_prompt = dataset[ii][prompt_key]
        
        if not args.prompt_unknown:
            text_embeddings_tuple = pipe.encode_prompt(
                current_prompt, 'cuda', 1, args.guidance_scale > 1.0, None)
            text_embeddings = torch.cat([text_embeddings_tuple[1], text_embeddings_tuple[0]])
            guidance_scale_for_inv = args.guidance_scale
        else:
            tester_prompt = ''
            text_embeddings = pipe.get_text_embedding(tester_prompt)
            guidance_scale_for_inv = 1.0

        # Setup empty arrays for each case
        init_latents_no_w_array = []
        init_latents_w_array = []
        watermarking_masks = []
        init_latents_ws = []
        ffts = []

        ## 1. Generation
        # Generation without watermarks
        set_random_seed(seed)

        for i in range(args.target_num):
            init_latents_no_w_array.append(pipe.get_random_latents())
            init_latents_w_array.append(copy.deepcopy(init_latents_no_w_array[-1]))

        # Get watermarking masks
        for i in range(args.target_num):
            watermarking_masks.append(get_watermarking_mask(init_latents_no_w_array[i], args, device))

        # Inject watermark
        for i in range(args.target_num):
            temp_init_latents_w, temp_fft = inject_watermark(init_latents_w_array[i], watermarking_masks[i], gt_patches[i], args)
            init_latents_ws.append(temp_init_latents_w)
            ffts.append(temp_fft)

        # Create image
        orig_image_ws = []
        for i in range(args.target_num):
            temp_outputs_w, temp_latents_w = pipe(
                current_prompt,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_ws[i],
            )
            orig_image_ws.append(temp_outputs_w.images[0])

        ## 2. Reversing image process. Reverse process is done with "forward_diffusion".
        img_ws = []
        image_latents_ws = []
        reversed_latents_ws = []
        runtimes = []
        peak_memories = []

        for i in range(args.target_num):
            start_time = time.time()
            torch.cuda.reset_max_memory_allocated()
            img_w = transform_img(orig_image_ws[i]).unsqueeze(0).to(text_embeddings.dtype).to(device)

            if args.mode == 'encoder':
                image_latents_w = pipe.get_image_latents(img_w, sample=False)
            elif args.mode == 'no_grad':
                image_latents_w = pipe.decoder_inv_nograd(img_w, num_steps=args.dinv_steps,float=True, LR_scheduling=True)
            elif args.mode == 'grad':
                image_latents_w = pipe.decoder_inv(img_w, num_steps=args.dinv_steps,LR_scheduling=True)
            else:
                raise("args.mode shoule be `encoder` or `no_grad` or `grad`")

        
            # forward_diffusion -> inversion
            reversed_latents_w = pipe.forward_diffusion(
                latents=image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale_for_inv,
                num_inference_steps=args.test_num_inference_steps,
                inverse_opt=(args.inv_order!=0),
                inv_order=args.inv_order,
            )
            end_time = time.time()
            # Record the runtime
            execution_time = end_time - start_time
            peak_memory_usage = torch.cuda.max_memory_allocated()

            img_ws.append(img_w)
            image_latents_ws.append(image_latents_w)
            reversed_latents_ws.append(reversed_latents_w)
            runtimes.append(execution_time)
            peak_memory_usage_GB = peak_memory_usage / (1024**3)
            peak_memories.append(peak_memory_usage_GB)

        ## 3. Evaluation Period
        # Second check w_metric of n*n cases
        for i in range(args.target_num): # i is index of original watermark
            for j in range(args.target_num): # j is index of checking watermark
                no_w_metric, w_metric = eval_watermark(init_latents_ws[i], reversed_latents_ws[i], watermarking_masks[j], gt_patches[j], args)
                w_metrics[i][j].append(w_metric)
                no_w_metrics[i][j].append(no_w_metric)

        if args.with_tracking:
            table.add_data(current_prompt,
                            wandb.Image(orig_image_ws[0]), wandb.Image(orig_image_ws[1]), wandb.Image(orig_image_ws[2]),
                            w_metrics[0][0][-1], w_metrics[0][1][-1], w_metrics[0][2][-1], w_metrics[1][0][-1], w_metrics[1][1][-1], w_metrics[1][2][-1], w_metrics[2][0][-1], w_metrics[2][1][-1], w_metrics[2][2][-1],
                            )
    #
    runtime_avg = sum(runtimes)/len(runtimes)
    peak_memory_avg = sum(peak_memories)/len(peak_memories)
    # Calculate confusion matrix of WM detection
    confusion_matrix = np.zeros((args.target_num, args.target_num), dtype=int)

    for k in range(args.end - args.start):
        # Determine wheter injected ith WM is detected properly
        for i in range(args.target_num):
            temp = []
            for j in range(args.target_num):
                temp.append(w_metrics[i][j][k])
            temp = np.array(temp)
            min_ind = np.argmin(temp)
            confusion_matrix[i][min_ind] += 1

    # Printing confusion matrix on terminal
    print("Each column represents WM used for comparison. Table represents the number of WM in match.")
    for i, row in enumerate(confusion_matrix):
        print(f"Generated with WM{i+1} : | ", end= " ")
        for value in row:
            print(f"{value} |", end=" ")
        print()
    print('done')
    if args.generate_txtfile:
        with open(args.run_name+".txt", "w") as f:
            f.write("Each column represents WM used for comparison. Table represents the number of WM in match.\n")
            for i, row in enumerate(confusion_matrix):
                f.write(f"Generated with WM{i+1} : | ")
                for value in row:
                    f.write(f"{value} | ")
                f.write("\n")
            f.write('done\n')
    if args.with_tracking:
        confusion_matrix_wandb = confusion_matrix.tolist()
        wandb.log({'Table': table, 'confusion_mtx': wandb.Table(data=confusion_matrix_wandb, columns=["detect_1", "detect_2", "detect_3"]), 'runtime':runtime_avg, 'peak_memory': peak_memory_avg})
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--proj_name', default='watermark_detection')
    parser.add_argument('--run_name', default='test run')
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=100, type=int)
    parser.add_argument('--target_num', default=3, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=1000, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='ring')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=6, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # experiment
    parser.add_argument('--num_inference_steps', default=50, type=int, help="steps of sampling")
    parser.add_argument("--solver_order", default=1, type=int, help="order of sampling, 1:DDIM, >=2:DPM-solver++") 
    parser.add_argument('--test_num_inference_steps', default=None, type=int, help="steps of inversion") 
    parser.add_argument("--inv_order", type=int, default=0, help="order of inversion, 0:Naive DDIM iversion, 1:Alg1, 2:Alg2")
    
    parser.add_argument('--mode', choices=['encoder', 'no_grad', 'grad'], help='Choose one of the options')
    parser.add_argument('--dinv_steps', default=100, type=int)
    parser.add_argument("--prompt_unknown", action='store_true', default=False, help="assume during inversion, the original prompt is unknown")
    parser.add_argument("--generate_txtfile", action='store_true', default=False)

    # Default parameter settings
    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    if args.inv_order is None:
        args.inv_order = args.solver_order
    
    main(args)