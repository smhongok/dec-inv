import argparse
import wandb
from tqdm import tqdm
from statistics import mean, stdev
import time

import torch
from torchvision.transforms.functional import to_pil_image, rgb_to_grayscale
import torchvision.transforms as transforms

from src.stable_diffusion.inverse_stable_diffusion import InversableStableDiffusionPipeline 
from diffusers import DPMSolverMultistepScheduler
from src.utils.optim_utils import *
from src.utils.io_utils import *
import torch

def main(args):
    # track with wandb
    table = None
    if args.with_tracking:
        wandb.init(project=args.proj_name, name=args.run_name)
        wandb.config.update(args)
        table = wandb.Table(columns=['image','NMSE_z', 'runtime', 'peak_memory'])
    
    # load stable diffusion pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler(
        beta_end=0.012,
        beta_schedule='scaled_linear',
        beta_start=0.00085,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset=1, 
        trained_betas=None,
        solver_order=args.solver_order,
    )

    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        )
    pipe = pipe.to(device)

    # load dataset
    dataset, prompt_key = get_dataset(args.dataset)

    z0_NMSE = []
    z0_NMSE_dB = []
    runtimes = []
    peak_memories = []

    ind = 0

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]


        text_embeddings_tuple = pipe.encode_prompt(
            current_prompt, 'cuda', 1, args.guidance_scale > 1.0, None)
        text_embeddings = torch.cat([text_embeddings_tuple[1], text_embeddings_tuple[0]])

        
        ### Generation

        # generate init latent
        set_random_seed(seed)
        init_latents = pipe.get_random_latents()

        # generate image
        outputs, z0_gt = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents,
            )
        orig_image = outputs.images[0]

        
        ### Inversion

        # for seeking runtime and peak memory usage
        start_time = time.time()
        torch.cuda.reset_max_memory_allocated()
        
        # image to latent
        img = transform_img(orig_image).unsqueeze(0).to(text_embeddings.dtype).to(device)
        if args.mode == 'encoder':
            image_latents = pipe.get_image_latents(img, sample=False)
            end_time = time.time()
            peak_memory_usage = torch.cuda.max_memory_allocated()
        elif args.mode == 'no_grad':
            if args.precision == 'half':
                image_latents = pipe.decoder_inv_nograd(img, num_steps=args.decoder_inv_numstep, lr=args.lr_no_grad, adam=args.adam)
                end_time = time.time()
                peak_memory_usage = torch.cuda.max_memory_allocated()
            elif args.precision == 'full':
                image_latents = pipe.decoder_inv_nograd(img, num_steps=args.decoder_inv_numstep, lr=args.lr_no_grad,float=True, adam=args.adam)
                end_time = time.time()
                peak_memory_usage = torch.cuda.max_memory_allocated()
            else:
                raise('precision shoule be `half` or `full`')

        elif args.mode=='grad':
            # Note that if we use the half precision for the grad-based method, underflows occur so that the loss them will go to NaN.
            assert args.precision=='full'
            # gradient-based, float
            image_latents = pipe.decoder_inv(img, num_steps=args.decoder_inv_numstep, lr=args.lr_grad, LR_scheduling=args.lr_scheduling)

            end_time = time.time()
            peak_memory_usage = torch.cuda.max_memory_allocated()
            image_latents = image_latents.half()

        else:
            raise("args.mode shoule be `encoder` or `no_grad` or `grad`")
        # Record the runtime
        execution_time = end_time - start_time
        z0_nmse = ((image_latents - z0_gt).norm()**2 / z0_gt.norm()**2).item()
        peak_memory_usage_GB = peak_memory_usage / (1024**3)


        if args.with_tracking:
            table.add_data(wandb.Image(orig_image),z0_nmse, execution_time, peak_memory_usage_GB)
        
        print(ind, z0_nmse, execution_time, peak_memory_usage_GB)
        z0_NMSE.append(z0_nmse)
        z0_NMSE_dB.append(10*np.log10(z0_nmse))
        runtimes.append(execution_time)
        peak_memories.append(peak_memory_usage_GB)            

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'z0_NMSE' : sum(z0_NMSE)/len(z0_NMSE), 'z0_NMSE_dB' : np.mean(z0_NMSE_dB), 'z0_NMSE_dB_stderr' : np.std(z0_NMSE_dB, ddof=1) / np.sqrt(len(z0_NMSE_dB)), 'Runtime': sum(runtimes)/len(runtimes), 'Peak_memory': sum(peak_memories)/len(peak_memories)})
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--proj_name', default='test project')
    parser.add_argument('--run_name', default='test run')
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=100, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--num_images', default=1, type=int, help="num_images per prompt")
    parser.add_argument('--guidance_scale', default=3.0, type=float, help="classifier-free guidance")
    parser.add_argument('--with_tracking', action='store_true', help="track with wandb")
    
    # experiment
    parser.add_argument('--num_inference_steps', default=50, type=int, help="steps of sampling")
    parser.add_argument("--solver_order", default=1, type=int, help="order of sampling, 1:DDIM, >=2:DPM-solver++")
    
    parser.add_argument('--mode', choices=['encoder', 'no_grad', 'grad'], help='Choose one of the options')
    parser.add_argument('--precision', choices=['half', 'full'], help='Choose one of the options')
    parser.add_argument('--decoder_inv_numstep', default=100, type=int)
    parser.add_argument('--lr_no_grad', default=0.01, type=float, help="lr for no_grad")
    parser.add_argument('--lr_grad', default=0.01, type=float, help="lr for grad")

    parser.add_argument('--adam', action='store_true', help="adam")
    parser.add_argument('--lr_scheduling', action='store_true', help='LR scheduling for grad-based method')

    args = parser.parse_args()


    main(args)