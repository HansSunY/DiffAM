import time
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
import lpips

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS
from utils.align_utils import run_alignment


class DiffAM_MR(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.src_txt = self.args.src_txts  # "face with makeup"
        self.trg_txt = self.args.trg_txts  # "face without makeup"

    def clip_finetune(self):
        print(self.args.exp)
        print(f'   {self.src_txt}')
        print(f'-> {self.trg_txt}')

        # ----------- Model -----------#
        url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"

        model = i_DDPM()
        if self.args.model_path:
            init_ckpt = torch.load(self.args.model_path)
        else:
            init_ckpt = torch.hub.load_state_dict_from_url(
                url, map_location=self.device)
        learn_sigma = True
        print("Original diffusion Model loaded.")
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(
            model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(
            optim_ft, step_size=1, gamma=self.args.sch_gamma)
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Loss -----------#
        print("Loading losses")
        clip_loss_func = CLIPLoss(
            self.device,
            lambda_makeup_direction=0,
            lambda_direction=1,
            clip_model=self.args.clip_model_name)
        id_loss_func = id_loss.IDLoss().to(self.device).eval()
        loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}
        for mode in ['train', 'test']:
            img_lat_pairs = []
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, x_lat, mask) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image(
                        (x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    mask_image = Image.fromarray(
                        mask.detach().clone().squeeze().cpu().numpy())
                    mask_image.convert('RGB').save(os.path.join(
                        self.args.image_folder, f'{mode}_{step}_0_mask.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(
                    self.config.data.dataset, DATASET_PATHS, self.config)
                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, (img, mask) in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image(
                    (x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                mask_image = Image.fromarray(
                    mask.detach().clone().squeeze().cpu().numpy())
                mask_image.convert('RGB').save(os.path.join(
                    self.args.image_folder, f'{mode}_{step}_0_mask.png'))
                x = x0.clone()
                model.eval()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma)
                            progress_bar.update(1)

                    img_lat_pairs.append(
                        [x0, x.detach().clone(), x_lat.detach().clone(), mask])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(
            f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(
                0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        print(f"CHANGE {self.src_txt} TO {self.trg_txt}")
        model.module.load_state_dict(init_ckpt)
        optim_ft.load_state_dict(init_opt_ckpt)
        scheduler_ft.load_state_dict(init_sch_ckpt)
        clip_loss_func.target_direction = None

        # ----------- Train -----------#
        for it_out in range(self.args.n_iter):
            exp_id = os.path.split(self.args.exp)[-1]
            save_name = f'checkpoint/{exp_id}_{self.trg_txt.replace(" ", "_")}-{it_out}.pth'
            if self.args.do_train:
                if os.path.exists(save_name):
                    print(f'{save_name} already exists.')
                    model.module.load_state_dict(torch.load(save_name))
                    continue
                else:
                    for step, (x0, x_id, x_lat, _) in enumerate(img_lat_pairs_dic['train']):
                        model.train()
                        time_in_start = time.time()

                        optim_ft.zero_grad()
                        x = x_lat.clone()

                        with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                            for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                                t = (torch.ones(n) * i).to(self.device)
                                t_next = (torch.ones(n) * j).to(self.device)

                                x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                   logvars=self.logvar,
                                                   sampling_type=self.args.sample_type,
                                                   b=self.betas,
                                                   eta=self.args.eta,
                                                   learn_sigma=learn_sigma)

                                progress_bar.update(1)
                        tvu.save_image(
                            (x0+1)/2, './sample_real/sample_{}.png'.format(step))
                        tvu.save_image(
                            (x+1)/2, './sample_fake/sample_{}.png'.format(step))

                        loss_clip = (2 - clip_loss_func(x0, None,
                                     x, None, self.src_txt, self.trg_txt)) / 2
                        loss_clip = -torch.log(loss_clip)
                        loss_id = torch.mean(id_loss_func(x0, x))
                        loss_l1 = nn.L1Loss()(x0, x)
                        loss_lpips = loss_fn_alex(x0, x)
                        loss = self.args.MR_clip_loss_w * loss_clip + self.args.MR_id_loss_w * loss_id + \
                            self.args.MR_l1_loss_w * loss_l1 + self.args.MR_lpips_loss_w * loss_lpips
                        loss.backward()

                        optim_ft.step()
                        print(
                            f"CLIP {step}-{it_out}: loss_id: {loss_id:.3f}, loss_clip: {loss_clip:.3f}")

                        if self.args.save_train_image:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'train_{step}_2_clip_{self.trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png'))
                        time_in_end = time.time()
                        print(
                            f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")
                        if step == self.args.n_train_img - 1:
                            break

                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.state_dict(), save_name)
                    else:
                        torch.save(model.state_dict(), save_name)
                    print(f'Model {save_name} is saved.')
                    scheduler_ft.step()

            # ----------- Eval -----------#
            if self.args.do_test:
                if not self.args.do_train:
                    print(save_name)
                    model.module.load_state_dict(torch.load(save_name))

                model.eval()
                img_lat_pairs = img_lat_pairs_dic[mode]
                for step, (x0, x_id, x_lat, _) in enumerate(img_lat_pairs):
                    with torch.no_grad():
                        x = x_lat
                        with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                            for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                                t = (torch.ones(n) * i).to(self.device)
                                t_next = (torch.ones(n) * j).to(self.device)

                                x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                   logvars=self.logvar,
                                                   sampling_type=self.args.sample_type,
                                                   b=self.betas,
                                                   eta=self.args.eta,
                                                   learn_sigma=learn_sigma)

                                progress_bar.update(1)
                        tvu.save_image(
                            (x0+1)/2, './sample_real_test/sample_{}.png'.format(step))
                        tvu.save_image(
                            (x+1)/2, './sample_fake_test/sample_{}.png'.format(step))
                        print(f"Eval {step}-{it_out}")
                        tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_2_clip_{self.trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_test_step}.png'))
                        if step == self.args.n_test_img - 1:
                            break

    def edit_one_image(self):
        # ----------- Data -----------#
        n = self.args.bs_test
        img = Image.open(self.args.img_path).convert("RGB")
        img = img.resize((self.config.data.image_size,
                         self.config.data.image_size), Image.ANTIALIAS)
        img = np.array(img)/255
        img = torch.from_numpy(img).type(torch.FloatTensor).permute(
            2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(
            self.args.image_folder, f'0_orig.png'))
        x0 = (img - 0.5) * 2.

        models = []
        model_paths = [None, self.args.model_path]

        for model_path in model_paths:
            model_i = i_DDPM()
            if model_path:
                ckpt = torch.load(model_path)
            else:
                ckpt = torch.load("pretrained/makeup.pt")
            learn_sigma = True
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        with torch.no_grad():
            # ---------------- Invert Image to Latent in case of Deterministic Inversion process -------------------#
            if self.args.deterministic_inv:
                x_lat_path = os.path.join(
                    self.args.image_folder, f'x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
                if not os.path.exists(x_lat_path):
                    seq_inv = np.linspace(
                        0, 1, self.args.n_inv_step) * self.args.t_0
                    seq_inv = [int(s) for s in list(seq_inv)]
                    seq_inv_next = [-1] + list(seq_inv[:-1])

                    x = x0.clone()
                    with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=models[0],
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma,
                                               ratio=0,
                                               )

                            progress_bar.update(1)
                        x_lat = x.clone()
                        torch.save(x_lat, x_lat_path)
                else:
                    print('Latent exists.')
                    x_lat = torch.load(x_lat_path)

            # ----------- Generative Process -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                  f" Steps: {self.args.n_test_step}/{self.args.t_0}")
            if self.args.n_test_step != 0:
                seq_test = np.linspace(
                    0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print('Uniform skip type')
            else:
                seq_test = list(range(self.args.t_0))
                print('No skip')
            seq_test_next = [-1] + list(seq_test[:-1])

            for it in range(self.args.n_iter):
                if self.args.deterministic_inv:
                    x = x_lat.clone()
                else:
                    e = torch.randn_like(x0)
                    a = (1 - self.betas).cumprod(dim=0)
                    x = x0 * a[self.args.t_0 - 1].sqrt() + e * \
                        (1.0 - a[self.args.t_0 - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'1_lat_ninv{self.args.n_inv_step}.png'))

                with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)

                        x = denoising_step(x, t=t, t_next=t_next, models=models,
                                           logvars=self.logvar,
                                           sampling_type=self.args.sample_type,
                                           b=self.betas,
                                           eta=self.args.eta,
                                           learn_sigma=learn_sigma,
                                           ratio=self.args.model_ratio)

                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'2_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_{i}_it{it}.png'))
                        progress_bar.update(1)

                x0 = x.clone()
                if self.args.model_path:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                               f"3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}_{self.args.model_path.split('/')[-1].replace('.pth','')}.png"))
                else:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                               f'3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png'))
