"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        """
        DDIM采样的主要实现函数
        参数:
            cond: 条件信息,用于条件生成
            shape: 生成图像的形状 (batch_size, channels, height, width)
            x_T: 初始噪声,如果为None则随机生成
            ddim_use_original_steps: 是否使用原始DDPM的时间步长,默认False使用DDIM的时间步长
            callback: 每一步采样后的回调函数
            timesteps: 采样步数,如果为None则使用预设步数
            quantize_denoised: 是否对去噪后的图像进行量化
            mask: 用于inpainting的mask
            x0: 原始图像,用于inpainting
            img_callback: 图像回调函数
            log_every_t: 每隔多少步记录一次中间结果
            temperature: 采样温度,控制随机性
            noise_dropout: 噪声dropout率
            score_corrector: 分数修正器
            corrector_kwargs: 分数修正器的参数
            unconditional_guidance_scale: 无条件引导比例
            unconditional_conditioning: 无条件引导的条件信息
        返回:
            img: 生成的最终图像
            intermediates: 包含中间步骤的字典
        """
        device = self.model.betas.device
        b = shape[0]
        # 如果没有提供初始噪声,则随机生成
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        # 确定采样的时间步长
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        # 用于存储中间结果的字典
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        # 确定时间步长的遍历顺序(从噪声到图像)
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        # 主采样循环
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # 处理inpainting的情况
            if mask is not None:
                # 如果存在mask(用于inpainting),需要特殊处理
                # mask是一个二值掩码,值为1的部分保留原图,值为0的部分需要重建
                # 这里将原始图像和生成的噪声图像按照mask进行混合
                # 确保在有mask的情况下必须提供原始图像x0
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # 对原始图像添加噪声
                img = img_orig * mask + (1. - mask) * img  # 将噪声图像与原始图像按mask混合

            # 执行DDIM采样步骤
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            # 记录中间结果
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        # 获取输入张量的形状和设备信息
        b, *_, device = *x.shape, x.device

        # 根据是否使用无条件引导来计算噪声预测
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            # 直接使用条件模型预测噪声
            e_t = self.model.apply_model(x, t, c)
        else:
            # 将输入和时间步复制一份用于无条件和有条件预测
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            # 处理条件输入,支持字典和张量两种形式
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [
                            torch.cat([unconditional_conditioning[k][i], c[k][i]])
                            for i in range(len(c[k]))
                        ]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            # 分别获取无条件和有条件的预测结果
            # 将模型应用于输入,得到预测结果
            # x_in: 包含无条件和有条件输入的拼接张量
            # t_in: 对应的时间步
            # c_in: 对应的条件信息
            # chunk(2): 将预测结果分成两半,分别对应无条件和有条件的预测
            # e_t_uncond: 无条件情况下预测的噪声
            # e_t: 有条件情况下预测的噪声
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            # 使用CFG(Classifier-Free Guidance)进行引导
            # 实现分类器自由引导(CFG)
            # e_t_uncond: 无条件预测的噪声
            # e_t: 有条件预测的噪声
            # unconditional_guidance_scale: CFG缩放因子,控制条件引导的强度
            # 这个参数实际上是在控制"相对于无条件预测的偏离程度"，因此被命名为"unconditional_guidance_scale"。
            # 当缩放因子为1时,等同于普通的条件生成
            # 当缩放因子>1时,会加强条件的影响
            # 计算公式: e_t = e_t_uncond + scale * (e_t - e_t_uncond)
            # 其中(e_t - e_t_uncond)表示条件带来的变化量
            # 整个公式可以理解为："无条件预测 + w倍的条件引导影响"
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        # 如果提供了分数修正器,则修正预测的噪声
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # 根据是否使用原始步长获取相应的参数
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # 为当前时间步选择对应的参数值
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # 预测x_0(原始图像)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # 如果需要量化去噪结果
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # 计算指向x_t的方向
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        # 生成随机噪声并应用温度系数
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        # 应用噪声dropout(如果指定)
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # 计算前一时间步的图像
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
