import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from clients import CheckpointClient, ImageClient, TrainingClient
from step_assert import StepAssert
import utils
import gc

def train_model(Args):
	"""
	training for model
	"""
	
	torch.manual_seed(Args.SEED)
	Args.DEVICE = torch.device(Args.DEVICE) if isinstance(Args.DEVICE, str) else Args.DEVICE
	
	checkpoint = CheckpointClient.get_checkpoint(Args)
	model_dict = checkpoint.get_dict()

	Args = model_dict["Args"]

	generator = model_dict["generator"]
	g_optim = model_dict["g_optimizer"]
	discriminator = model_dict["discriminator"]
	d_optim = model_dict["d_optimizer"]
	preview_noise = model_dict["preview_noise"]

	samples = model_dict["samples"]
	step = model_dict["step"]
	alpha = model_dict["alpha"]

	step_assert = StepAssert(tolerance=3)
	gen_parameter_count = utils.count_parameters(generator)
	disc_parameter_count = utils.count_parameters(discriminator)

	print(f"Starting Training for Model {Args.MODEL_ID} with resolution {utils.get_resolution(step)}x{utils.get_resolution(step)} and {samples} samples used.")
	print(f"Generator parameters: {utils.format_large_nums(gen_parameter_count)}")
	print(f"Discriminator parameters: {utils.format_large_nums(disc_parameter_count)}")

	last_FID_score = 0

	for step in range(step, Args.MAX_STEPS):

		best_FID_score = None

		if step != 0:
			print("Loading Best Checkpoint...")
			checkpoint = CheckpointClient.get_checkpoint(Args)
			model_dict = checkpoint.get_dict()
			generator = model_dict["generator"]
			g_optim = model_dict["g_optimizer"]
			discriminator = model_dict["discriminator"]
			d_optim = model_dict["d_optimizer"]
			preview_noise = model_dict["preview_noise"]
			samples = model_dict["samples"]

		utils.adjust_lr(d_optim, Args.LR[step])
		utils.adjust_lr(g_optim, Args.LR[step])

		img_size = utils.get_resolution(step)
		data = TrainingClient.create_dataloader(Args.BATCH_SIZE, img_size)
		loader = iter(data)
		alpha_samples = 0

		pbar = tqdm()
		pbar.update(samples)
		pbar.set_description(f"Resolution: {img_size}x{img_size} | Samples: {samples} | Alpha: {alpha} | FID: {last_FID_score}")
		
		g_loss_list = []
		d_loss_list = []
		d_real_output_list = []
		d_fake_output_list = []

		real_fid_images = []
		fid_loader = iter(TrainingClient.create_dataloader(100, img_size))
		for i in range(10):
			fid_images = TrainingClient.merge_images(next(fid_loader)[0], alpha = 1)
			real_fid_images.append(fid_images)
		ImageClient.save_real_fid_images(real_fid_images)
		

		for iteration in range(1_000_000):
			try:
				image_batch = next(loader)[0]
			except StopIteration:
				loader = iter(data)
				image_batch = next(loader)[0]

			image_batch = image_batch.float().to(Args.DEVICE)
			alpha = min([1, round(alpha_samples / Args.FADE_SIZE, 2)]) if step != 0 else 1

			image_batch = TrainingClient.merge_images(image_batch, alpha)

			utils.toggle_grad(discriminator, True)
			utils.toggle_grad(generator, True)

			current_batch_size = image_batch.shape[0]
			image_batch.requires_grad = True

			real_prediction = discriminator(image_batch.float().to(Args.DEVICE)).mean()
			d_real_output_list.append(real_prediction.item())

			generator_noise = ImageClient.make_image_noise(current_batch_size, Args.NOISE_DIM, Args.DEVICE)
			generated_images = generator(generator_noise, Args.DEVICE)
			fake_prediction = discriminator(generated_images).mean()
			d_fake_output_list.append(fake_prediction.item())

			grad_penalty = TrainingClient.gradient_penalty(
				discriminator, 
				image_batch, 
				generated_images,
				Args.DEVICE
			)
			
			disc_loss = F.softplus(-(real_prediction.mean() - fake_prediction.mean()) + (grad_penalty * 10))
			d_loss_list.append(disc_loss.item())
			discriminator.zero_grad()
			disc_loss.backward(retain_graph = True)
			d_optim.step()

			utils.toggle_grad(discriminator, False)
			utils.toggle_grad(generator, True)	
			gen_predict = discriminator(generated_images).mean()
			gen_loss = -gen_predict
			g_loss_list.append(gen_loss.item())
			generator.zero_grad()
			gen_loss.backward()
			g_optim.step()

			samples += current_batch_size
			alpha_samples += current_batch_size
			pbar.update(current_batch_size)
			pbar.set_description(f"Resolution: {img_size}x{img_size} | Samples: {utils.format_large_nums(samples)} | Alpha: {alpha} | FID: {best_FID_score} | tolerance: {step_assert.get_tolerance()}, Learning Rate: {Args.LR[step]}")
			
			if samples % 10_000 < current_batch_size:
				
				with torch.no_grad():
					generated_images = generator(preview_noise, Args.DEVICE)
				ImageClient.save_progress_images(generated_images, Args.MODEL_ID, samples)
				del generated_images
				torch.cuda.empty_cache()
				# calculate fid
				fake_fid_images = []

				with torch.no_grad():
					for i in range(10):
						noise = ImageClient.make_image_noise(100, Args.NOISE_DIM, Args.DEVICE)
						fake_fid_images.append(generator(noise, Args.DEVICE).cpu())
				ImageClient.save_fake_fid_images(fake_fid_images)
				del fake_fid_images
				gc.collect()
				new_FID_score = TrainingClient.calculate_FID_score()
				
				if best_FID_score is None or new_FID_score < best_FID_score:
					print("Saving best Model...")
					# save checkpoint
					model_dict["generator"] = generator
					model_dict["g_optimizer"] = g_optim
					model_dict["discriminator"] = discriminator
					model_dict["d_optimizer"] = d_optim

					model_dict["samples"] = samples
					model_dict["step"] = step

					checkpoint.save(new_dict = model_dict)

					best_FID_score = new_FID_score

				avg_real_output = np.array(d_real_output_list).mean().round(2)
				d_real_output_list = []

				avg_fake_output = np.array(d_fake_output_list).mean().round(2)
				d_fake_output_list = []

				avg_d_loss = np.array(d_loss_list).mean().round(2)
				d_loss_list = []

				avg_gen_loss = np.array(g_loss_list).mean().round(2)
				g_loss_list = []

				print(f" {avg_real_output=}, {avg_fake_output=}, {avg_d_loss=}, {avg_gen_loss=}")

				gc.collect()
				torch.cuda.empty_cache()

				if alpha == 1 and (alpha_samples >= Args.PHASE_SIZE or not step_assert(new_FID_score)):
					print("Going to next Step...")
					step_assert.reset()					
					
