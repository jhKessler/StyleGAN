import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from clients import CheckpointClient, ImageClient, TrainingClient
import utils

def train_model(Args):
	"""
	training for model
	"""
	
	torch.manual_seed(Args.SEED)
	Args.DEVICE = torch.device(Args.DEVICE) if isinstance(Args.DEVICE, str) else Args.DEVICE
	
	checkpoint = CheckpointClient.get_checkpoint(Args)
	model_dict = checkpoint.get_dict()


	generator = model_dict["generator"]
	g_optim = model_dict["g_optimizer"]
	discriminator = model_dict["discriminator"]
	d_optim = model_dict["d_optimizer"]
	preview_noise = model_dict["preview_noise"]

	iteration = model_dict["iteration"]
	samples = model_dict["samples"]
	step = model_dict["step"]
	alpha = model_dict["alpha"]

	gen_parameter_count = utils.count_parameters(generator)
	disc_parameter_count = utils.count_parameters(discriminator)

	print(f"Starting Training for Model {Args.MODEL_ID} with resolution {utils.get_resolution(step)}x{utils.get_resolution(step)} and {samples} used.")
	print(f"Generator parameters: {utils.format_large_nums(gen_parameter_count)}")
	print(f"Discriminator parameters: {utils.format_large_nums(disc_parameter_count)}")

	
	for step in range(step, Args.MAX_STEPS):
		img_size = utils.get_resolution(step)
		data = TrainingClient.create_dataloader(Args.BATCH_SIZE, img_size)
		loader = iter(data)
		
		pbar = tqdm(total = Args.PHASE_SIZE)
		pbar.update(samples)
		pbar.set_description(f"Resolution: {img_size}x{img_size} | Samples: {samples} | Alpha: {alpha}")
		
		g_loss_list = []
		d_loss_list = []
		d_real_output_list = []
		d_fake_output_list = []
		for iteration in range(iteration, Args.PHASE_SIZE):
			try:
				image_batch = next(loader)[0]
			except StopIteration:
				loader = iter(data)
				image_batch = next(loader)[0]

			image_batch = image_batch.float().to(Args.DEVICE)

			alpha = max([1, round(samples / Args.FADE_SIZE, 2)]) if step != 0 else 1

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

			disc_loss = F.softplus(-real_prediction + fake_prediction) + (grad_penalty * 10)
			disc_loss = F.softplus(disc_loss)
			d_loss_list.append(disc_loss.item())
			discriminator.zero_grad()
			disc_loss.backward(retain_graph = True)
			d_optim.step()
						
			gen_predict = discriminator(generated_images).mean()
			gen_loss = -gen_predict
			g_loss_list.append(gen_loss.item())
			generator.zero_grad()
			gen_loss.backward()
			g_optim.step()

			samples += current_batch_size
			pbar.update(current_batch_size)
			pbar.set_description(f"Resolution: {img_size}x{img_size} | Samples: {samples} | Alpha: {alpha}")
			
			if samples % 5_000 < current_batch_size:
				avg_real_output = np.array(d_real_output_list).mean().round(2)
				d_real_output_list = []

				avg_fake_output = np.array(d_fake_output_list).mean().round(2)
				d_fake_output_list = []

				avg_d_loss = np.array(d_loss_list).mean().round(2)
				d_loss_list = []

				avg_gen_loss = np.array(g_loss_list).mean().round(2)
				g_loss_list = []

				print(f"{avg_real_output=}, {avg_fake_output=}, {avg_d_loss=}, {avg_gen_loss=}")
				generator_noise = ImageClient.make_image_noise(Args.NUM_PROGRESS_IMGS, Args.NOISE_DIM, Args.DEVICE)
				generated_images = generator(generator_noise, Args.DEVICE)
				ImageClient.save_progress_images(generated_images, Args.MODEL_ID, samples)
	
	
	




	
