import os
import numpy as np
import torch
import matplotlib.pyplot as plt

class PlotCallback:
    def __init__(self, num_samples=3, save_dir=None):
        self.num_samples = num_samples
        self.save_dir = save_dir  
        self.counter = 0
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __cal__(self, trainer):
        trainer.model.eval()
        with torch.no_grad():
            input = self._batch_random_sample(trainer)
            output,_,_z = trainer.model(input)

            input_img = self._reshape_to_image(inputs, numpy=True)
            output_img = self._reshape_to_image(output, numpy=True)
            z_ = self._to_numpy(z)

            self._plot_images(input_img, output_img, z_)
        trainer.model.train()

    def _batch_random_sample(self, trainer):
        """Sample a batch of random inputs from the training dataset."""
        dataset = trainer.dataloader.dataset                
        ids = np.random.randint(len(dataset), size=self.num_samples)
        samples = [dataset[i][0] for i in ids]

        batch = torch.stack(samples)
        batch = batch.view(batch.size(0), -1).to(trainer.device)
        return batch

    def _reshape_to_image(self, tensor, numpy=False):
        """Reshape the tensor to image format."""
        if numpy:
            return tensor.cpu().numpy().reshape(-1, 28, 28)
        return tensor.view(-1, 1, 28, 28)
    
    def _to_numpy(self, tensor):
        """Convert a tensor to a numpy array."""
        return tensor.cpu().numpy()
    def _plot_images(self, input_img, output_img, z):
        """Plot input and output images side by side."""
        fig, axes = plt.subplots(self.num_samples, 2, figsize=(10, 5 * self.num_samples))
        for i in range(self.num_samples):
            # Images
            axes[i][0].imshow(input_images[i], cmap="gray")
            axes[i][0].set_axis_off()

            # Variable z
            axes[i][1].bar(np.arange(len(z[i])), z[i])

            # Reconstructed images
            axes[i][2].imshow(recon_images[i], cmap="gray")
            axes[i][2].set_axis_off()


        plt.tight_layout()
        if self.save_dir:
            fig.savefig(self.save_dir + "/results_{}.png".format(self.counter))
            plt.close(fig)
            self.counter += 1
        else:
            plt.show()

