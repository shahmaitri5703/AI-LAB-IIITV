import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

def load_octave_text_matrix(filename):
    
    print(f"Loading Octave text file from: {filename}")
    with open(filename, 'r') as f:
        all_lines = f.readlines()
    data_start_index = 0
    for i, line in enumerate(all_lines):
        if line.strip().startswith('# ndims:'):
            data_start_index = i + 2
            break
    if data_start_index == 0:
        raise ValueError("Could not find '# ndims:' header in file.")
    data_lines_only = [line.strip() for line in all_lines[data_start_index:]]
    image_flat = np.fromstring(" ".join(data_lines_only), dtype=np.uint8, sep=' ')
    image_dim = 512
    if image_flat.size != image_dim * image_dim:
        raise ValueError(f"Data size mismatch! Expected {image_dim*image_dim}, found {image_flat.size}.")
    return image_flat.reshape((image_dim, image_dim))

class JigsawSolver:
    
    def __init__(self, scrambled_image, grid=4):
        self.scrambled_image = scrambled_image
        self.grid = grid
        self.blocks, self.block_size = self._split_into_blocks()
        self.num_blocks = len(self.blocks)
        self.perm = list(range(self.num_blocks))

    def _split_into_blocks(self):
       
        h, w = self.scrambled_image.shape
        bh, bw = h // self.grid, w // self.grid
        blocks = []
        for i in range(self.grid):
            for j in range(self.grid):
                blocks.append(self.scrambled_image[i*bh:(i+1)*bh, j*bw:(j+1)*bw])
        return blocks, (bh, bw)

    def _calculate_energy(self, perm):
       
        total = 0.0
        grid_blocks = [[self.blocks[perm[i*self.grid + j]] for j in range(self.grid)] for i in range(self.grid)]
        
        
        for i in range(self.grid):
            for j in range(self.grid - 1):
                a, b = grid_blocks[i][j], grid_blocks[i][j+1]
                total += np.sum(np.abs(a[:, -1].astype(np.int32) - b[:, 0].astype(np.int32)))
        
       
        for i in range(self.grid - 1):
            for j in range(self.grid):
                a, b = grid_blocks[i][j], grid_blocks[i+1][j]
                total += np.sum(np.abs(a[-1, :].astype(np.int32) - b[0, :].astype(np.int32)))
                
        return float(total)

    def solve(self, initial_temp, cooling_rate,
              iter_per_temp, max_steps, seed,
              visualize_every):
        
        if seed is not None:
            random.seed(seed)
        
        best_perm = self.perm[:]
        current_energy = self._calculate_energy(self.perm)
        best_energy = current_energy
        
        print(f"Initial adjacency energy: {current_energy:.0f}")

        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        temp = initial_temp
        step = 0

        while step < max_steps and temp > 0.1:
            for _ in range(iter_per_temp):
                if step >= max_steps: break
                
                a, b = random.sample(range(self.num_blocks), 2)
                self.perm[a], self.perm[b] = self.perm[b], self.perm[a]
                
                prop_energy = self._calculate_energy(self.perm)
                dE = prop_energy - current_energy
                
                if dE < 0 or (temp > 1e-9 and random.random() < math.exp(-dE / temp)):
                    current_energy = prop_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_perm = self.perm[:]
                else:
                    self.perm[a], self.perm[b] = self.perm[b], self.perm[a]
                
                step += 1

                if step % visualize_every == 0:
                    self._update_plot(ax, fig, best_perm, step, temp, best_energy)
            
            temp *= cooling_rate

        plt.ioff()
        print(f"\nFinished. Final best energy: {best_energy:.0f}")
        return best_perm, best_energy

    def _update_plot(self, ax, fig, perm, step, temp, energy):

        img = self.stitch_image(perm)
        ax.clear()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"SA step {step} | temp {temp:.2f}\nbest energy {energy:.0f}")
        ax.axis('off')
        fig.canvas.draw()
        plt.pause(0.001)

    def stitch_image(self, perm):
       
        bh, bw = self.block_size
        img = np.zeros((self.grid*bh, self.grid*bw), dtype=self.blocks[0].dtype)
        idx = 0
        for i in range(self.grid):
            for j in range(self.grid):
                img[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = self.blocks[perm[idx]]
                idx += 1
        return img

if __name__ == "__main__":
    try:
        scrambled_image = load_octave_text_matrix('scrambled_lena.mat').T

        if scrambled_image is not None:
            solver = JigsawSolver(scrambled_image, grid=4)

          
            best_perm, best_energy = solver.solve(
                initial_temp=5500.0,       
                cooling_rate=0.999,        
                iter_per_temp=250,         
                max_steps=50000,           
                seed=99,                  
                visualize_every=1000
            )

            final_img = solver.stitch_image(best_perm)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(scrambled_image, cmap='gray')
            plt.title('Scrambled Input')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(final_img, cmap='gray')
            plt.title(f'Unscrambled simulated annealing" - Energy {best_energy:.0f}')
            plt.axis('off')
            plt.show()

    except Exception as e:
        print(f"An error occurred in the main block: {e}")