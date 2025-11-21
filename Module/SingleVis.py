import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
class Vis():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['xtick.labelsize'] = 10    # Axis label size
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 17     # Title font size
    plt.rcParams['axes.labelsize'] = 16     # Axis name size
    plt.rcParams['axes.linewidth'] = 1      # Axis thickness

    def __init__(self, ques_name, ini_num, file_desti, module_name, input =[], u = [], u_true = None, mode: str = 'teacher'):
        input_num = input.T.shape[0]
        if input_num == 2:
            self.x, self.y = input[:,0], input[:,1]
        elif input_num == 3:
            self.x, self.y, self.z = input[:,0], input[:,1], input[:,2]
        self.u = u
        self.u_true = u_true  # Store ground truth
        self.ques_name = ques_name
        self.ini_num = ini_num
        self.file_densti =  file_desti 
        self.module_name = module_name
        if mode == 'student':
            self.module_name += '_student'

    def loss_vis(self):
        self.loss_desti = self.file_densti + '/Loss/'
        df = pd.read_csv(f'{self.loss_desti}{self.ques_name}_{self.ini_num}_loss_{self.module_name}.csv').values
        header = pd.read_csv(f'{self.loss_desti}{self.ques_name}_{self.ini_num}_loss_{self.module_name}.csv', nrows=0).columns

        # Since sometimes the iter in front is a continuation, we need to create a new iter list here
        iter = np.arange(0, len(df[:,0]), 1)

        for j in range (len(header)-1):
            # Skip if the value is zero
            if df[0,j+1] == 0:
                continue
            plt.figure(figsize=(3.85, 3.5)) # Adjust image size
            plt.plot(iter , df[:,j+1])
            plt.yscale('log')
            ax = plt.gca()
            ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')    # Use scientific notation for x-axis
            plt.grid()
            plt.xlabel(header[0])
            plt.ylabel(header[j+1])
            plt.title(self.ques_name + ' ' + header[j+1] + ' ' +self.module_name, pad=10)
            plt.savefig(self.loss_desti + self.ques_name +'_'+str(self.ini_num)+'_' + header[j+1] + '_' + self.module_name + '.png', bbox_inches='tight')
            plt.close()
    
    def figure_2d(self):
        self.figure_desti = self.file_densti + '/Figure/'
        if not os.path.exists(self.figure_desti):
            os.mkdir(self.figure_desti)
        
        for i in range(len(self.u.T)):
            print(f"Drawing {self.ques_name} {self.module_name} figure {i+1}...")

            if 'Flow' in self.ques_name:
                fig, ax = plt.subplots(figsize=(4.4, 2)) # Flow case specific length
                x = self.x.reshape([-1,])
                y = (self.y + 0.2).reshape([-1,])  # in order to align the coordinate axis, we need to add 0.2 from the y values

                # Cylinder parameters
                center_x, center_y, radius = 0.2, 0.2, 0.05

                # Triangulation
                triang = tri.Triangulation(x, y)

                # Mask triangles inside cylinder area
                mask = []
                for tri_idx in triang.triangles:
                    # Calculate triangle centroid
                    xc = x[tri_idx].mean()
                    yc = y[tri_idx].mean()
                    # Check if centroid is inside cylinder
                    if ((xc - center_x)**2 + (yc - center_y)**2) < radius**2:
                        mask.append(True)
                    else:
                        mask.append(False)
                triang.set_mask(mask)

                cf = plt.tripcolor(triang, self.u[:,i], cmap='rainbow', vmin=0 if i < 2 else -0.6, vmax=4 if i == 0 else 1.3 if i == 1 else 0.6)

            else: 
                fig, ax = plt.subplots(figsize=(3.85, 3.5))        # This is the size of the image
                cf = plt.scatter(self.x, self.y, c=self.u[:,i], alpha=1 - 0.1, edgecolors='none', cmap='rainbow',marker='s', s=int(8))  # s is the size of the points
            plt.xlabel('x', style='italic')
            plt.ylabel('y', style='italic')
            plt.margins(0) # Set axis to be compact
            plt.title (self.ques_name + ' ' + self.module_name, pad=10)
            fig.colorbar(cf, fraction=0.046, pad=0.04)
            plt.savefig(f"{self.figure_desti}{self.ques_name}_figure_{i+1}_{self.module_name}.png", bbox_inches='tight')
            plt.close()

    def figure_3d(self):
        self.figure_desti = self.file_densti + '/Figure/'
        if not os.path.exists(self.figure_desti):
            os.mkdir(self.figure_desti)
        
        for i in range(len(self.u.T)):
            print(f"Drawing {self.ques_name} {self.module_name} figure {i+1}...")
            fig = plt.figure(figsize=(4.4, 4))  
            cf = fig.add_subplot(111, projection='3d')
            scatter = cf.scatter(self.x, self.y, self.z, c= self.u[:,i], cmap='rainbow', edgecolors='none', vmin=self.u[:,i].min(), vmax=self.u[:,i].max())
            cf.set_xlabel('x', style='italic')
            cf.set_ylabel('y', style='italic')
            cf.set_zlabel('z', style='italic')
            cf.view_init(elev=20, azim=160)
            colorbar = plt.colorbar(scatter, fraction=0.04, pad=0.2)  # Add color bar in 3d axis
            colorbar.set_label('T')  # Set color bar label
            plt.savefig(f"{self.figure_desti}{self.ques_name}_figure_{i+1}_{self.module_name}.png", bbox_inches='tight')
            plt.close()

    def para_vis(self):
        self.para_desti = self.file_densti + '/Parameters/'
        df = pd.read_csv(self.para_desti + self.ques_name + '_' + str(self.ini_num) + '_paras_' + self.module_name + '.csv').values
        header = pd.read_csv(self.para_desti + self.ques_name + '_' + str(self.ini_num) + '_paras_' + self.module_name + '.csv',  nrows=0).columns

        iter = np.arange(0, len(df[:,0]), 1)

        for j in range (len(header)-1):
            plt.plot(iter , df[:,j+1])
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
            plt.xlabel(header[0], fontdict=font)
            plt.ylabel(header[j+1], fontdict=font)
            plt.title(self.ques_name + ' ' + header[j+1] + ' ' +self.module_name)
            plt.savefig(self.para_desti + self.ques_name +'_'+str(self.ini_num)+'_' + header[j+1] + '_' +self.module_name + '.png', bbox_inches='tight')
            plt.close()

    def comprehensive_results_figure(self):
        """
        Create comprehensive visualization with:
        - Column 1: MSE propagation (loss curve)
        - Column 2: Ground truth
        - Column 3: PINN prediction
        - Column 4: Absolute error
        """
        self.figure_desti = self.file_densti + '/Figure/'
        if not os.path.exists(self.figure_desti):
            os.mkdir(self.figure_desti)

        # Read loss data
        loss_desti = self.file_densti + '/Loss/'
        df_loss = pd.read_csv(f'{loss_desti}{self.ques_name}_{self.ini_num}_loss_{self.module_name}.csv')

        # Calculate MSE between prediction and ground truth
        if self.u_true is not None:
            mse_values = np.mean((self.u - self.u_true)**2, axis=0)
            absolute_error = np.abs(self.u - self.u_true)

        # Reshape for 2D plotting
        grid_size = int(np.sqrt(len(self.x)))
        x_grid = self.x.reshape(grid_size, grid_size)
        y_grid = self.y.reshape(grid_size, grid_size)

        for i in range(self.u.shape[1]):
            print(f"Creating comprehensive figure for {self.ques_name} {self.module_name} output {i+1}...")

            fig = plt.figure(figsize=(16, 8))

            # 1. MSE Propagation (Loss curve)
            ax1 = plt.subplot(2, 4, 1)
            if 'loss' in df_loss.columns:
                ax1.plot(df_loss['iter'], df_loss['loss'], 'b-', linewidth=2)
                ax1.set_yscale('log')
                ax1.grid(True, alpha=0.3)
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('MSE (Loss)')
                ax1.set_title('MSE Propagation')

            # Also plot individual loss components in second row
            ax5 = plt.subplot(2, 4, 5)
            if 'loss_f' in df_loss.columns and 'loss_b' in df_loss.columns:
                ax5.plot(df_loss['iter'], df_loss['loss_f'], 'r-', label='Physics Loss', linewidth=1.5)
                ax5.plot(df_loss['iter'], df_loss['loss_b'], 'g-', label='Boundary Loss', linewidth=1.5)
                if 'loss_d' in df_loss.columns:
                    ax5.plot(df_loss['iter'], df_loss['loss_d'], 'orange', label='Data Loss', linewidth=1.5)
                ax5.set_yscale('log')
                ax5.grid(True, alpha=0.3)
                ax5.set_xlabel('Iteration')
                ax5.set_ylabel('Loss Components')
                ax5.legend(fontsize=8)

            # 2. Ground Truth (Top row)
            if self.u_true is not None:
                ax2 = plt.subplot(2, 4, 2)
                u_true_grid = self.u_true[:, i].reshape(grid_size, grid_size)
                im2 = ax2.contourf(x_grid, y_grid, u_true_grid, levels=50, cmap='rainbow')
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                ax2.set_title('Ground Truth')
                ax2.set_aspect('equal')
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # 3. PINN Prediction (Top row)
            ax3 = plt.subplot(2, 4, 3)
            u_pred_grid = self.u[:, i].reshape(grid_size, grid_size)
            im3 = ax3.contourf(x_grid, y_grid, u_pred_grid, levels=50, cmap='rainbow')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_title(f'PINN Prediction')
            ax3.set_aspect('equal')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # 4. Absolute Error (Top row)
            if self.u_true is not None:
                ax4 = plt.subplot(2, 4, 4)
                error_grid = absolute_error[:, i].reshape(grid_size, grid_size)
                im4 = ax4.contourf(x_grid, y_grid, error_grid, levels=50, cmap='hot')
                ax4.set_xlabel('x')
                ax4.set_ylabel('y')
                ax4.set_title(f'Absolute Error\n(MSE: {mse_values[i]:.2e})')
                ax4.set_aspect('equal')
                plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

                # 5. Error statistics in bottom right
                ax6 = plt.subplot(2, 4, 6)
                ax6.hist(absolute_error[:, i], bins=50, color='red', alpha=0.7, edgecolor='black')
                ax6.set_xlabel('Absolute Error')
                ax6.set_ylabel('Frequency')
                ax6.set_title('Error Distribution')
                ax6.grid(True, alpha=0.3)

                # 6. Relative error
                ax7 = plt.subplot(2, 4, 7)
                relative_error = np.abs(self.u[:, i] - self.u_true[:, i]) / (np.abs(self.u_true[:, i]) + 1e-10)
                rel_error_grid = relative_error.reshape(grid_size, grid_size)
                im7 = ax7.contourf(x_grid, y_grid, rel_error_grid, levels=50, cmap='hot')
                ax7.set_xlabel('x')
                ax7.set_ylabel('y')
                ax7.set_title('Relative Error')
                ax7.set_aspect('equal')
                plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

                # 7. Error metrics text
                ax8 = plt.subplot(2, 4, 8)
                ax8.axis('off')
                max_error = np.max(absolute_error[:, i])
                mean_error = np.mean(absolute_error[:, i])
                std_error = np.std(absolute_error[:, i])
                max_rel_error = np.max(relative_error)
                mean_rel_error = np.mean(relative_error)

                metrics_text = f"""Error Metrics:
                
MSE: {mse_values[i]:.4e}
Max Abs Error: {max_error:.4e}
Mean Abs Error: {mean_error:.4e}
Std Abs Error: {std_error:.4e}

Max Rel Error: {max_rel_error:.4e}
Mean Rel Error: {mean_rel_error:.4e}
                """
                ax8.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                        verticalalignment='center')

            plt.suptitle(f'{self.ques_name} - {self.module_name} - Comprehensive Results',
                        fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"{self.figure_desti}{self.ques_name}_comprehensive_{i+1}_{self.module_name}.png",
                       bbox_inches='tight', dpi=300)
            plt.close()

    def simple_comparison_figure(self):
        """
        Create simple 2x2 comparison: Ground Truth, Prediction, Error, Loss
        """
        self.figure_desti = self.file_densti + '/Figure/'
        if not os.path.exists(self.figure_desti):
            os.mkdir(self.figure_desti)

        # Read loss data
        loss_desti = self.file_densti + '/Loss/'
        df_loss = pd.read_csv(f'{loss_desti}{self.ques_name}_{self.ini_num}_loss_{self.module_name}.csv')

        grid_size = int(np.sqrt(len(self.x)))
        x_grid = self.x.reshape(grid_size, grid_size)
        y_grid = self.y.reshape(grid_size, grid_size)

        for i in range(self.u.shape[1]):
            fig, axes = plt.subplots(2, 2, figsize=(10, 9))

            # Loss curve
            axes[0, 0].plot(df_loss['iter'], df_loss['loss'], 'b-', linewidth=2)
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].set_title('Loss Propagation')

            # Ground Truth
            if self.u_true is not None:
                u_true_grid = self.u_true[:, i].reshape(grid_size, grid_size)
                im1 = axes[0, 1].contourf(x_grid, y_grid, u_true_grid, levels=50, cmap='rainbow')
                axes[0, 1].set_title('Ground Truth')
                axes[0, 1].set_xlabel('x')
                axes[0, 1].set_ylabel('y')
                axes[0, 1].set_aspect('equal')
                plt.colorbar(im1, ax=axes[0, 1])

            # Prediction
            u_pred_grid = self.u[:, i].reshape(grid_size, grid_size)
            im2 = axes[1, 0].contourf(x_grid, y_grid, u_pred_grid, levels=50, cmap='rainbow')
            axes[1, 0].set_title('PINN Prediction')
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('y')
            axes[1, 0].set_aspect('equal')
            plt.colorbar(im2, ax=axes[1, 0])

            # Error
            if self.u_true is not None:
                error = np.abs(self.u[:, i] - self.u_true[:, i])
                error_grid = error.reshape(grid_size, grid_size)
                im3 = axes[1, 1].contourf(x_grid, y_grid, error_grid, levels=50, cmap='hot')
                mse = np.mean(error**2)
                axes[1, 1].set_title(f'Absolute Error (MSE: {mse:.2e})')
                axes[1, 1].set_xlabel('x')
                axes[1, 1].set_ylabel('y')
                axes[1, 1].set_aspect('equal')
                plt.colorbar(im3, ax=axes[1, 1])

            plt.suptitle(f'{self.ques_name} - {self.module_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.figure_desti}{self.ques_name}_simple_{i+1}_{self.module_name}.png",
                       bbox_inches='tight', dpi=300)
            plt.close()
