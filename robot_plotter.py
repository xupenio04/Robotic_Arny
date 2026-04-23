import matplotlib.pyplot as plt
import numpy as np
import math

import homogeneous_matrix as hm
import omx_kinematic as ok

class omxPlotterClass():

    def plot_frame(self, ax, T, axis_length=0.08, label=None):
        """Plot one coordinate frame given its homogeneous transform T."""
        origin = T[:3, 3]
        R = T[:3, :3]

        x_axis = origin + axis_length * R[:, 0]
        y_axis = origin + axis_length * R[:, 1]
        z_axis = origin + axis_length * R[:, 2]

        # X axis in red
        ax.plot(
            [origin[0], x_axis[0]],
            [origin[1], x_axis[1]],
            [origin[2], x_axis[2]],
            color='r', linewidth=2
        )

        # Y axis in green
        ax.plot(
            [origin[0], y_axis[0]],
            [origin[1], y_axis[1]],
            [origin[2], y_axis[2]],
            color='g', linewidth=2
        )

        # Z axis in blue
        ax.plot(
            [origin[0], z_axis[0]],
            [origin[1], z_axis[1]],
            [origin[2], z_axis[2]],
            color='b', linewidth=2
        )

        if label is not None:
            ax.text(origin[0], origin[1], origin[2], label)

    def plot_robot(self, matrices):
        """
        Plot robot frames and links.
        
        Args:
            matrices: Can be either:
                - List of tuples [(T1, label1), (T2, label2), ...]
                - List of transformation matrices [T1, T2, ...]
        """
        frames = []
        labels = []
        
        # Check if matrices is a list of tuples or just matrices
        if len(matrices) > 0 and isinstance(matrices[0], tuple):
            # Format: list of tuples (matrix, label)
            for m in matrices:
                frames.append(m[0])
                labels.append(m[1])
        else:
            # Format: list of matrices
            for i, m in enumerate(matrices):
                frames.append(m)
                labels.append(f"Frame {i+1}")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the links between frames
        for i in range(1, len(frames)):
            ax.plot([frames[i-1][0,3], frames[i][0,3]], 
                   [frames[i-1][1,3], frames[i][1,3]], 
                   [frames[i-1][2,3], frames[i][2,3]], 
                   '-o', linewidth=2, color='gray')

        # Plot the frames and labels
        for i in range(len(frames)):
            self.plot_frame(ax, frames[i], axis_length=0.05, label=labels[i])

        # Set equal aspect ratio
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Robot frames and links')
        
        # Get bounds for equal aspect ratio
        all_points = []
        for frame in frames:
            all_points.append(frame[:3, 3])
        all_points = np.array(all_points)
        
        if len(all_points) > 0:
            max_range = np.max(all_points, axis=0) - np.min(all_points, axis=0)
            max_range = np.max(max_range) / 2.0
            mid_x = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) * 0.5
            mid_y = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) * 0.5
            mid_z = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()


def test_with_your_angles():
    """Test with your specific angles [0, 45, -45, -90, 45]"""
    
    # Define robot link lengths (adjust these to match your robot)
    l1 = 40
    l2 = 44.5
    l3 = 113.2
    l4 = 41.5
    l5 = 162
    l6 = 43.2
    l7 = 80.5
    
    # Create robot instance
    robot = ok.omxKinematicClass(l1, l2, l3, l4, l5, l6, l7)
    
    your_angles = [0, 0, 0, np.pi/4, 0]  
    
    print(f"Testing with angles (degrees): {your_angles}")
    
    # Get the transformation result
    result = robot.forward_kinematics(*your_angles)
    
    # Debug information
    if isinstance(result, np.ndarray):
        print(f"Matrix shape: {result.shape}")
        print(f"\nTransformation matrix:")
        print(result)

    # Prepare transforms for plotting
    if isinstance(result, np.ndarray) and result.shape == (4,4):
        transforms = [(result, "End Effector")]

    elif isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], tuple):
            transforms = result
        elif len(result) > 0 and isinstance(result[0], np.ndarray):
            transforms = [(result[i], f"Joint {i+1}") for i in range(len(result))]
        else:
            # Assume it's a list of matrices in a different format
            print("Attempting to interpret the result...")
            transforms = []
            for i, item in enumerate(result):
                if isinstance(item, np.ndarray) and item.shape == (4,4):
                    transforms.append((item, f"Frame {i+1}"))
                else:
                    print(f"Unexpected item {i}: {type(item)}") 

    else:
        print("Cannot interpret the return value for plotting")
        return
    
    print(f"\nPlotting {len(transforms)} frames...")
    plotter = omxPlotterClass()
    plotter.plot_robot(transforms)

def main(args=None):
    """
    Main function to validate the kinematics implementation.
    Run different validation tests.
    """
    
    test_with_your_angles()


if __name__ == '__main__':
    main()