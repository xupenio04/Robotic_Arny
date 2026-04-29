import matplotlib.pyplot as plt
import numpy as np
import math

import homogeneous_matrix as hm
import omx_kinematic as ok
import trajectory_generator as tg

from matplotlib.animation import FuncAnimation


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
        """
        frames = []
        labels = []

        if len(matrices) > 0 and isinstance(matrices[0], tuple):
            for m in matrices:
                frames.append(m[0])
                labels.append(m[1])
        else:
            for i, m in enumerate(matrices):
                frames.append(m)
                labels.append(f"Frame {i+1}")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # links
        for i in range(1, len(frames)):
            ax.plot(
                [frames[i-1][0,3], frames[i][0,3]],
                [frames[i-1][1,3], frames[i][1,3]],
                [frames[i-1][2,3], frames[i][2,3]],
                '-o',
                linewidth=2,
                color='gray'
            )

        # frames
        for i in range(len(frames)):
            self.plot_frame(ax, frames[i], axis_length=20, label=labels[i])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Robot frames and links')

        plt.show()

    def animate_robot(self, robot, trajectory, interval=50):
        """
        trajectory shape = (N,5)
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):

            ax.cla()

            q = trajectory[frame]

            transforms = robot.forward_kinematics(*q)

            # pontos das juntas
            pts = np.array([T[:3, 3] for T in transforms])

            # links
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                '-o',
                linewidth=3,
                color='black'
            )

            # frames locais
            for i, T in enumerate(transforms):
                self.plot_frame(ax, T, axis_length=20)

            ax.set_xlim(-300, 300)
            ax.set_ylim(-300, 300)
            ax.set_zlim(0, 500)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            ax.set_title(f"Frame {frame+1}/{len(trajectory)}")

        ani = FuncAnimation(
            fig,
            update,
            frames=len(trajectory),
            interval=interval,
            repeat=False
        )

        plt.show()


def test_with_your_angles():

    l1 = 40
    l2 = 44.5
    l3 = 113.2
    l4 = 41.5
    l5 = 162
    l6 = 43.2
    l7 = 80.5

    robot = ok.omxKinematicClass(l1, l2, l3, l4, l5, l6, l7)

    your_angles = [0.78539815, 1.93086212, -0.95438512, 1.02167603, 0]

    result = robot.forward_kinematics(*your_angles)

    plotter = omxPlotterClass()
    plotter.plot_robot(result)


def main(args=None):

    l1 = 40
    l2 = 44.5
    l3 = 113.2
    l4 = 41.5
    l5 = 162
    l6 = 43.2
    l7 = 80.5

    Tg = tg.TrajectoryGenerator(
        5,
        [2,2,2,2,2],
        [1,1,1,1,1]
    )

    traj = Tg.compute_trajectory(
        [0,0,0,0,0],
        [1.56,0,0,0,0]
    )

    robot = ok.omxKinematicClass(
        l1,l2,l3,l4,l5,l6,l7
    )

    plotter = omxPlotterClass()

    plotter.animate_robot(robot, traj)


if __name__ == '__main__':
    main()