from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import trajectory_generator as tg


# def compute_derivatives(traj, ts):

#     vel = np.gradient(traj, ts, axis=0)
#     acc = np.gradient(vel,  ts, axis=0)
#     return vel, acc


def plot_joint_trajectories(traj, vel, acc, ts, joint_names=None):

    traj = np.array(traj)
    vel  = np.array(vel)
    acc  = np.array(acc)

    n_samples, n_joints = traj.shape

    time = np.arange(n_samples) * ts

    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(n_joints)]

    colors = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#BA7517']

    fig = plt.figure(figsize=(14, 3.2 * n_joints))
    fig.suptitle("Trajetórias das Juntas", fontsize=15, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(
        n_joints, 3,
        figure=fig,
        hspace=0.55,
        wspace=0.35
    )

    col_titles = ["Posição (rad)", "Velocidade (rad/s)", "Aceleração (rad/s²)"]
    data_sets  = [traj, vel, acc]

    for j in range(n_joints):

        color = colors[j % len(colors)]

        for c, (data, col_title) in enumerate(zip(data_sets, col_titles)):

            ax = fig.add_subplot(gs[j, c])

            # ax.plot(time, data[:, j], color=color, linewidth=1.8)

            if col_title == "Aceleração (rad/s²)":
                ax.step(time, data[:, j], where='post', color=color, linewidth=1.8)
            else:
                ax.plot(time, data[:, j], color=color, linewidth=1.8)

            ax.set_ylabel(joint_names[j], fontsize=10)
            ax.set_xlabel("Tempo (s)", fontsize=9)

            ax.grid(True, alpha=0.3, linewidth=0.6)
            ax.tick_params(labelsize=8)

            if j == 0:
                ax.set_title(col_title, fontsize=11, fontweight='bold', pad=8)

            ax.set_facecolor(f"{color}08")

    plt.tight_layout()
    plt.savefig("joint_trajectories.png", dpi=150, bbox_inches="tight")
    plt.show()

def main():

    n_joints = 5
    Vmax = [1, 1, 1, 1, 1]   
    Amax = [2, 2, 2, 2, 2]   
    ts   = 0.01               

    qi = [0, 0, 0, 0, 0]
    qf = [0.2, 0.1, -0.15, 0.05, 0.1]

    # ── Gera trajetória ───────────────────────────────────────────────────────
    Tg = tg.TrajectoryGenerator(n_joints, Vmax, Amax, ts=ts)
    traj, vel, acc = Tg.compute_trajectory(qi, qf)

    joint_names = [f"J{i+1}" for i in range(n_joints)]
    plot_joint_trajectories(traj, vel, acc, ts, joint_names)

if __name__ == "__main__":
    main()