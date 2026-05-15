import numpy as np
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    """
    Geração de trajetórias com perfis triangular/trapezoidal.
    """

    def __init__(self, n, Vn, An, ts=0.01):
 
        self.n = int(n)
        self.Vn = np.array(Vn, dtype=float)
        self.An = np.array(An, dtype=float)
        self.ts = float(ts)

        if len(self.Vn) != self.n:
            raise ValueError("Vn deve ter tamanho n")

        if len(self.An) != self.n:
            raise ValueError("An deve ter tamanho n")

    def compute_max_duration(self, qi, qf):

        qi = np.array(qi, dtype=float)
        qf = np.array(qf, dtype=float)

        if len(qi) != self.n or len(qf) != self.n:
            raise ValueError("qi e qf devem ter tamanho n")

        durations = []

        for n in range(self.n):

            dq = abs(qf[n] - qi[n])

            Vmax = self.Vn[n]
            Amax = self.An[n]

            # tempo para atingir velocidade máxima
            Tacc = Vmax / Amax

            # distância durante aceleração
            Dacc = 0.5 * Amax * Tacc**2

            # aceleração + desaceleração
            Dacc_dec = 2 * Dacc

            # Perfil triangular
            if dq < Dacc_dec:
                T = 2 * np.sqrt(dq / Amax)

            # Perfil trapezoidal
            else:
                Tcst = (dq - Dacc_dec) / Vmax
                T = 2 * Tacc + Tcst

            durations.append(T)

        Tmax = max(durations)

        return Tmax

    def generate_trajectory(self, qi, qf):

        qi = np.array(qi, dtype=float)
        qf = np.array(qf, dtype=float)

        T = self.compute_max_duration(qi, qf)

        time = np.arange(0, T + self.ts, self.ts)

        traj = np.zeros((len(time), self.n))
        vel  = np.zeros((len(time), self.n))
        acc  = np.zeros((len(time), self.n))

        for n in range(self.n):

            dq = abs(qf[n] - qi[n])
            s = np.sign(qf[n] - qi[n])
            print(f"dq:{dq}, s:{s}")
            A = self.An[n]

            disc = (A*T)**2 - 4*A*dq

            if disc < 0:
                A_eff = 4 * dq / T**2
                Tacc = T / 2
                V = A_eff * Tacc
                Tcst = 0

            else:
                V = (A*T - np.sqrt(disc)) / 2
                Tacc = V / A
                Tcst = T - 2*Tacc

            for k, t in enumerate(time):

                if t <= Tacc:
                    # FASE DE ACELERAÇÃO
                    q = qi[n] + s * 0.5 * A * t**2
                    v = s * A * t
                    a = s * A

                elif t <= Tacc + Tcst:
                    # FASE DE VELOCIDADE CONSTANTE
                    q1 = 0.5 * A * Tacc**2
                    q = qi[n] + s * (q1 + V * (t - Tacc))
                    v = s * V
                    a = 0

                else:
                    # FASE DE DESACELERAÇÃO
                    q = qf[n] - s * 0.5 * A * (T - t)**2
                    v = s * A * (T - t)
                    a = -s * A

                traj[k, n] = q
                vel[k, n]  = v
                acc[k, n]  = a


        return traj, vel, acc

    def plot_trajectory(self, traj):

        traj = np.array(traj)

        n_steps = traj.shape[0]

        time = np.arange(n_steps) * self.ts

        fig, axes = plt.subplots(
            self.n, 1,
            figsize=(10, 2.8*self.n),
            sharex=True
        )

        # caso exista só 1 junta
        if self.n == 1:
            axes = [axes]

        for n in range(self.n):

            axes[n].plot(time, traj[:, n], linewidth=2)

            axes[n].set_ylabel(f'Joint {n+1}')
            axes[n].grid(True)

        axes[-1].set_xlabel("Time (s)")

        fig.suptitle("Joint Trajectories", fontsize=14)

        plt.tight_layout()
        plt.show()

    def resample_trajectory(self, trajectories):
        max_samples = max(len(traj) for traj in trajectories)

        resampled = []

        for traj in trajectories:

            old_n = len(traj)

            # grade antiga normalizada
            old_x = np.linspace(0, 1, old_n)

            # nova grade comum
            new_x = np.linspace(0, 1, max_samples)

            # interpolação
            new_traj = np.interp(new_x, old_x, traj)

            resampled.append(new_traj)

        # cada coluna = junta
        return np.array(resampled).T


    def compute_trajectory(self, qi, qf):

        traj, vel, acc = self.generate_trajectory(qi, qf)

        print(f"Generated trajectory with {len(traj)}, {len(vel)}, {len(acc)} samples before resampling.")

        # 3. reamostrar
        traj = self.resample_trajectory(
            [traj[:, k] for k in range(self.n)]
        )

        vel = self.resample_trajectory(
            [vel[:, k] for k in range(self.n)]
        )

        acc = self.resample_trajectory(
            [acc[:, k] for k in range(self.n)]
        )

        print(f"Generated trajectory with {len(traj)}, {len(vel)}, {len(acc)} samples.")

        return traj, vel, acc