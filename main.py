import math
import numpy as np

from omx_driver import OmxDriver
from trajectory_generator import TrajectoryGenerator

# ── Parâmetros do gerador de trajetória ───────────────────────────────────────

N_JOINTS = 6
VMAX     = [math.radians(20)] * N_JOINTS   # 20 °/s por junta
AMAX     = [math.radians(10)] * N_JOINTS   # 10 °/s² por junta
TS       = 0.01                             # período de controle (s)


def get_joint_input(n_joints: int) -> tuple[int, float]:
    """Solicita ao usuário a junta e o ângulo desejado.

    Returns
    -------
    joint_index : int
        Índice da junta (0-based internamente).
    angle_rad : float
        Ângulo de rotação desejado em radianos.
    """
    while True:
        try:
            joint = int(input(f"\nJunta a mover [1-{n_joints}]: "))
            if not (1 <= joint <= n_joints):
                print(f"  ✗ Escolha uma junta entre 1 e {n_joints}.")
                continue
            break
        except ValueError:
            print("  ✗ Digite um número inteiro.")

    while True:
        try:
            angle_deg = float(input("Ângulo de rotação (graus): "))
            break
        except ValueError:
            print("  ✗ Digite um número válido.")

    return joint - 1, math.radians(angle_deg)


def main():
    print("=" * 50)
    print("   Teste de trajetória — OpenManipulator-X")
    print("=" * 50)

    # ── Conecta ao robô ───────────────────────────────────────────────────────
    try:
        driver = OmxDriver()
    except Exception as e:
        print(f"\n[ERRO] Não foi possível conectar ao robô: {e}")
        return

    # ── Lê posição atual como ponto de partida ────────────────────────────────
    print("\nLendo posição atual das juntas...")
    qi = driver.read_joint_positions()
    print("Posição atual (rad):", [f"{q:.4f}" for q in qi])
    print("Posição atual (°):  ", [f"{math.degrees(q):.2f}" for q in qi])

    # ── Input do usuário ──────────────────────────────────────────────────────
    joint_idx, delta_rad = get_joint_input(N_JOINTS)

    # ponto final: só a junta escolhida se move
    qf = list(qi)
    qf[joint_idx] += delta_rad

    print(f"\nJunta {joint_idx + 1}:")
    print(f"  qi = {math.degrees(qi[joint_idx]):.2f}°  →  "
          f"qf = {math.degrees(qf[joint_idx]):.2f}°  "
          f"(Δ = {math.degrees(delta_rad):+.2f}°)")

    # ── Gera trajetória ───────────────────────────────────────────────────────
    print("\nGerando trajetória...")
    tg = TrajectoryGenerator(N_JOINTS, VMAX, AMAX, ts=TS)
    traj, vel, acc = tg.compute_trajectory(qi, qf)

    n_steps = traj.shape[0]
    T_total = n_steps * TS
    print(f"  {n_steps} amostras  |  duração ≈ {T_total:.2f} s")

    # ── Confirmação antes de executar ─────────────────────────────────────────
    confirm = input("\nExecutar no robô? [s/N]: ").strip().lower()
    if confirm != "s":
        print("Execução cancelada.")
        return

    # ── Executa no robô ───────────────────────────────────────────────────────
    print("\nExecutando trajetória...")
    driver.execute_trajectory(traj.tolist(), TS)
    print("Trajetória concluída.")

    # ── Lê posição final ──────────────────────────────────────────────────────
    qf_real = driver.read_joint_positions()
    print("\nPosição final lida (rad):", [f"{q:.4f}" for q in qf_real])
    print("Posição final lida (°):  ", [f"{math.degrees(q):.2f}" for q in qf_real])

    err_deg = math.degrees(abs(qf_real[joint_idx] - qf[joint_idx]))
    print(f"\nErro na junta {joint_idx + 1}: {err_deg:.3f}°")


if __name__ == "__main__":
    main()