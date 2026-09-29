"""Exported verbatim from notebook code cell 10."""

import numpy as np
import matplotlib.pyplot as plt


def plot_fim_system(P_B, P_B_def, P_U, P_U_def,
                    R_B=None, R_U=None,
                    NH_B=None, NV_B=None, NH_U=None, NV_U=None):
    """
    Python conversion of plot_fim_system.m
    """

    # =========================================================
    # 1) FULL SYSTEM VIEW
    # =========================================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(P_B[:, 0], P_B[:, 1], P_B[:, 2], marker="o", label="BS rigid")
    ax.scatter(P_B_def[:, 0], P_B_def[:, 1], P_B_def[:, 2], marker="x", label="BS deformed")

    ax.scatter(P_U[:, 0], P_U[:, 1], P_U[:, 2], marker="o", label="UE rigid")
    ax.scatter(P_U_def[:, 0], P_U_def[:, 1], P_U_def[:, 2], marker="x", label="UE deformed")

    if R_B is not None:
        ax.scatter(R_B[:, 0], R_B[:, 1], R_B[:, 2], marker="s", label="BS antennas")

    if R_U is not None:
        ax.scatter(R_U[:, 0], R_U[:, 1], R_U[:, 2], marker="s", label="UE antennas")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Full FIM system")
    ax.legend()
    ax.view_init(elev=20, azim=-30)
    ax.set_box_aspect([1, 1, 1])

    # =========================================================
    # 2) DISPLACEMENT VIEW
    # =========================================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(P_B[:, 0], P_B[:, 1], P_B[:, 2], marker="o", label="Rigid")
    ax.scatter(P_B_def[:, 0], P_B_def[:, 1], P_B_def[:, 2], marker="x", label="Deformed")

    for n in range(P_B.shape[0]):
        ax.plot(
            [P_B[n, 0], P_B_def[n, 0]],
            [P_B[n, 1], P_B_def[n, 1]],
            [P_B[n, 2], P_B_def[n, 2]],
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("BS FIM deformation displacement vectors")
    ax.legend()
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])

    # =========================================================
    # 3) MESH VIEW
    # =========================================================
    if None not in (NH_B, NV_B, NH_U, NV_U):
        XB_r = P_B[:, 0].reshape((NV_B, NH_B))
        YB_r = P_B[:, 1].reshape((NV_B, NH_B))
        ZB_r = P_B[:, 2].reshape((NV_B, NH_B))

        XB_d = P_B_def[:, 0].reshape((NV_B, NH_B))
        YB_d = P_B_def[:, 1].reshape((NV_B, NH_B))
        ZB_d = P_B_def[:, 2].reshape((NV_B, NH_B))

        XU_r = P_U[:, 0].reshape((NV_U, NH_U))
        YU_r = P_U[:, 1].reshape((NV_U, NH_U))
        ZU_r = P_U[:, 2].reshape((NV_U, NH_U))

        XU_d = P_U_def[:, 0].reshape((NV_U, NH_U))
        YU_d = P_U_def[:, 1].reshape((NV_U, NH_U))
        ZU_d = P_U_def[:, 2].reshape((NV_U, NH_U))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_wireframe(XB_r, YB_r, ZB_r, label="BS rigid")
        ax.plot_wireframe(XB_d, YB_d, ZB_d, label="BS deformed")

        ax.plot_wireframe(XU_r, YU_r, ZU_r, label="UE rigid")
        ax.plot_wireframe(XU_d, YU_d, ZU_d, label="UE deformed")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("FIM surfaces rigid vs deformed")
        ax.legend()
        ax.view_init(elev=30, azim=-60)
        ax.set_box_aspect([1, 1, 1])

    plt.show()
