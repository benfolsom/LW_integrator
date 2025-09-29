import numpy as np

def calculate_plotting_variables(retarded_traj, retarded_drv_traj, init_rider, init_driver, steps, E_MeV_rest_rider, E_MeV_rest_driver, c_ms):
    delta_e = [np.mean(retarded_traj[i]['gamma'] - init_rider['gamma']) * E_MeV_rest_rider for i in range(2, steps - 1)]
    delta_e_drv = [np.mean(retarded_drv_traj[i]['gamma'] - init_driver['gamma']) * E_MeV_rest_driver for i in range(2, steps - 1)]

    e = [np.mean(retarded_traj[i]['gamma']) * E_MeV_rest_rider for i in range(2, steps - 1)]
    e_drv = [np.mean(retarded_drv_traj[i]['gamma']) * E_MeV_rest_driver for i in range(2, steps - 1)]

    delta_e_keV = np.multiply(delta_e, 1E3)  # convert 1 amu*c^2 to keV
    delta_e_MeV = np.multiply(delta_e, 1)  # convert 1 amu*c^2 to MeV
    e_GeV = np.multiply(e, 1E-3)

    delta_e_keV_drv = np.multiply(delta_e_drv, 1e3)  # convert 1 amu*c^2 to keV
    delta_e_MeV_drv = np.multiply(delta_e_drv, 1)  # convert 1 amu*c^2 to MeV
    e_GeV_drv = np.multiply(e_drv, 1e-3)

    zs = [np.mean(retarded_traj[i]['z']) for i in range(2, steps - 1)]
    xs = [np.mean(retarded_traj[i]['x']) for i in range(2, steps - 1)]
    zs_drv = [np.mean(retarded_drv_traj[i]['z']) for i in range(2, steps - 1)]
    xs_drv = [np.mean(retarded_drv_traj[i]['x']) for i in range(2, steps - 1)]

    bzs = [retarded_traj[i]['bz'][0] for i in range(2, steps - 1)]
    bxs = [retarded_traj[i]['bx'][0] for i in range(2, steps - 1)]
    bys = [retarded_traj[i]['by'][0] for i in range(2, steps - 1)]
    bdotxs = [retarded_traj[i]['bdotx'][0] for i in range(2, steps - 1)]
    bdotys = [retarded_traj[i]['bdoty'][0] for i in range(2, steps - 1)]
    bdotzs = [retarded_traj[i]['bdotz'][0] for i in range(2, steps - 1)]

    Pzs = [retarded_traj[i]['Pz'][0] for i in range(2, steps - 1)]
    Pxs = [retarded_traj[i]['Px'][0] for i in range(2, steps - 1)]
    Pys = [retarded_traj[i]['Py'][0] for i in range(2, steps - 1)]
    Pts = [retarded_traj[i]['Pt'][0] for i in range(2, steps - 1)]
    gammas = [retarded_traj[i]['gamma'][0] for i in range(2, steps - 1)]
    tees = [retarded_traj[i]['t'][0] for i in range(2, steps - 1)]

    bzs_drv = [retarded_drv_traj[i]['bz'][0] for i in range(2, steps - 1)]
    bxs_drv = [retarded_drv_traj[i]['bx'][0] for i in range(2, steps - 1)]
    bys_drv = [retarded_drv_traj[i]['by'][0] for i in range(2, steps - 1)]
    bdotxs_drv = [retarded_drv_traj[i]['bdotx'][0] for i in range(2, steps - 1)]
    bdotys_drv = [retarded_drv_traj[i]['bdoty'][0] for i in range(2, steps - 1)]
    bdotzs_drv = [retarded_drv_traj[i]['bdotz'][0] for i in range(2, steps - 1)]

    Pzs_drv = [retarded_drv_traj[i]['Pz'][0] for i in range(2, steps - 1)]
    Pxs_drv = [retarded_drv_traj[i]['Px'][0] for i in range(2, steps - 1)]
    Pys_drv = [retarded_drv_traj[i]['Py'][0] for i in range(2, steps - 1)]
    Pts_drv = [retarded_drv_traj[i]['Pt'][0] for i in range(2, steps - 1)]
    gammas_drv = [retarded_drv_traj[i]['gamma'][0] for i in range(2, steps - 1)]
    tees_drv = [retarded_drv_traj[i]['t'][0] for i in range(2, steps - 1)]
    qs_drv = [retarded_drv_traj[i]['q'] for i in range(2, steps - 1)]

    gam_fixed = 1 / np.sqrt(1 - np.square(np.sqrt(np.square(bzs) + np.square(bys) + np.square(bxs))))
    bdotzs_seconds = np.multiply(bdotzs, 1E9)
    q_statC_squared = (4.8032E-10) ** 2
    q_kg_m3_sneg2 = q_statC_squared / 1000 * (1 / 100 ** 3)
    Pows_z_rider_watts = 2 / 3 * q_kg_m3_sneg2 / c_ms * np.power(gammas, 6) * np.power(bdotzs_seconds, 2)

    return (delta_e, delta_e_drv, e, e_drv, delta_e_keV, delta_e_MeV, e_GeV, delta_e_keV_drv,
            delta_e_MeV_drv, e_GeV_drv, zs, xs, zs_drv, xs_drv, bzs, bxs, bys, bdotxs, bdotys, bdotzs,
            Pzs, Pxs, Pys, Pts, gammas, tees, bzs_drv, bxs_drv, bys_drv, bdotxs_drv, bdotys_drv, bdotzs_drv,
            Pzs_drv, Pxs_drv, Pys_drv, Pts_drv, gammas_drv, tees_drv, qs_drv, gam_fixed, bdotzs_seconds,
            q_statC_squared, q_kg_m3_sneg2, Pows_z_rider_watts)
