import torch

# Given the filename, load the specified model.
# These are hardcoded, based on the MATLAB results.
# The WSINDy results are in the from u_tt = (f(u))_{...}
# We write out just the right hand side, expanding f(u) if needed.
def get_wsindy_model(filename, m = 3):
    # get just the file, without path (remove .mat extension)
    filename = filename.split("/")[-1].split(".mat")[0]

    filenames = [
        'ssx1_sst1_nl0.0_nx100_nt2000_breather',
        'ssx5_sst80_nl0.0_nx100_nt2000_breather',
        'ssx5_sst80_nl0.05_nx100_nt2000_breather',
        'ssx5_sst80_nl0.2_nx100_nt2000_breather',
        'ssx5_sst80_nl0.5_nx100_nt2000_breather'
    ]

    # going to change to support some hyperparameter. Likely m_x and m_t
    # Other settings from matlab:
    # polys = 0 1 2 3 4 5
    # trigs =
    # Max derivs[t x] = 2 2
    # [m_x m_t] = 3 3

    if m == 3:
        # polys = 0 1 2 3
        # [m_x m_t] = 3 3
        # [s_x s_t] = 1 1
        # [p_x p_t] = 15 13
        if filename == filenames[0]:
            # -0.9887268501174269u^{1}_{}
            # + 0.9985737786157007u^{1}_{xx}
            # + 0.1520814930607303u^{3}_{}
            # + -0.005237418280344855u^{5}_{}
            def N(big_u):
                u = big_u[:, 0:1]
                u_xx = big_u[:, 2:3]
                out = -0.9887268501174269 * u + 0.9985737786157007 * u_xx + \
                    0.1520814930607303 * u**3 - 0.005237418280344855 * u**5
                return out
        elif filename == filenames[1]:
            # -0.822140455755885u^{1}_{} + 0.751698664403035u^{1}_{xx} + 0.09050432817958863u^{3}_{}
            def N(big_u):
                u = big_u[:, 0:1]
                u_xx = big_u[:, 2:3]
                out = -0.822140455755885 * u + 0.751698664403035 * u_xx + 0.09050432817958863 * u**3
                return out
        elif filename == filenames[2]:
            # -0.8365898601806127u^{1}_{} + 0.8901034142656095u^{1}_{xx} + 0.09742200214850787u^{3}_{}
            def N(big_u):
                u = big_u[:, 0:1]
                u_xx = big_u[:, 2:3]
                out = -0.8365898601806127 * u + 0.8901034142656095 * u_xx + 0.09742200214850787 * u**3
                return out
        elif filename == filenames[3]:
            # -0.9596113071178547u^{1}_{} + 1.44581134619929u^{1}_{xx} + 0.1228362744846409u^{3}_{}
            def N(big_u):
                u = big_u[:, 0:1]
                u_xx = big_u[:, 2:3]
                out = -0.9596113071178547 * u + 1.44581134619929 * u_xx + 0.1228362744846409 * u**3
                return out
        else:
            N = None
    elif m == 4:
        # match filename:
        if filename == filenames[1]:
            # -0.8381789079447528u ^ {1}_{} + 0.8711070327415621u ^ {1}_{xx} + 0.08094585188231207u ^ {3}_{}
            def N(big_u):
                u = big_u[:, 0:1]
                u_xx = big_u[:, 2:3]
                out = -0.8381789079447528 * u + 0.8711070327415621 * u_xx + 0.08094585188231207 * u**3
                return out
        elif filename == filenames[2]:
            # -0.8093538356858091u ^ {1}_{} + 0.8462217731332211u ^ {1}_{xx} + 0.07513729884203607u ^ {3}_{}
            def N(big_u):
                u = big_u[:, 0:1]
                u_xx = big_u[:, 2:3]
                out = -0.8093538356858091 * u + 0.8462217731332211 * u_xx + 0.07513729884203607 * u**3
                return out
        elif filename == filenames[3]:
            # -0.7986241642033896u^{1}_{} + 0.9602487576975071u^{1}_{xx} + -0.0438081646146268u^{2}_{}
            # + 0.05578803891678051u^{3}_{}
            def N(big_u):
                u = big_u[:, 0:1]
                u_xx = big_u[:, 2:3]
                out = -0.7986241642033896 * u + 0.9602487576975071 * u_xx + -0.0438081646146268 * u**2 +\
                      0.05578803891678051 * u**3
                return out
        elif filename == filenames[4]:
            # -0.8021026221898665u^{1}_{} + 1.11840741844505u^{1}_{xx} + -0.1677771849025387u^{2}_{}
            def N(big_u):
                u = big_u[:, 0:1]
                u_xx = big_u[:, 2:3]
                out = -0.8021026221898665 * u + 1.11840741844505 * u_xx - 0.1677771849025387 * u**2
                return out
        else:
            N = None


    return N




