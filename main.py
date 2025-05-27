import matplotlib.pyplot as plt
import numpy as np
import random
import time
import datetime


def RungeKutta4(r, params, dt, noisex=0., noisey=0.):  # edited; no need for input f
    """ Runge-Kutta 4 method """
    k1 = dt * LV_equations(r, params, noisex, noisey)
    k2 = dt * LV_equations(r + 0.5 * k1, params, noisex, noisey)
    k3 = dt * LV_equations(r + 0.5 * k2, params, noisex, noisey)
    k4 = dt * LV_equations(r + k3, params, noisex, noisey)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


def logisticRungeKutta4(r, params, K, dt, noisex=0., noisey=0.):  # edited; no need for input f
    """ Runge-Kutta 4 method """
    k1 = dt * logistic_LV_equations(r, params, K, noisex, noisey)
    k2 = dt * logistic_LV_equations(r + 0.5 * k1, params, K, noisex, noisey)
    k3 = dt * logistic_LV_equations(r + 0.5 * k2, params, K, noisex, noisey)
    k4 = dt * logistic_LV_equations(r + k3, params, K, noisex, noisey)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


def LV_equations(r, params, noisex=0., noisey=0.):
    alpha = params[0][0]
    beta = params[0][1]
    gamma = params[1][1]
    sigma = params[1][0]
    x, y = r[0], r[1]
    fxd = x * (alpha + beta * y) + noisex
    fyd = y * (gamma + sigma * x) + noisey
    return np.asarray([fxd, fyd])


def logistic_LV_equations(r, params, K, noisex=0., noisey=0.):
    # Parametry
    rx = params[0][0]  # współczynnik wzrostu ofiar
    axy = params[0][1]  # współczynnik drapieżnictwa
    ry = params[1][0]  # śmiertelność drapieżników
    ayx = params[1][1]  # przyrost drapieżników dzięki ofiarom

    # Zmienne stanu
    x, y = r[0], r[1]

    # Równania różniczkowe
    dxdt = rx * x * (1 - (x + axy * y) / K[0]) + noisex
    dydt = ry * y * (1 - (y + ayx * x) / K[1]) + noisey

    return np.asarray([dxdt, dydt])


# def norder_LV_equations(r, params, noise_amp):
#     ret_arr = []
#     for i in range(len(r)):
#         frd = 0
#         for j in range(1, len(r)):
#             if i != j:
#                 frd += r[i] * r[j] * params[i][j]
#             else:
#                 frd += r[i] * params[i][j] + noise_amp[i] * (random.random() - 0.5)
#         ret_arr.append(frd)
#
#     return np.array([ret_arr], float)


def simulate(start_cond, params, duration, noise_par1=0., noise_par2=1., noise_type='uniform'):
    timestep = 0.05

    xpoints, ypoints, xnoise_points, ynoise_points = [], [], [], []
    r = start_cond.copy()
    stop = 0

    for t in range(int(duration / timestep)):
        for i in range(len(r)):
            # print(r[i])
            if r[i] > 10000 or r[i] < 0.001:
                if params[i][i] < 0:
                    stop = 1
                    r[i] = 0
                break
        xpoints.append(r[0])
        ypoints.append(r[1])

        if noise_type == "uniform":
            xnoise, ynoise = noise_par1 * (random.random() - 0.5), noise_par1 * (random.random() - 0.5)
        elif noise_type == "sin":
            xnoise, ynoise = noise_par1 * np.sin(noise_par2 * t * timestep), noise_par1 * np.sin(
                noise_par2 * t * timestep)
        elif noise_type == "gaus":
            xnoise, ynoise = np.random.normal(noise_par1, noise_par2), np.random.normal(noise_par1, noise_par2)
        else:
            xnoise, ynoise = 0, 0

        r += RungeKutta4(r, params, timestep, xnoise, ynoise)
        xnoise_points.append(xnoise - 10)
        # print(np.max(np.abs(xnoise_points + 10 * np.ones(len(xnoise_points)))))

        if stop:
            break
    return xpoints, ypoints, stop


def simulate_draw(start_cond, params, duration, noise_par1=0., noise_par2=1., noise_type='uniform'):
    timestep = 0.001

    xpoints, ypoints, xnoise_points, ynoise_points = [], [], [], []
    r = start_cond.copy()
    stop = 0

    for t in range(int(duration / timestep)):
        for i in range(len(r)):
            # print(r[i])
            if r[i] > 10000 or r[i] < 0.001:
                if params[i][i] < 0:
                    stop = 1
                    r[i] = 0
                break
        xpoints.append(r[0])
        ypoints.append(r[1])

        if noise_type == "uniform":
            xnoise, ynoise = noise_par1 * (random.random() - 0.5), noise_par1 * (random.random() - 0.5)
        elif noise_type == "sin":
            xnoise, ynoise = noise_par1 * np.sin(noise_par2 * t * timestep), noise_par1 * np.sin(
                noise_par2 * t * timestep)
        elif noise_type == "gaus":
            xnoise, ynoise = np.random.normal(noise_par1, noise_par2), np.random.normal(noise_par1, noise_par2)
        else:
            xnoise, ynoise = 0, 0

        r += RungeKutta4(r, params, timestep, xnoise, ynoise)
        xnoise_points.append(30*xnoise - 1)
        ynoise_points.append(30*ynoise - 1)

        if stop:
            break

    tpoints = np.arange(0, len(xpoints) * timestep, timestep)
    fig1 = plt.figure("IN TIME")
    # plt.suptitle("Lotka-Volterra Model - Population in time")
    # plt.title(
    #     r"$\frac{dx}{dt} = \alpha \cdot x - \beta \cdot xy$,      $\frac{dy}{dt} = -\gamma \cdot y + \delta \cdot xy$")
    plt.plot(tpoints, xpoints, 'g')
    plt.plot(tpoints, ypoints, 'r')

    plt.plot(tpoints, xnoise_points, 'b')
    # plt.plot(tpoints, ynoise_points, 'r--')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.savefig("LV_time_graph" + str(round(time.time())) + ".png")

    fig2 = plt.figure("IN XY")
    # plt.suptitle("Lotka-Volterra Model - XY coordinates")
    # plt.title(
    #     r"$\frac{dx}{dt} = \alpha \cdot x - \beta \cdot xy$,      $\frac{dy}{dt} = -\gamma \cdot y + \delta \cdot xy$")
    plt.plot(xpoints, ypoints, 'k')
    plt.scatter(start_cond[0],
                start_cond[1],
                zorder=5,
                color='g',
                label=r"$x_0 =$" + str(np.round(start_cond[0], 2)) + r"$y_0 =$" + str(np.round(start_cond[0], 2)))
    plt.scatter(xpoints[-1],
                ypoints[-1],
                zorder=5,
                color=(1, 0, 0),
                label=r"$x_k =$" + str(np.round(xpoints[-1], 5)) + r"$y_k =$" + str(np.round(ypoints[-1], 5)))
    plt.xlabel("Prey population (x)")
    plt.ylabel("Predator population (y)")
    plt.savefig("LV_XY_graph" + str(round(time.time())) + ".png")

    return xpoints, ypoints, stop


def logistic_simulate_draw(start_cond, params, K, duration, noise_par1=0., noise_par2=1., noise_type='uniform'):
    timestep = 0.001

    xpoints, ypoints, xnoise_points, ynoise_points = [], [], [], []
    r = start_cond.copy()
    stop = 0

    for t in range(int(duration / timestep)):
        xpoints.append(r[0])
        ypoints.append(r[1])

        if noise_type == "uniform":
            xnoise, ynoise = noise_par1 * (random.random() - 0.5), noise_par1 * (random.random() - 0.5)
        elif noise_type == "sin":
            xnoise, ynoise = noise_par1 * np.sin(noise_par2 * t * timestep), noise_par1 * np.sin(
                noise_par2 * t * timestep)
        elif noise_type == "gaus":
            xnoise, ynoise = np.random.normal(noise_par1, noise_par2), np.random.normal(noise_par1, noise_par2)
        else:
            xnoise, ynoise = 0, 0

        r += logisticRungeKutta4(r, params, K, timestep, xnoise, ynoise)
        xnoise_points.append(xnoise - 10)
        ynoise_points.append(ynoise - 10)

        if stop:
            break

    tpoints = np.arange(0, len(xpoints) * timestep, timestep)
    fig1 = plt.figure("IN TIME")
    # plt.suptitle("Lotka-Volterra Model - Population in time")
    # plt.title(
    #     r"$\frac{dx}{dt} = \alpha \cdot x - \beta \cdot xy$,      $\frac{dy}{dt} = -\gamma \cdot y + \delta \cdot xy$")
    plt.plot(tpoints, xpoints, 'g')
    plt.plot(tpoints, ypoints, 'r')
    # plt.plot(tpoints, xnoise_points, 'g--')
    # plt.plot(tpoints, ynoise_points, 'r--')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.savefig("LV_time_graph" + str(round(time.time())) + ".png")

    fig2 = plt.figure("IN XY")
    # plt.suptitle("Lotka-Volterra Model - XY coordinates")
    # plt.title(
    #     r"$\frac{dx}{dt} = \alpha \cdot x - \beta \cdot xy$,      $\frac{dy}{dt} = -\gamma \cdot y + \delta \cdot xy$")
    plt.plot(xpoints, ypoints, 'k')
    plt.xlabel("Prey population (x)")
    plt.ylabel("Predator population (y)")
    plt.savefig("LV_XY_graph" + str(round(time.time())) + ".png")

    return xpoints, ypoints, stop


def gradient_color(color, j, iterations):
    if color == 0:
        return 0.9 - 0.9 / (iterations - 1) * j, 0.2 - 0.2 / (iterations - 1) * j, 0
    if color == 1:
        return 0, 0.9 - 0.9 / (iterations - 1) * j, 0.7 - 0.7 / (iterations - 1) * j
    if color == 2:
        return 0.9 - 0.9 / (iterations - 1) * j, 0.9 - 0.9 / (iterations - 1) * j, 0
    if color == 3:
        return 0, 0.3 - 0.3 / (iterations - 1) * j, 0.7 - 0.7 / (iterations - 1) * j


def abgd_param_changes_quadgraph(start_cond, params_in, iterations):
    fig_quad = plt.figure("Quadgraph")
    # plt.suptitle("LV effects of changing reproduction and death rate")
    # plt.title(
    #     r"$\frac{dx}{dt} = \alpha \cdot x - \beta \cdot xy$,      $\frac{dy}{dt} = -\gamma \cdot y + \delta \cdot xy$")
    names = [r"$\alpha$ =", r"$\beta$ =", r"$\gamma$ =", r"$\delta$ ="]

    for k in range(4):
        para = params_in.copy()
        para[int(k / 2) % 2][k % 2] += 0.1 * iterations
        for j in range(iterations):
            para[int(k / 2) % 2][k % 2] -= 0.1
            xpoints, ypoints, ext = simulate(start_cond, para, 30)
            plt.subplot(220 + 1 + k)
            plt.plot(xpoints, ypoints, color=gradient_color(k, j, iterations),
                     label=names[k] + str(np.round(para[int(k / 2) % 2][k % 2], 2)))
            plt.scatter(abs(para[1][1] / para[1][0]), abs(para[0][0] / para[0][1]), zorder=5,
                        color=gradient_color(k, j, iterations), label="")
        plt.plot(xpoints, ypoints, 'k')
        plt.xlabel("Prey population (x)")
        plt.ylabel("Predator population (y)")
        plt.axis((-0.1, 3.8, -0.1, 4.8))
        plt.legend()
    fig_quad.set_size_inches(16, 12)
    plt.savefig("LV_param_change_graph" + str(round(time.time())) + ".png", bbox_inches='tight')


def start_cond_changes_graph(start_cond, iterations, params_in):
    # fig_cond_change = plt.figure("Start condition changes")
    # plt.suptitle("Lotka-Volterra Model - XY coordinates")
    for j in range(iterations):
        r = [2.8 - 0.2 * j, 2.8 - 0.2 * j]
        xpoints, ypoints, ext = simulate(r, params_in, 60)
        plt.plot(xpoints,
                 ypoints,
                 color=gradient_color(1, j, iterations),
                 label=r"$x_0 =$" + str(np.round(1 + 0.2 * j, 2)) + r"$y_0 =$" + str(np.round(1 + 0.2 * j, 2)))
    plt.xlabel("Prey population (x)")
    plt.ylabel("Predator population (y)")
    plt.legend()
    plt.savefig("LV_start_cond_graph" + str(round(time.time())) + ".png")


def uniform_noise_graph(start_cond, params_in, noise):
    r0 = start_cond.copy()
    xpoints, ypoints, ext = simulate(start_cond, params_in, 100, noise_par1=noise)
    # ___________ plotting _________________________________
    fig_noise_graph = plt.figure("Noise graph")
    # plt.title(
    #     r"$\frac{dx}{dt} = \alpha \cdot x - \beta \cdot xy$,      $\frac{dy}{dt} = -\gamma \cdot y + \delta \cdot xy$")
    plt.scatter(r0[0],
                r0[1],
                zorder=5,
                color='g',
                label=r"$x_0 =$" + str(np.round(r0[0], 2)) + r"$y_0 =$" + str(np.round(r0[0], 2)))
    plt.plot(xpoints, ypoints, 'k')
    plt.scatter(xpoints[-1],
                ypoints[-1],
                zorder=5,
                color=(1, 0, 0),
                label=r"$x_k =$" + str(np.round(xpoints[-1], 5)) + r"$y_k =$" + str(np.round(ypoints[-1], 5)))
    plt.xlabel("Prey population (x)")
    plt.ylabel("Predator population (y)")
    plt.legend()
    plt.savefig("LV_uniform_noise_graph" + str(round(time.time())) + ".png")

    return ext


def log_headline(start_cond, params_in, start, stop, noise_step=1., runs=10):
    return str(datetime.datetime.now()) + ": start_conditions=" + str(start_cond) + ", equation parameters=" + \
           str(params_in) + "\n" + "start=" + str(start) + ", stop=" + str(stop) + ", stop=" + \
           str(noise_step) + ", stop=" + str(runs) + '\n'


def noise_and_extinction(start_cond, params_in, start, stop, noise_step=1., noisepar2=2, noise_type="uniform", runs=10):
    st = time.time()
    r = start_cond.copy()
    extinctions = []
    fig_noise_extiction = plt.figure("Noise extinction graph")

    for n in range(int(start / noise_step), int(stop / noise_step)):
        count = 0
        print(np.round(noise_step * n, int(-np.log10(noise_step) + 1)), ". _____________")
        for i in range(runs):
            xpoints, ypoints, ext = simulate(r, params_in, 100, noise_par1=noise_step * n, noise_par2=noisepar2, noise_type=noise_type)
            count += ext
        print(count)
        extinctions.append(count)
        print(np.round(time.time() - st, 2), "s")
        with open(noise_type + "_noise_logs.txt", 'a') as file:
            if n == int(start / noise_step):
                file.write(log_headline(start_cond, params_in, start, stop, noise_step, runs))
                # plt.axvline(x=2 * min(xpoints), color='b', linewidth=0.5, linestyle=':', label="Note boundaries")
            file.write(
                "    " + str(np.round(noise_step * n, int(-np.log10(noise_step) + 1))) + ", " + str(count) + "\n")

    plt.scatter(np.arange(start, stop, noise_step), np.asarray(extinctions) / runs)
    plt.ylabel("Probability of cycle ending in extinction")
    plt.xlabel("Uniform noise amplitude")
    plt.savefig("LV_noise_and_extinction_grap" + str(round(time.time())) + ".png")


start_r22 = [2., 2.]
start_r11 = [1., 1.]
start_r33 = [3., 3.]
start_params1 = [[0.6666, -1.3333], [1., -1.]]
start_params2 = [[1.5, -0.5], [1.5, -1.5]]
# abgd_param_changes_quadgraph(start_r22, start_params1, 10)
# noise_and_extinction(start_r11, start_params1, 0, 6, runs=10, noise_step=0.2)
# noise_and_extinction(start_r22, start_params1, 0, 4, runs=100, noise_step=0.05)
# noise_and_extinction(start_r33, start_params1, 0, 6, runs=10, noise_step=0.2)

# noise_and_extinction(start_r11, start_params1, 0, 6, runs=10, noise_step=0.1)
# noise_and_extinction(start_r11, start_params1, 0, 5, runs=20, noise_step=0.1, noisepar2=0.1, noise_type="sin")
# noise_and_extinction(start_r11, start_params1, 1, 2.5, runs=20, noise_step=0.01, noisepar2=10, noise_type="sin")
# noise_and_extinction(start_r11, start_params1, 0, 5, runs=20, noise_step=0.1, noisepar2=20, noise_type="sin")

# noise_and_extinction(start_r11, start_params2, 0, 6, runs=100, noise_step=0.1)
# noise_and_extinction(start_r22, start_params2, 0, 6, runs=100, noise_step=0.1)
# noise_and_extinction(start_r33, start_params2, 0, 6, runs=100, noise_step=0.1)

# uniform_noise_graph(start_r22, start_params1, noise=3)
# # start_cond_changes_graph(start_r11, 10, start_params1)
# simulate_draw(start_r22, start_params1, 100, noise_par1=0)
xpoints, ypoints, ext = simulate_draw(start_r22, start_params1, 100, noise_par1=0.01, noise_par2=2*np.pi/1000, noise_type="sin")
# simulate_draw(start_r22, start_params1, 100, noise_par1=0.01, noise_par2=2*np.pi/10, noise_type="sin")
# simulate_draw(start_r22, start_params1, 100, noise_par1=0.01, noise_par2=2*np.pi, noise_type="sin")


# start_params3 = [[0.5, 0.01], [0.4, 0.02]]
# logistic_simulate_draw(start_r11, start_params3, [100,80], 100, noise_par1=100)
# print("minx =",np.min(xpoints), "minx =", np.min(ypoints))

plt.show()
