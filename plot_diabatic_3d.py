import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Plot 3D data')
parser.add_argument('filename', metavar='filename', type=str,
                    help='filename for input')

parser.add_argument('--output_folder', metavar='distance', type=str, default='3d_plot/',
                    help='folder to store PDF plots')

parser.add_argument('--show_plots', action='store_true',
                   help='show plots')

parser.add_argument('--full', action='store_true',
                   help='apply periodicity')

parser.add_argument('--interpolate', action='store_true',
                   help='interpolate missing points')


args = parser.parse_args()


# Simple order
# set higher dipole moment states first if energy gap is lower than eps_energy
def get_order_states_list(states, eps_moment=0.1, eps_energy=0.05):

    import itertools
    order = np.arange(len(states))

    for subset in itertools.combinations(range(len(states)), 2):
        i, j = subset

        if np.abs(states[i]['total energy'] - states[j]['total energy']) < eps_energy:
            tmi = np.linalg.norm(states[i]['transition moment'])
            tmj = np.linalg.norm(states[j]['transition moment'])
            if tmi - tmj < eps_moment:
                order[i], order[j] = order[j], order[i]

    return order


def correct_order_list(list, order):

    if np.array(order).shape != np.array(list).T.shape:
        print(np.array(order).shape, np.array(list).T.shape)
        raise Exception('Error in correcting order (not same shape)')

    alist = np.array(list)

    ordered_list = []
    for l, o in zip(alist.T, order):
        ordered_list.append(l[o])

    return np.array(ordered_list).T.tolist()


def interpolate_data(points, data , y_range, z_range):
    from scipy.interpolate import griddata

    Y, Z = np.meshgrid(y_range, z_range)

    grid_z2 = griddata(points, data, (Y, Z), method='cubic')

    return grid_z2.T.flatten()


def triplot(data1, data2, label1, label2, y_range, z_range, wireframe=False, pdf=None,
            zlabel='Energy [eV]', zlevels=np.arange(-0.2, 0.2 + 0.025, 0.025), show_plot=True):

    # from matplotlib.colors import LinearSegmentedColormap
    # from matplotlib.colors import BoundaryNorm
    # from matplotlib.ticker import MaxNLocator

    cmap = plt.get_cmap('PiYG')

    Y, Z = np.meshgrid(z_range, y_range)

    plt.figure(1)

    plt.title(label1)
    CS = plt.contourf(Y, Z, np.array(data1).reshape(len(y_range), len(z_range)), levels=zlevels, cmap=cmap)
    CS2 = plt.contour(CS, levels=CS.levels[::1], colors='black')
    plt.clabel(CS2, inline=1, fontsize=10)
    plt.xlabel('distance Z [Å]')
    plt.ylabel('distance Y [Å]')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    cbar = plt.figure(1).colorbar(CS)

    cbar.ax.set_ylabel(zlabel)

    if pdf is not None:
        pdf.savefig()

    plt.figure(2)

    plt.title(label2)
    CS = plt.contourf(Y, Z, np.array(data2).reshape(len(y_range), len(z_range)), levels=zlevels, cmap=cmap)
    CS2 = plt.contour(CS, levels=CS.levels[::1], colors='black')
    plt.clabel(CS2, inline=1, fontsize=10)
    plt.xlabel('distance Z [Å]')
    plt.ylabel('distance Y [Å]')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    cbar = plt.figure(2).colorbar(CS)
    cbar.ax.set_ylabel(zlabel)

    if pdf is not None:
        pdf.savefig()

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    if wireframe:
        ax.plot_wireframe(Y, Z, np.array(data1).reshape(len(y_range), len(z_range)), color='b')
        ax.plot_wireframe(Y, Z, np.array(data2).reshape(len(y_range), len(z_range)), color='r')
    else:
        ax.plot_surface(Y, Z, np.array(data1).reshape(len(y_range), len(z_range)), color='b')
        ax.plot_surface(Y, Z, np.array(data2).reshape(len(y_range), len(z_range)), color='r')

    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    ax.legend([fake2Dline, fake2Dline2], [label1, label2], numpoints=1)
    ax.set_xlabel('distance Z [Å]')
    ax.set_ylabel('distance Y [Å]')
    ax.set_zlabel(zlabel)

    if pdf is not None:
        pdf.savefig(fig)

    if show_plot:
        plt.show()


def biplot2(data1, data2, label1, label2, y_range, z_range, pdf=None,
            zlabel='Energy [eV]', zrange=(-0.2, 0.2), zp=0.025, show_plot=True):

    from scipy import interpolate

    Y, Z = np.meshgrid(z_range, y_range)

    print('----------------')
    print(Y.shape)
    print(len(Y), len(data1))

    #f1 = interpolate.interp2d(Y, Z, data1, kind='cubic')
    f1 = interpolate.interp2d(Y, Z, data1, kind='linear')
    f2 = interpolate.interp2d(Y, Z, data2, kind='linear')

    x = np.arange(-4, 4, zp)

    plt.figure(3)

    plt.xlim([-4, 4])
    plt.ylim([zrange[0], zrange[1]])
    plt.xlabel('distance Z [Å]')
    plt.ylabel(zlabel)

    plt.plot(x, f1(x, 0), label=label1)
    plt.plot(x, f2(x, 0), label=label2)
    plt.legend()

    if pdf is not None:
        pdf.savefig()

    if show_plot:
        plt.show()


def biplot(data1, data2, label1, label2, y_range, z_range, pdf=None,
            zlabel='Energy [eV]', zrange=(-0.2, 0.2), title=None, show_plot=True):

    data1 = np.array(data1).reshape([len(y_range), len(z_range)])[len(z_range)//2]
    data2 = np.array(data2).reshape([len(y_range), len(z_range)])[len(z_range)//2]

    plt.title(title)
    plt.xlim([-4, 4])
    if zrange is not None:
        plt.ylim([zrange[0], zrange[1]])
    plt.xlabel('distance Z [Å]')
    plt.ylabel(zlabel)

    plt.plot(z_range, data1, label=label1)
    plt.plot(z_range, data2, label=label2)
    plt.legend()

    if pdf is not None:
        pdf.savefig()

    if show_plot:
        plt.show()


#############################
folder = args.output_folder
#############################

with open(args.filename, 'rb') as input:
    calculation_data = pickle.load(input)
    print('Loaded data from: {}'.format(args.filename))

do_full = args.full
interpolate = args.interpolate

for slide_y in calculation_data['range_y']:
    for slide_z in calculation_data['range_z']:
        print(slide_y, slide_z)
        if '{}_{}'.format(slide_y, slide_z) in calculation_data:
            print(calculation_data['{}_{}'.format(slide_y, slide_z)])
            data_i = calculation_data['{}_{}'.format(slide_y, slide_z)]
            print(data_i['states_info'])
            if do_full:

                calculation_data.update({'{}_{}'.format(-slide_y, slide_z): data_i,
                                         '{}_{}'.format(slide_y, -slide_z): data_i,
                                         '{}_{}'.format(-slide_y, -slide_z): data_i})


if do_full:
    y_range = np.unique((-np.array(calculation_data['range_y'])).tolist() + calculation_data['range_y'])
    z_range = np.unique((-np.array(calculation_data['range_z'])).tolist() + calculation_data['range_z'])
else:
    y_range = calculation_data['range_y']
    z_range = calculation_data['range_z']

points = []
total_data = []
full_range = []
i = 0
for slide_y in y_range:
    for slide_z in z_range:
        print('{}_{}'.format(slide_y, slide_z))
        if '{}_{}'.format(slide_y, slide_z) in calculation_data:
            data = calculation_data['{}_{}'.format(slide_y, slide_z)]
            total_data.append(data)
            full_range.append(i)
            points.append([slide_y, slide_z])
            i += 1

states_orders = [get_order_states_list(data['states_info']) for data in total_data]

########################  W_DC  ######################

data_1 = []
data_2 = []
data_3 = []
data_4 = []

for diab, coeff in [[data['diabatic_energies'], data['coefficients']] for data in total_data]:

    factor = coeff['S_01'][0] * coeff['S_10'][0]
    data_1.append(diab['V_DC'] * np.sign(factor))

    factor = coeff['S_01'][1] * coeff['S_10'][1]
    data_2.append(diab['V_DC'] * np.sign(factor))

    factor = coeff['S_01'][2] * coeff['S_10'][2]
    data_3.append(diab['V_DC'] * np.sign(factor))

    factor = coeff['S_01'][3] * coeff['S_10'][3]
    data_4.append(diab['V_DC'] * np.sign(factor))

data_1 = [data['diabatic_contributions']['W_DC'][0] for data in total_data]
data_2 = [data['diabatic_contributions']['W_DC'][1] for data in total_data]
data_3 = [data['diabatic_contributions']['W_DC'][2] for data in total_data]
data_4 = [data['diabatic_contributions']['W_DC'][3] for data in total_data]

data_1, data_2, data_3, data_4 = correct_order_list([data_1, data_2, data_3, data_4], states_orders)

if interpolate:
    data_1 = interpolate_data(points, data_1, y_range, z_range)
    data_2 = interpolate_data(points, data_2, y_range, z_range)

wdc1 = np.array(data_1)
wdc2 = np.array(data_2)

with PdfPages(folder + 'W_DC.pdf') as pdf:
    triplot(data_1, data_2, 'W_DC_1', 'W_DC_2', y_range, z_range, pdf=pdf, wireframe=True, show_plot=args.show_plots)
    biplot(data_1, data_2, 'W_DC_1', 'W_DC_2', y_range, z_range, show_plot=args.show_plots)


########################  W_CT  ######################

data_1 = [data['diabatic_contributions']['W_CT'][0] for data in total_data]
data_2 = [data['diabatic_contributions']['W_CT'][1] for data in total_data]
data_3 = [data['diabatic_contributions']['W_CT'][2] for data in total_data]
data_4 = [data['diabatic_contributions']['W_CT'][3] for data in total_data]

data_1, data_2, data_3, data_4 = correct_order_list([data_1, data_2, data_3, data_4], states_orders)

if interpolate:
    data_1 = interpolate_data(points, data_1, y_range, z_range)
    data_2 = interpolate_data(points, data_2, y_range, z_range)

wct1 = np.array(data_1)
wct2 = np.array(data_2)

with PdfPages(folder + 'W_CT.pdf') as pdf:
    triplot(data_1, data_2, 'W_CT_1', 'W_CT_2', y_range, z_range, pdf=pdf, wireframe=True, show_plot=args.show_plots)
    biplot(data_1, data_2, 'W_CT_1', 'W_CT_2', y_range, z_range, show_plot=args.show_plots)



########################  W_e  ######################


data_1 = [data['diabatic_contributions']['W_e'][0] for data in total_data]
data_2 = [data['diabatic_contributions']['W_e'][1] for data in total_data]
data_3 = [data['diabatic_contributions']['W_e'][2] for data in total_data]
data_4 = [data['diabatic_contributions']['W_e'][3] for data in total_data]

data_1, data_2, data_3, data_4 = correct_order_list([data_1, data_2, data_3, data_4], states_orders)

if interpolate:
    data_1 = interpolate_data(points, data_1, y_range, z_range)
    data_2 = interpolate_data(points, data_2, y_range, z_range)

we1 = np.array(data_1)
we2 = np.array(data_2)

with PdfPages(folder + 'W_e.pdf') as pdf:
    triplot(data_1, data_2, 'W_e_1', 'W_e_2', y_range, z_range, pdf=pdf, wireframe=True, show_plot=args.show_plots)
    biplot(data_1, data_2, 'W_e_1', 'W_e_2', y_range, z_range, show_plot=args.show_plots)


#######################  W_h  ######################


data_1 = [data['diabatic_contributions']['W_h'][0] for data in total_data]
data_2 = [data['diabatic_contributions']['W_h'][1] for data in total_data]
data_3 = [data['diabatic_contributions']['W_h'][2] for data in total_data]
data_4 = [data['diabatic_contributions']['W_h'][3] for data in total_data]

data_1, data_2, data_3, data_4 = correct_order_list([data_1, data_2, data_3, data_4], states_orders)

if interpolate:
    data_1 = interpolate_data(points, data_1, y_range, z_range)
    data_2 = interpolate_data(points, data_2, y_range, z_range)

wh1 = np.array(data_1)
wh2 = np.array(data_2)

with PdfPages(folder + 'W_h.pdf') as pdf:
    triplot(data_1, data_2, 'W_h_1', 'W_h_2', y_range, z_range, pdf=pdf, wireframe=True, show_plot=args.show_plots)
    biplot(data_1, data_2, 'W_h_1', 'W_h_2', y_range, z_range, show_plot=args.show_plots)

#######################  diabatic_energies  ######################

data_1 = [data['diabatic_energies']['E_LE'][0] for data in total_data]
data_2 = [data['diabatic_energies']['E_CT'][0] for data in total_data]

#data_1, data_2 = correct_order_list([data_1, data_2], states_orders)

if interpolate:
    data_1 = interpolate_data(points, data_1, y_range, z_range)
    data_2 = interpolate_data(points, data_2, y_range, z_range)

e_le = np.array(data_1)
e_ct = np.array(data_2)

with PdfPages(folder + 'diabatic_energies.pdf') as pdf:
    triplot(data_1, data_2, 'E_LE', 'E_CT', y_range, z_range, pdf=pdf, wireframe=True, show_plot=args.show_plots, zlevels=None)
    biplot(data_1, data_2, 'E_LE', 'E_CT', y_range, z_range, show_plot=args.show_plots, zrange=None)


#######################  lambda  ######################

data_1 = [data['lambda'][0] for data in total_data]
data_2 = [data['lambda'][1] for data in total_data]
data_3 = [data['lambda'][2] for data in total_data]
data_4 = [data['lambda'][3] for data in total_data]

data_1, data_2, data_3, data_4 = correct_order_list([data_1, data_2, data_3, data_4], states_orders)

if interpolate:
    data_1 = interpolate_data(points, data_1, y_range, z_range)
    data_2 = interpolate_data(points, data_2, y_range, z_range)

l1 = np.array(data_1)
l2 = np.array(data_2)

with PdfPages(folder + 'lambda.pdf') as pdf:
    triplot(data_1, data_2, 'lambda 1', 'lambda 2', y_range, z_range, pdf=pdf, wireframe=True, show_plot=args.show_plots)
    biplot(data_1, data_2, 'lambda 1', 'lambda 2', y_range, z_range, show_plot=args.show_plots)

##########################  OMEGA h  #######################

data_1 = 2 * l1 * np.sqrt(1 - l1**2) * np.array(wh1)
data_2 = 2 * l2 * np.sqrt(1 - l2**2) * np.array(wh2)


with PdfPages(folder + 'omega_h.pdf') as pdf:
    triplot(data_1, data_2, 'omega_h 1', 'omaga_h 2', y_range, z_range, wireframe=True, pdf=pdf, show_plot=args.show_plots)
    biplot(data_1, data_2, 'state 1', 'state 2', y_range, z_range, pdf=pdf, show_plot=args.show_plots, title='omega_h')

##########################  OMEGA e  #######################

data_1 = 2 * l1 * np.sqrt(1 - l1**2) * np.array(we1)
data_2 = 2 * l2 * np.sqrt(1 - l2**2) * np.array(we2)


with PdfPages(folder + 'omega_e.pdf') as pdf:
    triplot(data_1, data_2, 'omega_e 1', 'omaga_e 2', y_range, z_range, wireframe=True, pdf=pdf, show_plot=args.show_plots)
    biplot(data_1, data_2, 'state 1', 'state 2', y_range, z_range, pdf=pdf, show_plot=args.show_plots, title='omega_e')


##########################  SUPEREXCHANGE(E1) #######################

data_1 = 2 * l1 * np.sqrt(1 - l1**2) * (np.array(we1) + np.array(wh1))
data_2 = 2 * l2 * np.sqrt(1 - l2**2) * (np.array(we2) + np.array(wh2))


with PdfPages(folder + 'E1(superexchange).pdf') as pdf:
    triplot(data_1, data_2, 'omega: state 1', 'omega: state 2', y_range, z_range, wireframe=True, pdf=pdf, show_plot=args.show_plots)
    biplot(data_1, data_2, 'state 1', 'state 2', y_range, z_range, pdf=pdf, show_plot=args.show_plots, title='superexchange')

e_11 = np.array(data_1)
e_12 = np.array(data_2)


#######################  diabatic_energies  ######################

data_1 = l1**2 * (e_ct - e_le + wct1 - wdc1)
data_2 = l2**2 * (e_ct - e_le + wct2 - wdc2)


with PdfPages(folder + 'E2.pdf') as pdf:
    triplot(data_1, data_2, 'E2-1', 'E2-2', y_range, z_range, wireframe=True, pdf=pdf, show_plot=args.show_plots)
    biplot(data_1, data_2, 'E2-1', 'E2-2', y_range, z_range, show_plot=args.show_plots)

e_21 = np.array(data_1)
e_22 = np.array(data_2)

#######################  adiabatic_energies calculated ######################

e1 = e_le + wdc1 + e_11 + e_21
e2 = e_le + wdc2 + e_12 + e_22


with PdfPages(folder + 'adiabatic_energies.pdf') as pdf:
    triplot(e1, e2, 'e1', 'e2', y_range, z_range, pdf=pdf, wireframe=True, show_plot=args.show_plots, zlevels=None)
    biplot(e1, e2, 'e1', 'e2', y_range, z_range, show_plot=args.show_plots, zrange=None)


#######################  adiabatic_energies extracted ######################

data_1 = [data['adiabatic_energies']['E_1'] for data in total_data]
data_2 = [data['adiabatic_energies']['E_2'] for data in total_data]
data_3 = [data['adiabatic_energies']['E_3'] for data in total_data]
data_4 = [data['adiabatic_energies']['E_4'] for data in total_data]

data_1, data_2, data_3, data_4 = correct_order_list([data_1, data_2, data_3, data_4], states_orders)

if interpolate:
    data_1 = interpolate_data(points, data_1, y_range, z_range)
    data_2 = interpolate_data(points, data_2, y_range, z_range)

with PdfPages(folder + 'adiabatic_energies.pdf') as pdf:
    triplot(data_1, data_2, 'E_1', 'E_2', y_range, z_range, pdf=pdf, wireframe=True, show_plot=args.show_plots, zlevels=None)
    biplot(data_1, data_2, 'E_1', 'E_2', y_range, z_range, show_plot=args.show_plots, zrange=None)


#######################  difference test ######################

with PdfPages(folder + 'adiabatic_energies_diff.pdf') as pdf:
    triplot(data_1-e1, data_2-e2, 'diff e1', 'diff e2', y_range, z_range, pdf=pdf, wireframe=True, show_plot=args.show_plots, zlevels=None)
    biplot(data_1-e1, data_2-e2, 'diff e1', 'diff e2', y_range, z_range, show_plot=args.show_plots,
           zrange=None,  title='Adiabatic energy differences (original - calculated)')

