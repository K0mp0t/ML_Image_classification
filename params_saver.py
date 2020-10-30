import h5py


def save_params_to_file(parameters, h5_path):
    L = len(parameters)//2

    h5 = h5py.File(h5_path, 'w')

    for i in range(L):
        h5.create_dataset('W'+str(i+1), dtype=float, data=parameters['W'+str(i+1)])
        h5.create_dataset('b'+str(i+1), dtype=float, data=parameters['b'+str(i+1)])

    h5.close()


def read_params_from_file(h5_path):
    h5 = h5py.File(h5_path, 'r')
    parameters = {}

    for key in h5.keys():
        parameters[str(key)] = h5[str(key)]

    return parameters


