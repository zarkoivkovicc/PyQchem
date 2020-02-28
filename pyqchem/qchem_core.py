__author__ = 'Abel Carreras'

import os
from subprocess import Popen, PIPE
import numpy as np
import hashlib
import pickle
import warnings
from pyqchem.qc_input import QchemInput

__calculation_data_filename__ = 'calculation_data.pkl'
try:
    with open(__calculation_data_filename__, 'rb') as input:
        calculation_data = pickle.load(input)
        print('Loaded data from {}'.format(__calculation_data_filename__))
except IOError:
    calculation_data = {}


def redefine_calculation_data_filename(filename):
    global __calculation_data_filename__
    global calculation_data

    __calculation_data_filename__ = filename
    print('Set data file to {}'.format(__calculation_data_filename__))

    try:
        with open(__calculation_data_filename__, 'rb') as input:
            calculation_data = pickle.load(input)
            print('Loaded data from {}'.format(__calculation_data_filename__))
    except IOError:
        calculation_data = {}


# Check if calculation finished ok
def finish_ok(output):
    return output[-1000:].find('Thank you very much for using Q-Chem') != -1


# Layer of compatibility with old version
def create_qchem_input(*args, **kwargs):
    return QchemInput(*args, **kwargs)


def parse_output(get_output_function):
    """
    to be deprecated

    :param get_output_function:
    :return:
    """

    global calculation_data

    def func_wrapper(*args, **kwargs):
        parser = kwargs.pop('parser', None)
        parser_parameters = kwargs.pop('parser_parameters', {})
        store_output = kwargs.pop('store_output', None)

        force_recalculation = kwargs.pop('force_recalculation', False)

        if parser is not None:
            hash_p = (args[0], parser.__name__)
            if hash_p in calculation_data and not force_recalculation:
                print('already calculated. Skip')
                return calculation_data[hash_p]

        output, err = get_output_function(*args, **kwargs)

        if store_output is not None:
            with open('{}'.format(store_output), 'w') as f:
                f.write(output)

        if len(err) > 0:
            print(output[-800:])
            print(err)
            raise Exception('q-chem calculation finished with error')

        if parser is None:
            return output

        parsed_output = parser(output, **parser_parameters)

        calculation_data[hash_p] = parsed_output
        with open(__calculation_data_filename__, 'wb') as output:
            pickle.dump(calculation_data, output, pickle.HIGHEST_PROTOCOL)

        return parsed_output

    return func_wrapper


def local_run(input_file_name, work_dir, fchk_file, use_mpi=False, processors=1):
    """
    Run Q-Chem locally

    :param input_file_name: Q-Chem input file in plain text format
    :param work_dir:  Scratch directory where calculation run
    :param fchk_file: filename of fchk
    :param use_mpi: use mpi instead of openmp
    :return output, err: Q-Chem standard output and standard error
    """

    if not use_mpi:
        os.environ["QCTHREADS"] = "{}".format(processors)
        os.environ["OMP_NUM_THREADS"] = "{}".format(processors)
        os.environ["MKL_NUM_THREADS"] = "1"

    os.environ["GUIFILE"] = fchk_file
    qc_dir = os.environ['QC']
    binary = "{}/exe/qcprog.exe".format(qc_dir)
    # command = binary + ' {} {} '.format(flag, processors) + ' {} '.format(temp_file_name)
    command = binary + ' {} '.format(os.path.join(work_dir, input_file_name)) + ' {} '.format(work_dir)

    qchem_process = Popen(command, stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True, cwd=work_dir)
    (output, err) = qchem_process.communicate()
    qchem_process.wait()
    output = output.decode()
    err = err.decode()

    return output, err


def remote_run(input_file_name, work_dir, fchk_file, remote_params, use_mpi=False, processors=1):
    """
    Run Q-Chem remotely

    :param input_file: Q-Chem input file in plain text format
    :param work_dir:  Scratch directory where calculation run
    :param fchk_file: filename of fchk
    :param remote_params: connection parameters for paramiko
    :param use_mpi: use mpi instead of openmp
    :return output, err: Q-Chem standard output and standard error
    """
    import paramiko

    # get precommands
    commands = remote_params.pop('precommand', [])
    remote_scratch = remote_params.pop('remote_scratch', None)

    # Setup SSH connection
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(**remote_params)

    ssh.get_transport()
    sftp = ssh.open_sftp()
    print('connected to {}..'.format(remote_params['hostname']))

    # Define temp remote dir
    _, stdout, _ = ssh.exec_command('pwd', get_pty=True)

    if remote_scratch is None:
        remote_scratch = stdout.read().decode().strip('\n').strip('\r')

    remote_dir = '{}/temp_pyqchem_remote/'.format(remote_scratch)

    # Create temp directory in remote machine
    try:
        sftp.mkdir(remote_dir)
    except OSError:
        pass
    sftp.chdir(remote_dir)

    # Copy all files in local workdir to remote machine
    file_list = os.listdir(work_dir)
    for file in file_list:
        sftp.put(os.path.join(work_dir, file), '{}'.format(file))

    flag = '-np' if use_mpi else '-nt'

    # Define commands to run Q-Chem in remote machine
    commands += ['cd  {}'.format(remote_dir),  # go to remote work dir
                 'qchem {} {} {}'.format(flag, processors, input_file_name)]  # run qchem

    # Execute command in remote machine
    stdin, stdout, stderr = ssh.exec_command('bash -l -c "{}"'.format(';'.join(commands)), get_pty=True)

    # Reformat output/error files
    output = ''.join(stdout.readlines())
    error = ''.join(stderr.readlines())

    # get files and remove them from remote server
    for file in sftp.listdir():
        sftp.get(os.path.join(remote_dir, file), os.path.join(work_dir, file))
        sftp.remove(os.path.join(remote_dir, file))
    sftp.rmdir(remote_dir)

    sftp.close()
    ssh.close()

    # Rename fchk file to match expected name
    if input_file_name + '.fchk' in os.listdir(work_dir):
        os.rename(os.path.join(work_dir, input_file_name + '.fchk'), os.path.join(work_dir, fchk_file))

    return output, error


def store_calculation_data(input_qchem, keyword, data, status=True, protocol=pickle.HIGHEST_PROTOCOL):
    # input_qchem.__dict__.pop('_set_iter', None)
    # not store data if not finished OK
    if status is False:
        return

    calculation_data[(hash(input_qchem), keyword)] = data
    with open(__calculation_data_filename__, 'wb') as f:
        pickle.dump(calculation_data, f, protocol)


def retrieve_calculation_data(input_qchem, keyword):
    return calculation_data[(hash(input_qchem), keyword)] if (hash(input_qchem), keyword) in calculation_data else None


def get_output_from_qchem(input_qchem,
                          processors=1,
                          use_mpi=False,
                          scratch=None,
                          read_fchk=False,
                          parser=None,
                          parser_parameters=None,
                          force_recalculation=False,
                          fchk_only=False,
                          store_full_output=False,
                          remote=None,
                          strict_policy=False):
    """
    Runs qchem and returns the output in the following format:
    1) If read_fchk is requested:
        [output, error, parsed_fchk]
    2) If read_fchk is not requested:
        [output, error]

    Note: if parser is set then output contains a dictionary with the parsed info
          else output contains the q-chem output in plain text

    error: contains the standard error data from the calculation (if all OK, then should contain nothing)
    read_fchk: contains a dictionary with the parsed info inside fchk file.

    :param input_qchem:
    :param processors:
    :param use_mpi:
    :param scratch:
    :param read_fchk:
    :param parser:
    :param parser_parameters:
    :param force_recalculation:
    :param fchk_only:
    :param remote:
    :param strict_policy:
    :return output, error[, fchk_dict]:
    """
    from pyqchem.parsers.parser_fchk import parser_fchk

    # check gui > 2 if read_fchk
    if read_fchk:
        if input_qchem.gui is None or input_qchem.gui < 1:
            input_qchem.gui = 2

    if scratch is None:
        scratch = os.environ['QCSCRATCH']

    work_dir = '{}/qchem{}/'.format(scratch, os.getpid())

    try:
        os.mkdir(work_dir)
    except FileExistsError:
        pass

    # check scf_guess if guess
    if input_qchem.mo_coefficients is not None:
        guess = input_qchem.mo_coefficients
        # set guess in place
        mo_coeffa = np.array(guess['alpha'], dtype=np.float)
        l = len(mo_coeffa)
        if 'beta' in guess:
            mo_coeffb = np.array(guess['beta'], dtype=np.float)
        else:
            mo_coeffb = mo_coeffa

        mo_ene = np.zeros(l)

        guess_file = np.vstack([mo_coeffa, mo_ene, mo_coeffb, mo_ene]).flatten()
        with open(work_dir + '53.0', 'w') as f:
            guess_file.tofile(f, sep='')

    input_txt = input_qchem.get_txt()

    # check if parameters is None
    if parser_parameters is None:
        parser_parameters = {}

    # check if full output is stored
    # print('input:', input_qchem)
    output, err = calculation_data[(hash(input_qchem), 'fullout')] if (hash(input_qchem), 'fullout') in calculation_data else [None, None]
    # output, err = retrieve_calculation_data(input_qchem, 'fullout') if retrieve_calculation_data(input_qchem, 'fullout') is not None else [None, None]

    if not force_recalculation and not store_full_output:

        data_fchk = retrieve_calculation_data(input_qchem, 'fchk')

        if parser is not None:

            data = retrieve_calculation_data(hash(input_qchem), parser.__name__)

            if data is not None:
                if read_fchk is False:
                    return data, err
                elif data_fchk is not None:
                    return data, err, data_fchk

        else:
            if fchk_only and data_fchk is not None:
                return None, None, data_fchk

    fchk_filename = 'qchem_temp_{}.fchk'.format(os.getpid())
    temp_filename = 'qchem_temp_{}.inp'.format(os.getpid())

    qchem_input_file = open(os.path.join(work_dir, temp_filename), mode='w')
    qchem_input_file.write(input_txt)
    qchem_input_file.close()

    # Q-Chem calculation
    if output is None or force_recalculation is True:
        if remote is None:
            output, err = local_run(temp_filename, work_dir, fchk_filename, use_mpi=use_mpi, processors=processors)
        else:
            output, err = remote_run(temp_filename, work_dir, fchk_filename, remote, use_mpi=use_mpi, processors=processors)

    if finish_ok(output):
        finished_ok = True
    else:
        err += '\n'.join(output.split('\n')[-10:])
        finished_ok = False

    if store_full_output:
        store_calculation_data(input_qchem, 'fullout', [output, err], status=finished_ok)

    if parser is not None:
        output = parser(output, **parser_parameters)
        store_calculation_data(input_qchem, parser.__name__, output, status=finished_ok)

    if read_fchk:

        data_fchk = retrieve_calculation_data(input_qchem, 'fchk')
        if data_fchk is not None and not force_recalculation:
            return output, err, data_fchk

        if not os.path.isfile(os.path.join(work_dir, fchk_filename)):
            warnings.warn('fchk not found! Make sure the input generates it (gui 2)')
            return output, err, []

        with open(os.path.join(work_dir, fchk_filename)) as f:
            fchk_txt = f.read()
        os.remove(os.path.join(work_dir, fchk_filename))

        data_fchk = parser_fchk(fchk_txt)
        store_calculation_data(input_qchem, 'fchk', data_fchk, status=finished_ok)

        return output, err, data_fchk

    return output, err


def get_input_hash(data):
    return hashlib.md5(data.encode()).hexdigest()

