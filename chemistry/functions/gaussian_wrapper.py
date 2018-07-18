import os
import threading
import pathlib
from chemistry.functions.transformations import BaseFunction
import numpy as np


class GaussianException(Exception):
    def __init__(self):
        self.thread_id = threading.get_ident()


class GaussianWrapper:
    SCF_METHOD = 'scf'
    FORCE_METHOD = 'force'
    HESS_METHOD = 'freq'
    OPT_METHOD = 'FOpt'

    GAUSSIAN_HEADER_TEMPLATE = '%RWF={0}/rwf\n' \
                               '%Int={0}/int\n' \
                               '%D2E={0}/d2e\n' \
                               '%Scr={0}/skr\n' \
                               '%NoSave\n' \
                               '%chk={0}/chk\n' \
                               '%nproc={2}\n' \
                               '%mem={3}mb\n' \
                               '# B3lyp/3-21g nosym {1}\n' \
                               '\n' \
                               '\n' \
                               '{4} {5}\n'
#важно следить за мультиплетностью и зарядом
    MAGIC_CONSTANT = 1.88972585931612435672

    def __init__(self, n_proc, mem, charge, multiplicity):
        self.n_proc = n_proc
        self.mem = mem
        self.charge = charge
        self.multiplicity = multiplicity

    def __call__(self, charges, x):
        return self._run_gaussian_and_parse_values(charges, x, self.SCF_METHOD, [self._parse_value])[0]

    def value_grad(self, charges, x):
        return self._run_gaussian_and_parse_values(charges, x, self.FORCE_METHOD, [self._parse_value, self._parse_grad])

    def value_grad_hess(self, charges, x):
        return self._run_gaussian_and_parse_values(charges, x, self.HESS_METHOD,
                                                   [self._parse_value, self._parse_grad, self._parse_hess])

    def _run_gaussian_and_parse_values(self, charges, x, method, parsers):
        output_file = self._run_gaussian(charges, x, method)
        with output_file.open('r') as f:
            lines = iter(f.readlines())
            return tuple(map(lambda parser: parser(charges, lines), parsers))

    def _create_input_file(self, charges, x, method):
        folder = pathlib.Path('./tmp/{}/'.format(threading.get_ident()))
        folder.mkdir(parents=True, exist_ok=True)

        with (folder / 'input').open('w') as f:
            f.write(self.GAUSSIAN_HEADER_TEMPLATE.format(folder, method, self.n_proc, self.mem, self.charge,
                                                         self.multiplicity))
            for i, charge in enumerate(charges):
                f.write('{}\t{:.30f}\t{:.30f}\t{:.30f}\n'.format(charge, *x[i * 3: i * 3 + 3]))
            f.write('\n')

        return folder

    def _run_gaussian(self, charges, x, method):
        folder = self._create_input_file(charges, x, method)

        if os.system('GAUSS_SCRDIR={0} mg09D {0}/input {0}/output > /dev/null'.format(folder)):
            raise GaussianException()
        if os.system('formchk {} > /dev/null'.format(folder / 'chk.chk')):
            raise GaussianException()

        return folder / 'chk.fchk'

    def _parse_value(self, charges, line_iterator):
        while True:
            s = next(line_iterator)
            if s.startswith('Total Energy'):
                return float(s.split()[3])

    def _parse_grad(self, charges, line_iterator):
        grad_values = []
        while True:
            if next(line_iterator).startswith('Cartesian Gradient'):
                while len(grad_values) != 3 * len(charges):
                    grad_values.extend(map(float, next(line_iterator).split()))
                return np.array(grad_values) * self.MAGIC_CONSTANT

    def _parse_hess(self, charges, line_iterator):
        while True:
            if next(line_iterator).startswith('Cartesian Force Constants'):
                hess_values = []
                while len(hess_values) != 3 * len(charges) * (3 * len(charges) + 1) // 2:
                    s = next(line_iterator)
                    hess_values.extend(map(float, s.split()))

                hess_values_iter = iter(hess_values)

                hess = np.zeros((3 * len(charges), 3 * len(charges)))
                for i in range(3 * len(charges)):
                    for j in range(0, i + 1):
                        hess[i, j] = hess[j, i] = next(hess_values_iter)

                return hess * self.MAGIC_CONSTANT**2


class Molecule(BaseFunction):
    def __init__(self, charges, charge=0, multiplicity=1, n_proc=3, mem=750):
        super(Molecule, self).__init__(len(charges) * 3)
        self.charges = charges
        self.gaussian = GaussianWrapper(n_proc, mem, charge, multiplicity)

    def __call__(self, x):
        return self.gaussian(self.charges, x)

    def value_grad(self, x):
        return self.gaussian.value_grad(self.charges, x)

    def value_grad_hess(self, x):
        return self.gaussian.value_grad_hess(self.charges, x)


if __name__ == '__main__':
    import time
    import concurrent.futures

    N_ITERS = 100
    X = np.array([0.000000000, -0.859799324, 0.835503236,
                  0.000000000, -0.100462324, 1.431546236,
                  0.000000000, -1.619136324, 1.431546236])

    runner = Molecule([1, 1, 8], 1, 1000)

    def runner_fn():
        local_start_time = time.time()
        value, grad, hess = runner.value_grad_hess(X + np.random.randn(len(X)) * .01)
        print(grad.shape, hess.shape, time.time() - local_start_time)

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for i in range(N_ITERS):
            futures.append(executor.submit(runner_fn))
        for future in futures:
            future.result()
    print(time.time() - start_time)
