import numpy as np
import sys
from functools import reduce

class Simplex():


    def __init__(self, nv, nr, fv, cv, rs, aux_flag=False):
        self.c_v = self.get_c_vec(cv, nr, nv, fv, aux_flag)
        b_v = self.get_b_vec(rs)
        a = self.get_a_matrix(rs)
        base = self.make_base(rs, nr)
        free_base = self.free_variables(nv, nr, fv, a)
        auxiliary_base = np.identity(nr)

        #full tableau
        self.tableau = self.make_tableau(self.c_v, b_v, a, base, free_base, auxiliary_base, aux_flag)
        self.aux_tableau = self.make_tableau(self.get_c_vec(cv, nr, nv,fv, True), b_v, a, base, free_base, auxiliary_base, aux_flag=True)
        self.row_r = nr
        self.col_v = nv
        self.solution_vector = [0]*(nv+nr+1)
        self.base_vec = [-1]*nr
        self.status = ''

    '''

    Funcs for building the tableau matrix

    '''

    def get_c_vec(self, cv, nr, nc, fv, aux_flag):
        extra = len(fv) - reduce((lambda x, y: x + y), fv)
        #               cv = restr | nr = slack base | extra = fv | 1 = result                           aux identity
        return np.multiply(cv + [0]*(extra + nr + 1), -1) if aux_flag is False else [0]*(nc + extra + nr) + [1]*(nr) + [0]

    def get_b_vec(self, rs):
        return [[int(i[-1])] for i in rs if i != ['']]

    def get_a_matrix(self, raw_restr):
        return [[float(n) for n in r[:-2]] for r in raw_restr if r != ['']]

    def make_base(self, raw_restr, nr):
        base = np.identity(nr)
        for i in range(nr):
            if raw_restr[i][-2] == '>=':
                base[i][i] = -1
            elif raw_restr[i][-2] == '==':
                base[i][i] = 0
        np.vstack((np.zeros(nr), base))
        return base

    def free_variables(self, nv, nr, fv, a):
        fv_v = [[]]
        for i in range(len(fv)):
            if fv[i] == 0:
                new_vec = np.array(a)[:,i]
                if(fv_v == [[]]):
                    fv_v = np.array([np.multiply(new_vec, -1)])
                else:
                    fv_v = np.append(fv_v, [np.multiply(new_vec, -1)], axis=0)
        return fv_v

    def replace_c_vector(self):
        self.tableau[0] = self.c_v

    def zero_aux(self):
        for i in range(len(self.tableau)):
            self.tableau[i][-(self.row_r + 1):-1] = [0]*self.row_r

    def make_tableau(self, c, b, a, base, fvm, aux, aux_flag):
        if fvm != [[]]:
            t = np.concatenate((np.array(a), np.transpose(fvm)), 1)
        else:
            t = np.array(a)
        t = np.concatenate((np.array(t), base), 1)
        indexes = [i for i in range(len(b)) if b[i][0] < 0]
        if(aux_flag):
            if(indexes != []):
                for i in indexes:
                    b[i] = [b[i][0] * -1]
                    t[i] = np.multiply(t[i], -1)
            t = np.concatenate((np.array(t), aux), 1)
        t = np.hstack((t, b))
        t = np.vstack((c, t))
        return t

    '''
    Funcs for manipulating rows and cols
    '''

    def choose_element(self, tab, aux_flag=False):
        #get all elements smaller than zero
        candidate_list = [i for i in range (len(tab[0])-1) if tab[0][i] < 0]
        min_ratio = 100000
        min_ratio_coordinates = []
        #iterate through candidate columns
        if candidate_list == []:
            self.status = 'Ilimitada'
            return
        for i in candidate_list:
            for r in range(1, len(tab)):
                if tab[r][i] != 0 and tab[r][-1]/tab[r][i] < min_ratio and tab[r][-1]/tab[r][i] >= 0:
                    min_ratio = tab[r][-1]/tab[r][i]
                    min_ratio_coordinates = [r, i]
        #retorna a localização do pivô
        return min_ratio_coordinates

    def pivot(self, tab):
        try:
            p_r, p_c = self.choose_element(tab)
        except:
            self.status = 'Ilimitada'
            return
        self.base_vec[p_r - 1] = p_c
        #divide a linha do pivô pelo pivô
        tab[p_r] = np.true_divide(tab[p_r], tab[p_r][p_c])


        #zera todas as outras linhas
        for i in range (len(tab)):
            if i != p_r:
                tab[i] = [tab[i][j] - (tab[p_r][j] * tab[i][p_c]) for j in range (len(tab[i]))]
        return

    def has_negative(self, l):
        if list(filter(lambda x: x < 0 and np.abs(x) > 0.000001, l[:-1])) != []:
            return True
        return False

    '''
    Checking status
    '''


    def check_auxiliary(self):
        for i in range (1, self.row_r+1):
            self.aux_tableau[0] = self.aux_tableau[0] + np.multiply(self.aux_tableau[i], -1)

        while(self.has_negative(self.aux_tableau[0])):
            self.pivot(tab=self.aux_tableau)
        if np.abs(self.aux_tableau[0][-1]<0.000001):
            return 'Otima'
        else:
            return 'Inviavel'

    '''
    getters
    '''

    def get_optimal(self):
        return self.tableau[0][-1]

    def get_solution_vector(self):
        return [self.tableau[self.base_vec.index(i)+1][-1] if i in self.base_vec else 0 for i in range(len(self.tableau[0]))]

    '''
    run
    '''

    def run(self):
        self.status = self.check_auxiliary()
        if self.status == 'Inviavel':
            return self.status
        #self.replace_c_vector()
        #self.zero_aux()
        while(self.has_negative(self.tableau[0])):
            self.pivot(tab=self.tableau)
            if self.status == 'Ilimitada':
                return self.status
        return self.status

'''
I/O
'''

def read_args():
    if len(sys.argv) == 3:
        i_filename = sys.argv[1]
        o_filename = sys.argv[2]
    else:
        i_filename = input("Type the name of the file with the LP data: ")
        o_filename = input("Type the name of the file to write: ")
    return i_filename, o_filename

def read_vectors(filename):
    problem_data = open(filename, "r")
    num_var = int(problem_data.readline().strip('\n'))
    num_restrictions = int(problem_data.readline().strip('\n'))
    free_vars = [int(i) for i in problem_data.readline().strip('\n').split(' ')]
    c_vec = [int(i) for i in problem_data.readline().strip('\n').split(' ')]
    restrictions = [r.split(' ') for r in problem_data.read().split('\n')]
    return num_var, num_restrictions, free_vars, c_vec, restrictions

def output(filename, status, sol=None, obj=None, cert=None):
    ans = open(filename, "w")
    print("Status: ", status, file=ans)
    if obj is not None:
        print("Objetivo: \n", obj, file=ans)
    if sol is not None:
        print("Solucao: \n", sol, file=ans)
    if cert is not None:
        print("Certificado: \n", cert, file=ans)

if __name__ == "__main__":
    ifl, ofl = read_args()
    nv, nr, fv, cv, rs = read_vectors(ifl)

    s = Simplex(nv, nr, fv, cv, rs)
    status = s.run()
    if status == 'Inviavel':
        output(ofl, status)

    obj = s.get_optimal()
    sol = s.get_solution_vector()

    output(ofl, status, sol, obj)
