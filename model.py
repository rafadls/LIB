import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

def interpolate1d(x2predict, xtrain, ytrain):
    f = interpolate.interp1d(xtrain, ytrain, fill_value='extrapolate')
    return f(x2predict)
    #return np.array(np.interp(x2predict, xtrain, ytrain))

def q_conductividad(_t):
    _t += 273.15
    _datos = np.array([22.3, 26.3, 30, 33.8, 37.3]) * 1e-3
    _temp = np.array([250, 300, 350, 400, 450]) * 1.
    _k = interpolate1d(_t, _temp, _datos)
    np.place(_k, _k < 0, 0.0001)
    np.place(_k, np.isnan(_k), 0.0001)
    return _k

def q_reynolds(_v, _t, _D, _d):
    _visc = q_viscosidad(_t)
    _re = _d * _v * _D / _visc
    return _re

def q_viscosidad(_t):
    _t += 273.15
    _datos = np.array([159.6, 184.6, 208.2, 230.1, 250.7]) * 1e-7
    _temp = np.array([250, 300, 350, 400, 450])
    _visc = interpolate1d(_t, _temp, _datos)
    np.place(_visc, np.isnan(_visc), 0)
    return _visc

def q_paramdrag(_s):
    _b1 = np.array([0.039, 0.028, 0.027, 0.028, 0.005])
    _b2 = np.array([3.270, 2.416, 2.907, 2.974, 2.063])
    _x = np.array([0.10, 0.25, 0.50, 0.75, 1.00])
    _a1 = interpolate1d(_s, _x, _b1)
    _a2 = interpolate1d(_s, _x, _b2)
    if _s > _x[-1]:
        _a1 = _b1[-1]
        _a2 = _b2[-1]
    if _a1 < 0:
        _a1 = 0
    if _a2 < 0:
        _a2 = 0
    return _a1, _a2

def q_densidad(_t):
    _datos = np.array([1.293, 1.205, 1.127])
    _temp = np.array([0, 20, 40]) * 1.
    _d = interpolate1d(_t, _temp, _datos)
    np.place(_d, np.isnan(_d), 1.0)
    return _d

def q_cp(_t):
    _t += 273.15
    _datos = np.array([1.006, 1.007, 1.009, 1.014, 1.021]) * 1e3
    _temp = np.array([250, 300, 350, 400, 450]) * 1.
    _cp = interpolate1d(_t, _temp, _datos)
    np.place(_cp, np.isnan(_cp), 1.0)
    return _cp

class Model:
    def __init__(self, current=5, cellDiamater=26, separation=1, initFlow=100, \
                 initTemperature=26, col_fluido=2, col_celda=1, n_fluido=3, n_celda=0,\
                 nmax=10):
        """
		"""
        self.current = 1.0 * current                   # Corriente                   [A]
        self.cellDiameter = 1.0*cellDiamater           # Diametro Celda              [mm]
        self.separation = 1.0*separation               # Separacion                  [adimensional]
        self.initFlow = 1.0*initFlow                   # Flujo de entrada del fluido [CFM]
        self.initTemperature  = 1.0*initTemperature    # Temperatura incial          [Degree C]
        self.col_fluido = col_fluido
        self.col_celda = col_celda
        self.n_fluido = n_fluido
        self.n_celda = n_celda
        self.R = 32 * 1e-3      # Resistencia interna [Ohm]
        self.largo = 65 * 1e-3  # Largo celdas        [m]
        self.e = 15 * 1e-3      # espacio pared-celda [m]
        self.atmPressure = 0    # presion atmosferica [Pa]
        self.errmax = 1e-3      # error corte
        self.nmax = nmax        # Iteraciones fluodinamicas

        self.sp = dict()        # Variables calculadas
        self.vars_vec = dict()
        self.err_vec = dict()

        self.cellDiameter_m = None
        self.initFlow_m3_sg = None
        self.individual = dict()


    def start(self):
        # ------------------------- Inputs Conversion --------------------------------
        self.cellDiameter_m = self.cellDiameter / 1000  # Diametro Celda              [mm]->[m]
        self.initFlow_m3_sg = self.initFlow * 0.00047   # Flujo de entrada del fluido [CFM]->[m3/s]

        # --------------------- Fixed parameters ------------------------------
        self.sp['piQuarter'] = np.pi / 4
        self.sp['doubleE'] = 2 * self.e
        self.sp['maxValue'] = np.finfo(np.float_).max

        # -------------- Speedup cache variables
        self.sp['cellArea'] = (self.cellDiameter_m ** 2) * self.sp['piQuarter']
        self.sp['diamTimesZ'] = self.cellDiameter_m * self.largo  
        self.sp['superficialArea'] = np.pi * self.cellDiameter_m * self.largo                # Area de la celda
        self.sp['sPlusOne'] = self.separation + 1
        self.sp['controlVolArea'] = self.sp['sPlusOne'] * self.cellDiameter_m * self.largo      # Area volumen control eje z

        self.sp['qdot'] = (self.current**2) * self.R  # Flujo de Calor total corte
        self.sp['height'] = self.sp['doubleE'] + self.cellDiameter_m * \
                            (self.n_fluido + self.separation*self.n_celda)         # Altura del pack
        self.sp['entranceArea'] = self.sp['height'] * self.largo                       # Area de entrada pack
        self.sp['innerArg'] = self.initFlow_m3_sg/self.sp['entranceArea']
        self.sp['sTerm'] = self.separation / self.sp['sPlusOne']
        self.sp['heatPerArea'] = self.sp['qdot'] / self.sp['superficialArea']
        self.sp['normalizedArea'] = self.sp['superficialArea'] / self.sp['controlVolArea']

        # /********************** Initialization *****************************************/

        self.vars_vec['tf'] = self.initTemperature * np.ones(self.col_fluido)      # [Degree C]
        self.vars_vec['pf'] = self.atmPressure * np.ones(self.col_fluido)          # [Pa]
        self.vars_vec['vf'] = self.sp['innerArg'] * np.ones(self.col_fluido)       # [m/s]
        self.vars_vec['vmf'] = self.sp['innerArg'] * np.ones(self.col_fluido)      # [m/s]
        self.vars_vec['df'] = 0.0 * np.ones(self.col_fluido)                       # [kg/m3]
        self.vars_vec['df'][0] = q_densidad(self.initTemperature)                  # Condicion Borde
        self.vars_vec['tc'] = self.initTemperature * np.ones(self.col_celda)       # [Degree C]
        self.vars_vec['ff'] = 0.0 * np.ones(self.col_fluido)                       # [N]
        self.vars_vec['rem'] = 1.0 * np.ones(self.col_fluido)                      # [adimensional]
        self.vars_vec['prandtl'] = 0.0 * np.ones(self.col_fluido)                      # [adimensional]

        self.vars_vec['cdrag'] = 0.0 * np.ones(self.col_fluido)                      # [adimensional]
        self.vars_vec['frctionFactor'] = 0.0 * np.ones(self.col_fluido)                      # [adimensional]
        self.vars_vec['nusselt'] = 0.0 * np.ones(self.col_fluido)                      # [adimensional]

        a1, a2 = q_paramdrag(self.separation)
        a3 = 0.653
        self.sp['a'] = np.array([a1, a2, a3])
        self.sp['initVelocity'] = a2 * self.sp['innerArg']                        # Condicion Borde - Velocidad entrada[m/s]
        self.sp['dfMultiplicationTerm'] = self.cellDiameter_m * self.largo  * self.sp['initVelocity'] * self.vars_vec['df'][0]
        self.sp['m_punto'] = self.sp['sPlusOne'] * self.sp['dfMultiplicationTerm'] # Flujo masico [kg/s]
        self.sp['initialFFTerm'] = 0.5 * self.sp['dfMultiplicationTerm'] * self.sp['initVelocity']
        self.sp['fluidTempTerm'] = self.sp['qdot'] / self.sp['m_punto']

        self.sp['cp'] = self.sp['qdot'] / self.sp['m_punto']

        self.vars_vec['fluidK'] = q_conductividad(self.vars_vec['tf'][0]) * np.ones(self.col_fluido) # [W/m k]

        # /********************** Errores en columnas ************************************/
        self.err_vec['cellTempError'] = self.sp['maxValue'] * np.ones(self.col_celda)
        self.err_vec['TFError'] = self.sp['maxValue'] * np.ones(self.col_fluido)
        self.err_vec['TFError'][0] = 0.0
        self.err_vec['velocityError'] = self.sp['maxValue'] * np.ones(self.col_fluido)
        self.err_vec['pressureError'] = self.sp['maxValue'] * np.ones(self.col_celda)

    def evolve(self):
        for _ in range(self.nmax):
            #  This gets updated with the first value from the last attempt to converge

            #  **************************************** Calculo de la velocidad en 1 ***********************************
            actualDF = self.vars_vec['df'][0]
            normalizedDF = actualDF / 1.205
            # param cdrag_tree:     inputs(5) (reynolds, separation, index, normalizedArea, normalizedDensity) 
            cdrag = self.individual['computeDragCoefficient'](self.vars_vec['rem'][0],self.separation, 0, self.sp['normalizedArea'], normalizedDF) 
            initialFF = self.sp['initialFFTerm'] * cdrag
            self.vars_vec['ff'][0] = initialFF
            actualVF = self.sp['initVelocity'] - initialFF / self.sp['m_punto']
            self._setValue(self.vars_vec['vf'], 0, self.err_vec['velocityError'], actualVF)

            # /*************************************** Calculo de la presion en 1 **************************************/
            actualVMF = self.sp['sTerm'] * actualVF
            self.vars_vec['vmf'][0] = actualVMF
            actualRem = q_reynolds(actualVMF, self.vars_vec['tf'][0], self.cellDiameter_m , actualDF)
            cp = q_cp(self.vars_vec['tf'][0])

            self.vars_vec['prandtl'][0] = q_viscosidad(self.vars_vec['tf'][0]) * q_cp(self.vars_vec['tf'][0]) / q_conductividad(self.vars_vec['tf'][0])
            self.vars_vec['rem'][0] = actualRem
            # param ffriction_tree: inputs(5) (reynolds, separation, index, normalizedVelocity, normalizedDensity)
            frictionFactor = self.individual['computeFrictionFactor'](actualRem, self.separation, 0, actualVMF / self.sp['initVelocity'], normalizedDF)
            actualPF = self.vars_vec['pf'][1] + 0.5 * frictionFactor * actualDF * (actualVMF **2)
            self._setValue(self.vars_vec['pf'], 0, self.err_vec['pressureError'], actualPF)
            self.vars_vec['frctionFactor'][0] = frictionFactor

            # Tf(0), pf(end), Df(0), vmf(end) and rem(0) aren't modified in the loop
            # Loop iterating across columns (Note that, when col_fluido=2, it is just one iteration)

            for i in range(self.col_fluido-1):
                actualVF = self.vars_vec['vf'][i]
                actualVMF = self.sp['sTerm'] * actualVF
                self.vars_vec['vmf'][i] = actualVMF
                actualTF = self.vars_vec['tf'][i]
                actualDF = self.vars_vec['df'][i]
                i2 = i+1
                self.vars_vec['rem'][i2] = q_reynolds(actualVMF, actualTF, self.cellDiameter_m, actualDF)
                self.vars_vec['prandtl'][i2] = q_viscosidad(actualTF) * q_cp(actualTF) / q_conductividad(actualTF)
                actualRem = self.vars_vec['rem'][i]
                actualprandtl = self.vars_vec['prandtl'][i]

                # ***************************************** Calculo de la presion **************************************
                normalizedDF = actualDF / 1.205
                # param ffriction_tree: inputs(5) (reynolds, separation, index, normalizedVelocity, normalizedDensity)
                frictionFactor = self.individual['computeFrictionFactor'](actualRem, self.separation, 0, self.vars_vec['vmf'][i] / self.sp['initVelocity'], normalizedDF)
                self.vars_vec['frctionFactor'][i] = frictionFactor
                nextPF = self.vars_vec['pf'][i2] # nextPF -> pf en la col 2
                # nextPF, presion en la salida, actualPF, presion en la entrada
                actualPF = nextPF + 0.5 * frictionFactor * actualDF * (actualVMF ** 2)
                self._setValue(self.vars_vec['pf'], i, self.err_vec['pressureError'], actualPF)

                # ***************************************** Calculo de la velocidad ************************************
                # param cdrag_tree:     inputs(5) (reynolds, separation, index, normalizedArea, normalizedDensity) 
                cdrag = self.individual['computeDragCoefficient'](actualRem, self.separation, i, self.sp['normalizedArea'], normalizedDF)
                self.vars_vec['cdrag'][i] = cdrag
                actualVFSquared = (actualVF ** 2)
                nextFF = 0.5 * self.cellDiameter_m * self.largo  * actualDF * actualVFSquared * cdrag
                self.vars_vec['ff'][i2] = nextFF

                # Conservacion de momento (arreglada, antes tenia velocidades al cuadrado)
                nextVF = ((self.sp['controlVolArea'] * (actualPF - nextPF) - nextFF) / self.sp['m_punto']) + actualVF
                # actualPF-nextPF ya que el delta de la memoria es P(i)-P(i+1)

                nextVFSquared = (nextVF ** 2)
                self._setValue(self.vars_vec['vf'], i2, self.err_vec['velocityError'], nextVF)

                # ********************************** Calculo de la temperatura fluido **********************************
                cp = q_cp(actualTF)
                # Conservacion de energia
                nextTF = actualTF + (self.sp['fluidTempTerm'] - 0.5 * (nextVFSquared - actualVFSquared)) / cp
                '''
                print('fluidTempTerm: ' + str(self.sp['fluidTempTerm']))
                print('cp: ' + str(cp))
                print('qdot: ' + str(self.sp['qdot']))
                print('m_punto: ' + str(self.sp['m_punto']))
                print()
                '''

                self._setValue(self.vars_vec['tf'], i2, self.err_vec['TFError'], nextTF)

                # ***************************************** Calculo de la densidad *************************************
                self.vars_vec['df'][i2] = q_densidad(nextTF)

                # ********************************** Calculo de temperatura de celda ************************************
                # param nnusselt_tree:  inputs(4) (reynolds, prandtl, separation, index)
                nu = self.individual['computeNusseltNumber'](actualRem, actualprandtl, self.separation, i) 
                self.vars_vec['nusselt'][i] = nu
                iniFluidK = q_conductividad(actualTF)
                self.vars_vec['fluidK'][i] = iniFluidK
                h = nu * iniFluidK / self.cellDiameter_m # [W m^-2 K^-1]
                # Transferencia de energia
                self._setValue(self.vars_vec['tc'], i, self.err_vec['cellTempError'], self.sp['heatPerArea'] / h + (actualTF + nextTF) / 2)

            # If convergence, stop iterating
            flag1 = self.err_vec['cellTempError'].max() <= self.errmax
            flag2 = self.err_vec['TFError'].max() <= self.errmax
            flag3 = self.err_vec['pressureError'].max() <= self.errmax
            flag4 = self.err_vec['velocityError'].max() <= self.errmax
            flag12 = np.logical_and(flag1, flag2)
            flag34 = np.logical_and(flag3, flag4)
            if np.logical_and(flag12, flag34):
                break

        # Al final necesitamos guardar: velocidad a la salida (1), presion a la entrada (0), temperatura central (0)
        results_dict = {}
        results_dict['cdrag'] = self.vars_vec['cdrag'][:-1] 
        results_dict['frctionFactor'] = self.vars_vec['frctionFactor'][:-1] 
        results_dict['nusselt'] = self.vars_vec['nusselt'][:-1]
        results_dict['vf'] = self.vars_vec['vf']
        results_dict['pf'] = self.vars_vec['pf']
        results_dict['tc'] = self.vars_vec['tc']
        results_dict['tf'] = self.vars_vec['tf']
        return results_dict

    def reset(self):
        self.__init__()

    def _setValue(self, vector, i, errorArray, value):
        errorArray[i] = vector[i]
        vector[i] = value
        errorArray[i] = np.abs((errorArray[i] - value) / value)

    def load_individual(self, cdrag_tree, ffriction_tree, nnusselt_tree):
        '''
        :param cdrag_tree:     inputs(5) (reynolds, separation, index, normalizedArea, normalizedDensity) 
        :param ffriction_tree: inputs(5) (reynolds, separation, index, normalizedVelocity, normalizedDensity)
        :param nnusselt_tree:  inputs(4) (reynolds, prandtl, separation, index)
        '''
        self.individual['computeDragCoefficient'] = cdrag_tree
        self.individual['computeFrictionFactor']  = ffriction_tree
        self.individual['computeNusseltNumber']   = nnusselt_tree


def pow2(x):
    return x**2