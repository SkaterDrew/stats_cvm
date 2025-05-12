import numpy as np
import matplotlib.pyplot as plt
from .variable import form, Variable

class Regression:
    def __init__(self, X, Y, coeff_linearise=False, decimales=3, decimales_pourcentages=1):
        self.X = X if isinstance(X, Variable) else Variable(X, type_var='c')
        self.Y = Y if isinstance(Y, Variable) else Variable(Y, type_var='c')

        valide = self.X.quantitative and self.Y.quantitative

        self.lineaire = self.regression_lineaire(decimales) if valide else [None, None, None]
        self.exp = self.regression_exp(decimales, coeff_linearise) if valide else [None, None, None]
        self.log = self.regression_log(decimales, coeff_linearise) if valide else [None, None, None]

        r_lin, r_exp, r_log = self.lineaire[0], self.exp[0], self.log[0]
        eqn_lin, eqn_exp, eqn_log = self.lineaire[1], self.exp[1], self.log[1]
        output = [
            "Équations et coefficients de détermination:",
            f"Linéaire: r^2 = {form(r_lin, decimales_pourcentages, '%')} --> {eqn_lin}",
            f"Exponentielle: r^2 = {form(r_exp, decimales_pourcentages, '%')} --> {eqn_exp}",
            f"Logarithmique: r^2 = {form(r_log, decimales_pourcentages, '%')} --> {eqn_log}"
        ] if valide else ["Impossible d'effectuer la régression, puisque les deux variables doivent être quantitatives"]

        self.infos = '\n'.join(output)
        print(self.infos)

        self.lin_f, self.exp_f, self.log_f = self.lineaire[2], self.exp[2], self.log[2]


    @classmethod
    def r(cls, X, Y):
        return np.corrcoef(X.data, Y.data)[0][1]


    @classmethod
    def droite_reg(cls, X, Y):
        m_x = X.moyenne
        s_x = X.ecart_type
        m_y = Y.moyenne
        s_y = Y.ecart_type
        a = cls.r(X, Y) * s_y / s_x
        b = m_y - a * m_x
        return a, b
    

    def regression_lineaire(self, decimales=3):
        r2 = Regression.r(self.X, self.Y) ** 2
        a, b = Regression.droite_reg(self.X, self.Y)
        eqn = f"y = {form(a, decimales)}x + {form(b, decimales)}"
        def f(arg):
            return a * arg + b
        
        return r2, eqn, f


    def regression_exp(self, decimales=3, linearise=False):
        x, y = self.X.data, self.Y.data
        a,b = Regression.droite_reg(self.X, self.Y.log())

        if linearise:
            r2 = Regression.r(self.X, self.Y.log()) ** 2
        else:
            y_transforme = np.exp(a * x + b)
            err = np.sum((y - y_transforme)**2)
            r2 = 1 - err / ((self.Y.n - 1) * self.Y.variance)

        eqn = f"y = {form(np.exp(b), decimales)} e^({form(a, decimales)}*x)"
        def f(arg):
            return np.exp(a * arg + b)
        
        return r2, eqn, f


    def regression_log(self, decimales=3, linearise=False):
        x, y = self.X.data, self.Y.data
        a,b = Regression.droite_reg(self.X.log(), self.Y)
        
        if linearise:
            r2 = Regression.r(self.X.log(), self.Y) ** 2
        else:
            y_transforme = a * np.log(x) + b
            err = np.sum((y - y_transforme)**2)
            r2 = 1 - err / ((self.Y.n - 1) * self.Y.variance)

        eqn = f"y = {form(a, decimales)} * log(x) + {form(b, decimales)}"
        def f(arg):
            return a * np.log(arg) + b
        
        return r2, eqn, f
    

    def nuage(self, decimales_pourcentages=1):
        x, y = self.X.data, self.Y.data
        plt.scatter(x, y, label='Données', color='blue', alpha=0.6)

        x_lin = np.linspace(self.X.min, self.X.max, 1000)
        y_lin = self.lin_f(x_lin)
        y_exp = self.exp_f(x_lin)
        y_log = self.log_f(x_lin)
        
        plt.plot(x_lin, y_lin, label=f'Régression Linéaire (r^2 = {form(self.lineaire[0], decimales_pourcentages, '%')})', color='red', linestyle='-')
        plt.plot(x_lin, y_exp, label=f'Régression Exponentielle (r^2 = {form(self.exp[0], decimales_pourcentages, '%')})', color='green', linestyle='-')
        plt.plot(x_lin, y_log, label=f'Régression Logarithmique (r^2 = {form(self.log[0], decimales_pourcentages, '%')})', color='orange', linestyle='-')

        # Add labels, legend, and title
        plt.xlabel(self.X.nom_court)
        plt.ylabel(self.Y.nom_court)
        plt.title(f"Distribution de {self.Y.nom_complet} selon {self.X.nom_complet} de {self.X.n} {self.X.unites_stat_complet}{', ' if self.X.lieu else ''}{self.X.lieu}{', ' if self.X.date else ''}{self.X.date}")
        plt.legend()

# FIN