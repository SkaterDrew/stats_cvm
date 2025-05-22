import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2
from .variable import Variable, form
from .regression import Regression
from typing import Literal, TypedDict, Optional, Self

class InfoPop(TypedDict):
    taille_population: Optional[int]  # int or None
    est_population: bool
    unites_stat_complet: str
    unites_stat: str
    source: str
    lieu: str
    date: str

class InfoVar(TypedDict):
    type_var: Literal['n', 'o', 'd', 'c', '']
    nom_complet: str
    nom_court: str
    legende: list
    unite_mesure: str

class Donnees:
    """
    Classe Donnees pour la gestion des données statistiques.

    Attributs:
        - data (pd.DataFrame): Le DataFrame contenant les données.
        - info_pop (InfoPop): Informations sur la population
            - taille_population (int): Taille de la population.
            - est_population (bool): Indique si c'est une population.
            - unites_stat_complet (str): Unités statistiques complètes.
            - unites_stat (str): Unités statistiques.
            - source (str): Source des données.
            - lieu (str): Lieu des données.
        - variables (list[Variable]): Liste d'objets Variable représentant les variables dans le DataFrame.
    """
    def __init__(self, df: pd.DataFrame, info_pop: InfoPop | None = None) -> None:
        """
        Initialise la classe Donnees avec un DataFrame et des informations sur la population.
        Args:
            df (pd.DataFrame): Le DataFrame contenant les données.
            info_pop (InfoPop, optional): Informations sur la population. Si None, des valeurs par défaut sont utilisées.
                taille_population (int): Taille de la population.
                est_population (bool): Indique si c'est une population.
                unites_stat_complet (str): Unités statistiques complètes.
                unites_stat (str): Unités statistiques.
                source (str): Source des données.
                lieu (str): Lieu des données.
                date (str): Date des données.
        """
        self.data = df.copy()
        if info_pop is not None:
            self.info_pop = info_pop
        else:
            self.info_pop = {
                'taille_population': None,
                'est_population': False,
                'unites_stat_complet': 'unités statistiques',
                'unites_stat': 'unités statistiques',
                'source': '',
                'lieu': '',
                'date': '',
            }
        
        self.taille_population, self.est_population, self.unites_stat_complet, self.unites_stat, self.source, self.lieu, self.date = self.info_pop.values()
        self.variables: list[Variable] = [Variable(df, colonne=nom).definir(info_pop=self.info_pop) for nom in range(len(df.columns))]


    def set_types_var(self, *args: Literal['n', 'o', 'd', 'c']) -> None:
        """
        Définit les types de variables pour chaque variable dans le DataFrame.
        Args:
            *args (Literal['n', 'o', 'd', 'c']): Types de variables à définir.
        """
        if len(args) != len(self.variables):
            print("Attention, le nombre de types et le nombre de variables ne sont pas les mêmes.")
        for i, var in enumerate(self.variables):
            var.definir(type_var=args[i])


    def echantillon(self, taille: int) -> "Donnees":
        """
        Crée un échantillon aléatoire de la taille spécifiée à partir des données.
        Args:
            taille (int): Taille de l'échantillon à créer.
        Returns:
            Donnees: Un nouvel objet Donnees contenant l'échantillon aléatoire.
        """
        infos_vars = [variable.infos()[1] for variable in self.variables]
        sample = Donnees(self.data.sample(taille), info_pop=self.info_pop)
        for variable, info in zip(sample.variables, infos_vars):
            variable.definir(info_var=info)
        return sample


    def id_var(self, var: str | int | Variable) -> Variable:
        """
        Renvoie l'objet Variable correspondant à l'identifiant spécifié.
        Args:
            var (str | int | Variable): L'identifiant de la variable (nom, index ou objet Variable).
        Returns:
            Variable: L'objet Variable correspondant à l'identifiant spécifié.
        """
        if not isinstance(var, Variable):
            if type(var) == int:
                index = var
            elif type(var) == str:
                index = [variable.nom_col for variable in self.variables].index(var)
            return self.variables[index].copy()
        else:
            return var.copy()


    def filtre_double(self, vertical, horizontal, pourcentage=False, valeur_depart_v=None, amplitude_v=None, valeur_fin_v=None, valeur_depart_h=None, amplitude_h=None, valeur_fin_h=None, bornes_vert=None, bornes_hor=None, decimales_v=0, decimales_h=0):
        vertical = self.id_var(vertical)
        horizontal = self.id_var(horizontal)

        def filtre(sens, valeur_depart, amplitude, valeur_fin, bornes, decimales):
            if sens.est_continue and (valeur_depart is not None or amplitude is not None or valeur_fin is not None):
                decimales = decimales if decimales is not None else sens.decimales
                bornes = sens.nouvelles_bornes(valeur_depart, amplitude, valeur_fin)
                modalites = sens.copy().format_classes(bornes, decimales)
            elif sens.est_continue:
                bornes = bornes if bornes is not None else sens.bornes
                decimales = decimales if decimales is not None else sens.decimales
                modalites = sens.copy().format_classes(bornes, decimales)
            elif sens.quantitative and bornes is not None:
                sens.est_continue = True
                decimales = decimales if decimales is not None else sens.decimales
                modalites = sens.copy().format_classes(bornes, decimales)
            else:
                bornes = sens.copy().unique
                modalites = sens.copy().modalites
            return modalites, sens.copy()._filtres_bornes(*bornes)
        
        modalites_v, filtres_v = filtre(vertical, valeur_depart_v, amplitude_v, valeur_fin_v, bornes_vert, decimales_v)
        modalites_h, filtres_h = filtre(horizontal, valeur_depart_h, amplitude_h, valeur_fin_h, bornes_hor, decimales_h)

        total = 0
        frequences = []
        totaux_h = []
        for i, filtre_h in enumerate(filtres_h):
            rangee_freq = []
            sous_total = 0
            for j, filtre_v in enumerate(filtres_v):
                count = self.data[filtre_v & filtre_h][vertical.nom_col].count()
                total += count
                sous_total += count
                rangee_freq.append(count)
            frequences.append(rangee_freq)
            totaux_h.append(sous_total)

        totaux_v = []
        for i in range(len(modalites_v)):
            sous_total = 0
            for rangee in frequences:
                sous_total += rangee[i]
            totaux_v.append(sous_total)

        frequences = [[frequence / total for frequence in rangee] for rangee in frequences] if pourcentage else frequences
        totaux_v = [sous_total / total for sous_total in totaux_v] if pourcentage else totaux_v
        totaux_h = [sous_total / total for sous_total in totaux_h] if pourcentage else totaux_h
        n = total
        total = 1 if pourcentage else total

        return vertical, horizontal, modalites_h, modalites_v, frequences, totaux_h, totaux_v, total, n

    
    def _tableau_double(self, vertical, horizontal, modalites_h, modalites_v, frequences, totaux_h, totaux_v, total, n, pourcentage=False, conditionel=None): #AJOUTER CONDITIONNEL
        def formattage(num):
            return form(num * 100, 1) if pourcentage else num
        
        tableau = [[modalites_h[i]] + [formattage(num) for num in rangee] + [formattage(totaux_h[i])] for i, rangee in enumerate(frequences)]
        num_rows = len(modalites_v) + 2 # pour colonnes modalites et total
        avant = num_rows // 2
        apres = avant if num_rows % 2 == 1 else avant - 1
        
        tableau.insert(0, ['']*avant + [f'{vertical.nom_court}{f'\n({vertical.unite_mesure})' if vertical.unite_mesure else ''}'] + ['']*apres)
        tableau.insert(1, [f'{horizontal.nom_court}{f'\n({horizontal.unite_mesure})' if horizontal.unite_mesure else ''}'] + modalites_v +['Total'])
        tableau.append(['Total'] + [formattage(sous_total) for sous_total in totaux_v] + [formattage(total)])
        
        title = f'Répartition{' en pourcentage' if pourcentage else ''} de {n} {self.unites_stat} selon {vertical.nom_complet} et {horizontal.nom_complet}{', ' if self.lieu else ''}{self.lieu}{', ' if self.date else ''}{self.date}'
        
        num_rows = len(tableau)
        num_cols = len(tableau[1])

        tableau.insert(1, ['']*num_cols)
        tableau.insert(3, ['']*num_cols)
        tableau.insert(-1, ['']*num_cols)
        tableau.append(['']*num_cols)
        num_cols += 4

        for rangee in tableau:
            rangee.insert(0, '')
            rangee.insert(2, '')
            rangee.insert(-1, '')
            rangee.append('')
        num_rows += 4
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(6, 3))

        # Hide the axes
        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=tableau, loc='center', cellLoc='center')  #[1:], colLabels=tableau[0]

        # Customize the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        # table.auto_set_column_width(col=list(range(len(columns))))  # Adjust column widths

        header_height = 0.23
        table_height = 0.15
        table_width = 0.5

        # Customize cell colors
        for (row, col), cell in table.get_celld().items():
            cell.set_width(table_width)
            if (row == 0 and col == avant + 2) or (row == 2 and col == 1):
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4CAF50')
            elif row == 2 and col > 2:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor('#A5D6A7')
                cell.set_height(header_height)
            elif col == 1 and row > 3 and row < num_rows - 3:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor('#A5D6A7')
                cell.set_height(table_height)
            elif row == 0 and col in [0, 1, num_cols-2, num_cols-1] :
                cell.set_height(header_height)
                cell.visible_edges = ''
            elif row == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_linewidth(None)
            else:
                cell.set_facecolor('#f2f2f2' if row % 2 == 0 else 'white')  # Alternating row colors
                cell.set_height(table_height)

            if row == 0 or row == 2 or row == num_rows - 2:
                cell.set_text_props(weight='bold')
                cell.set_height(header_height)
                cell.set_edgecolor('black')
            if row in [1, 3, num_rows - 3, num_rows - 1]:
                cell.set_linewidth(3)
                cell.set_height(0)
            if col in [0, 2, num_cols - 3, num_cols - 1]:
                cell.set_width(0)
                cell.set_linewidth(3)
                cell.set_edgecolor('black')

        for i in range(2,num_cols-2):
            cell = table.add_cell(-1, i, width=table_width, height=0)
            cell.set_linewidth(3)
            if i in [2, num_cols - 3, num_cols - 1]:
                cell.set_width(0)

        footnote_text = f"Source: {self.source}"
    
        if self.source:
            footnote = table.add_cell(num_rows + 2, 1, width=table_width, height=header_height, text=footnote_text)
            footnote.set_linestyle('')
            footnote.set_text_props(fontsize='8', fontstyle='italic', ha='left', va='bottom', color='gray')

        title_max_length = 20 * num_cols - 6
        if len(title) > title_max_length:
            line_break = 0
            for i in range(title_max_length,0,-1):
                if title[i] == ' ':
                    line_break = i
                    break
            title = title[:line_break] + '\n' + title[line_break+1:]
            
        title_cell = table.add_cell(-2, math.ceil(num_cols / 2 - 1), width=table_width, height=header_height, text=title)
        title_cell.set_text_props(weight='bold', fontsize=14, ha='center', va='center')
        title_cell.visible_edges = ''  # Hide borders

        return table


    def tableau_double(self, vertical, horizontal, pourcentage=False, valeur_depart_v=None, amplitude_v=None, valeur_fin_v=None, valeur_depart_h=None, amplitude_h=None, valeur_fin_h=None, bornes_vert=None, bornes_hor=None, decimales_v=0, decimales_h=0, conditionel=None):
        vertical = self.id_var(vertical)
        horizontal = self.id_var(horizontal)

        return self._tableau_double(*self.filtre_double(vertical, horizontal, pourcentage, valeur_depart_v, amplitude_v, valeur_fin_v, valeur_depart_h, amplitude_h, valeur_fin_h, bornes_vert, bornes_hor, decimales_v, decimales_h), pourcentage=pourcentage, conditionel=conditionel)


    def khi_deux(self, vertical, horizontal, valeur_depart_v=None, amplitude_v=None, valeur_fin_v=None, valeur_depart_h=None, amplitude_h=None, valeur_fin_h=None, bornes_vert=None, bornes_hor=None, decimales_v=0, decimales_h=0, seuil=0.05, tableaux=False, tableau_ft=False):
        vertical = self.id_var(vertical)
        horizontal = self.id_var(horizontal)

        vertical, horizontal, modalites_h, modalites_v, frequences_observees, totaux_h, totaux_v, total, n = self.filtre_double(vertical, horizontal, False, valeur_depart_v, amplitude_v, valeur_fin_v, valeur_depart_h, amplitude_h, valeur_fin_h, bornes_vert, bornes_hor, decimales_v, decimales_h)
        frequences_observees = np.array(frequences_observees)
        frequences_theoriques = [np.array([totaux_h[i] * totaux_v[j] / total for j in range(len(totaux_v))]) for i in range(len(totaux_h))]
        min_f_t = min([min(rangee) for rangee in frequences_theoriques])
        calculs = [(f_o - f_t)**2 / f_t for f_o, f_t in zip(frequences_observees, frequences_theoriques)]
        khi_deux = sum([sum(rangee) for rangee in calculs])

        valeur_critique = chi2.ppf(1-seuil, total - 1)
        if min_f_t < 5:
            decision = 'TEST NON VALIDE --> fréquence(s) théorique(s) inférieure(s) à 5'
            tableau_ft = True
        else:
            decision = 'On ne peut pas conclure que les variables sont dépendantes.' if khi_deux <= valeur_critique else 'On peut conclure que les variables NE sont PAS indépendantes.'

        output = [
            f'La valeur critique est: {valeur_critique}',
            f'La valeur du test Khi-Deux est: {khi_deux}',
            f'La décision est: {decision}',
        ]

        print('\n'.join(output))

        if tableaux:
            self._tableau_double(vertical, horizontal, modalites_h, modalites_v, frequences_observees, totaux_h, totaux_v, total, n)
            self._tableau_double(vertical, horizontal, modalites_h, modalites_v, frequences_theoriques, totaux_h, totaux_v, total, n)
            self._tableau_double(vertical, horizontal, modalites_h, modalites_v, calculs, totaux_h, totaux_v, total, n)
        elif tableau_ft:
            tableau = self._tableau_double(vertical, horizontal, modalites_h, modalites_v, frequences_theoriques, totaux_h, totaux_v, total, n)
            for (row,col), cell in tableau.get_celld().items():
                text = cell.get_text().get_text()
                if (text) and (col > 2) and (col < 3 + len(modalites_v)) and (row > 3):
                    num = float(text)
                    if num < 5:
                        cell.set_facecolor('#FF6F61')

        return valeur_critique, khi_deux, decision
    

    def regression(self, X, Y, coeff_linearise=False, decimales=3, decimales_pourcentages=1):
        X = self.id_var(X)
        Y = self.id_var(Y)

        df = self.data.dropna(subset=[X.nom_col, Y.nom_col])
        x_info_pop, x_info_var = X.infos()
        X = Variable(df[X.nom_col]).definir(info_pop=x_info_pop, info_var=x_info_var)
        y_info_pop, y_info_var = Y.infos()
        Y = Variable(df[Y.nom_col]).definir(info_pop=y_info_pop, info_var=y_info_var)
        return Regression(X, Y, coeff_linearise, decimales, decimales_pourcentages)
    

    def test_hypo_moyenne_difference(self, var1, var2, seuil=0.05, lateralite='bilateral', echantillons_independants=True, decimales=3): # >= 30 ET < 30 ET appariés --> Note sur les conditions dans OUTPUT
        var1 = self.id_var(var1)
        var2 = self.id_var(var2)

        if var1.qualitative or var2.qualitative:
            return "Impossible d'effectuer le test, puisque les deux variables doivent être quantitatives"

        h0 = f'les deux moyennes, sont égales'
        h1 = f'la première moyenne est {['différente de', 'plus petite que', 'plus grande que'][['bilateral', 'gauche', 'droite'].index(lateralite)]} la deuxième moyenne'

        alpha = seuil / 2 if lateralite == 'bilateral' else seuil

        if not echantillons_independants:
            if var1.n != var2.n:
                print("ATTENTION, IL S'AGIT D'ÉCHANTILLON APPARIÉES, MAIS LES TAILLES DES DONNÉES NE SONT PAS LES MÊMES.")
            serie = self.data[var1.nom_col] - self.data[var2.nom_col]
            diff = Variable(serie, type_var='c')
            z = diff.moyenne * np.sqrt(diff.n) / diff.ecart_type
            z_critique = t.ppf(1-alpha, diff.n - 1)
            loi = 'student'
            condition = 'Attention, la différence des données doit sembler suivre une loi normale pour que le test soit valide.'
        elif var1.n >= 30 and var2.n >= 30:
            z = (var1.moyenne - var2.moyenne) / np.sqrt(var1.variance / var1.n + var2.variance / var2.n)
            z_critique = norm.ppf(1-alpha)
            loi = 'normale'
            condition = ''
        elif (var1.n < 30 or var2.n < 30):
            ddof = var1.n + var2.n - 2
            b = ((var1.n - 1) * var1.variance + (var2.n - 1) * var2.variance) / ddof
            c = 1 / var1.n + 1 / var2.n
            z = (var1.moyenne - var2.moyenne) / np.sqrt(b * c)
            z_critique = t.ppf(1-alpha, ddof)
            loi = 'student'
            condition = "Attention, la variable doit sembler suivre une loi normale pour que le test soit valide, puisqu'au moins un des échantillons est plus petit que 30."
        
        if lateralite == 'bilateral':
            rejeter = abs(z) > z_critique
            comparaison = f"|{form(z, decimales)}| est plus grand que {form(z_critique, decimales)}" if rejeter else f'{form(z, decimales)} est entre {form(-z_critique, decimales)} et {form(z_critique, decimales)}'
        elif lateralite == 'gauche':
            z_critique *= -1
            rejeter = z < z_critique
            comparaison = f"{form(z, decimales)} est plus petit que {form(z_critique, decimales)}" if rejeter else f"{form(z, decimales)} est plus grand ou égal à {form(z_critique, decimales)}"
        elif lateralite == 'droite':
            rejeter = z > z_critique
            comparaison = f"{form(z, decimales)} est plus grand que {form(z_critique, decimales)}" if rejeter else f"{form(z, decimales)} est plus petit ou égal à {form(z_critique, decimales)}"
        else:
            print("ATTENTION: La lateralité doit être exactement un de ces 3 choix: 'bilateral', 'gauche' ou 'droite'.")

        conclusion = 'On peut donc conlure que' if rejeter else 'On ne peut donc pas conclure que'
        decision = f"on rejette l'hypothèse nulle" if rejeter else f"on ne rejette pas l'hypothèse nulle"

        decimales_moyenne_1 = 1 if var1.decimales < 1 else var1.decimales
        decimales_moyenne_2 = 1 if var2.decimales < 1 else var2.decimales

        output = [
            f"Moyenne du premier échantillon: {form(var1.moyenne, decimales_moyenne_1)}",
            f"Moyenne du deuxième échantillon: {form(var2.moyenne, decimales_moyenne_2)}",
            f"Valeur critique: {form(z_critique, 3)} obtenue avec la loi {loi}",
            f"La statistique calculée est: {form(z, decimales)}",
            f"Puisque {comparaison}, {decision} voulant que {h0}.",
            f"{conclusion} {h1}."
        ]

        if z < 0 and lateralite == 'droite':
            output.insert(0, "Attention, peut-être que la latéralité devrait être gauche ou bilatérale, au lieu de droite.")
        elif z > 0 and lateralite == 'gauche':
            output.insert(0, "Attention, peut-être que la latéralité devrait être droite ou bilatérale, au lieu de gauche.")

        if condition:
            output.insert(0, condition)

        return '\n'.join(output)


    def test_hypo_proportion_difference(self, var1, p1, var2, p2, seuil=0.05, lateralite='bilateral', decimales=3): # > 30 --> Note sur les conditions dans OUTPUT
        var1 = self.id_var(var1)
        var2 = self.id_var(var2)
        
        conditions = [var1.n * p1 < 5, var1.n * (1 - p1) < 5, var2.n * p2 < 5, var2.n * (1 - p2) < 5]
        conditions_text = ['n1 * p1 < 30', 'n1 * (1 - p1) < 30', 'n2 * p2 < 30', 'n2 * (1 - p2) < 30']
        if var1.n < 30 or var2.n < 30:
            return "Le test est invalide puisqu'au moins un des échantillons est plus petit que 30."
        elif sum(conditions) > 0:
            ouput = []
            for condition, text in zip(conditions, conditions_text):
                if condition:
                    output.append(text)
            return f"Le test est invalide puisque {' et '.join(output)}."

        h0 = f'la première proportion est la même que la deuxième proportion'
        h1 = f'la première proportion est {['différente de', 'plus petite que', 'plus grande que'][['bilateral', 'gauche', 'droite'].index(lateralite)]} la deuxième proportion'

        alpha = seuil / 2 if lateralite == 'bilateral' else seuil
        p_barre = (var1.n * p1 + var2.n * p2) / (var1.n + var2.n)
        z = (p1 - p2) / np.sqrt(p_barre * (1 - p_barre) * (1 / var1.n + 1 / var2.n))
        z_critique = norm.ppf(1-alpha)
        
        if lateralite == 'bilateral':
            rejeter = abs(z) > z_critique
            comparaison = f"|{form(z, decimales)}| est plus grand que {form(z_critique, decimales)}" if rejeter else f'{form(z, decimales)} est entre {form(-z_critique, decimales)} et {form(z_critique, decimales)}'
        elif lateralite == 'gauche':
            z_critique *= -1
            rejeter = z < z_critique
            comparaison = f"{form(z, decimales)} est plus petit que {form(z_critique, decimales)}" if rejeter else f"{form(z, decimales)} est plus grand ou égal à {form(z_critique, decimales)}"
        elif lateralite == 'droite':
            rejeter = z > z_critique
            comparaison = f"{form(z, decimales)} est plus grand que {form(z_critique, decimales)}" if rejeter else f"{form(z, decimales)} est plus petit ou égal à {form(z_critique, decimales)}"
        else:
            print("La lateralité doit être exactement un de ces 3 choix: 'bilateral', 'gauche' ou 'droite'.")

        conclusion = 'On peut donc conlure que' if rejeter else 'On ne peut donc pas conclure que'
        decision = f"on rejette l'hypothèse nulle" if rejeter else f"on ne rejette pas l'hypothèse nulle"

        output = [
            "Attention, les échantillons doivent être indépendants.",
            f"Proportion du premier échantillon: {form(p1, 1, '%')}",
            f"Proportion du deuxième échantillon: {form(p2, 1, '%')}",
            f"Valeur critique: {form(z_critique, 3)} obtenue avec la loi normale",
            f"La statistique calculée est: {form(z, decimales)}",
            f"Puisque {comparaison}, {decision} voulant que {h0}.",
            f"{conclusion} {h1}."
        ]

        if z < 0 and lateralite == 'droite':
            output.insert(0, "Attention, peut-être que la latéralité devrait être gauche ou bilatérale, au lieu de droite.")
        elif z > 0 and lateralite == 'gauche':
            output.insert(0, "Attention, peut-être que la latéralité devrait être droite ou bilatérale, au lieu de gauche.")

        return '\n'.join(output)

# FIN