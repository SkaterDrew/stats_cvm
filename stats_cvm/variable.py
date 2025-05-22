import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import warnings
from .formattage import form
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

class Variable:
    """
    Classe représentant une variable statistique.
    Elle permet de stocker des données, de calculer des statistiques descriptives,
    de créer des classes et de générer des tableaux et graphiques.

    Attributs:
        data (pd.Series): Données de la variable.
        nom_col (str): Nom de la colonne si les données proviennent d'un DataFrame.
        type_var (str): Type de variable
            'n': Nominale
            'o': Ordinale
            'd': Discrète
            'c': Continue
        nom_complet (str): Nom complet de la variable.
        nom_court (str): Nom court de la variable.
        legende (list[str]): Légende associée à la variable Qualitative.
        unite_mesure (str): Unité de mesure de la variable.
        taille_population (int): Taille de la population, si connu, sinon None.
        est_population (bool): Indique si la variable est une population.
        unites_stat_complet (str): Nom complet des unités statistiques.
        unites_stat (str): Nom abrégées des unités statistiques.
        source (str): Source des données.
        lieu (str): Lieu des données.
        date (str): Date des données.
        qualitative (bool): Indique si la variable est qualitative.
        quantitative (bool): Indique si la variable est quantitative.
        est_continue (bool): Indique si la variable est continue (inclus discrète avec beaucoup de valeurs).
        moyenne (float): Moyenne de la variable.
        ecart_type (float): Ecart-type de la variable.
        variance (float): Variance de la variable.
        coeff_var (float): Coefficient de variation de la variable.
        mediane (float): Médiane de la variable.
        mode (list): Mode(s) de la variable.
        ecart_interquartile (float): Ecart interquartile de la variable.
        min (float): Valeur minimale de la variable.
        max (float): Valeur maximale de la variable.
        etendue (float): Etendue de la variable.
        n (int): Nombre d'observations.
        unique (list): Valeurs uniques de la variable.
        modalites (list): Modalités de la variable.
        frequences (np.ndarray): Fréquences de la variable.
        pourcentages (np.ndarray): Pourcentages de la variable.
        pourcentges_cumul (np.ndarray): Pourcentages cumulés de la variable.
        bornes (np.ndarray): Bornes des classes de la variable.
    """
    def __init__(self,
        df: pd.DataFrame | pd.Series | list | dict,
        colonne: int | str = 0,
        type_var: Literal['n', 'o', 'd', 'c'] = 'n'
    ) -> None:
        """
        Initialise la variable avec un DataFrame, une Series, une liste ou un dictionnaire.

        Args:
            df (pd.DataFrame | pd.Series | list | dict): Données à utiliser pour initialiser la variable.
            colonne (int | str, optional): Nom ou index de la colonne à utiliser si df est un DataFrame. Par défaut, 0.
            type_var (str, optional): Type de variable ('n', 'o', 'd', 'c'). Par défaut, 'n'.

        Retourne:
            None
        """
        if isinstance(df, pd.DataFrame):
            self.nom_col = df.columns[colonne] if type(colonne) == int else colonne
            self.data = df[self.nom_col].dropna()
        elif isinstance(df, pd.Series):
            self.nom_col = df.name
            self.data = df
        else:
            self.data = pd.Series(df)
            self.nom_col = ''
        self.dataframe = df

        self.isinit = False
        self.definir(type_var=type_var)

    # region DEFINITION

    def definir(
        self,
        info_pop: InfoPop | None = None,
        info_var: InfoVar | None = None,
        type_var: Literal['n', 'o', 'd', 'c' , ''] = '',
        nom_complet: str = '',
        nom_court: str = '',
        legende: list[str] = [],
        unite_mesure: str = '',
        taille_population: int = None,
        est_population = False,
        unites_stat_complet: str = '',
        unites_stat: str = '',
        source: str = '',
        lieu: str = '',
        date: str = ''
    ) -> Self:
        """
        Définit les attributs de la variable.
        Si info_pop ou info_var est fourni, les attributs sont définis à partir de ces informations.
        Sinon, ils sont définis à partir des arguments fournis.
        Si aucun argument n'est fourni, les attributs sont définis à partir des valeurs par défaut.
            - info_pop: Informations sur la population.
                - taille_population: Taille de la population.
                - est_population: Indique si la variable est une population.
                - unites_stat_complet: Nom complet des unités statistiques.
                - unites_stat: Nom abrégé des unités statistiques.
                - source: Source des données.
                - lieu: Lieu des données.
                - date: Date des données.
            - info_var: Informations sur la variable.
                - type_var: Type de variable ('n', 'o', 'd', 'c').
                - nom_complet: Nom complet de la variable.
                - nom_court: Nom court de la variable.
                - legende: Légende associée à la variable.
                - unite_mesure: Unité de mesure de la variable.
        Retourne:
            Self: L'instance de la variable.
        """
        if info_pop is not None:
            self.N, self.est_population, self.unites_stat_complet, self.unites_stat, self.source, self.lieu, self.date = info_pop.values()
        else:
            self.N = taille_population if taille_population is not None else (self.N if self.isinit else None)
            self.est_population = est_population if est_population else (self.est_population if self.isinit else False)
            self.unites_stat_complet = unites_stat_complet if unites_stat_complet else (self.unites_stat_complet if self.isinit else 'unités statistiques')
            self.unites_stat = unites_stat if unites_stat else (self.unites_stat if self.isinit else 'unités stats')

            self.lieu = lieu if lieu else (self.lieu if self.isinit else '')
            self.date = date if date else (self.date if self.isinit else '')
            self.source = source if source else (self.source if self.isinit else '')
        if info_var is not None:
            self.type_var, self.nom_complet, self.nom_court, self.legende, self.unite_mesure = info_var.values()
        else:
            self.type_var = type_var if type_var else (self.type_var if self.isinit else 'n')
            self.nom_complet = nom_complet if nom_complet else (self.nom_complet if self.isinit else self.nom_col)
            self.nom_court = nom_court if nom_court else (self.nom_court if self.isinit else self.nom_col)
            self.unite_mesure = unite_mesure if unite_mesure else (self.unite_mesure if self.isinit else '')
            self.legende = legende if legende else (self.legende if self.isinit else [])

        self.quantitative = True if self.type_var == 'd' or self.type_var == 'c' else False
        self.qualitative = not self.quantitative
        self.est_continue = True if self.type_var == 'c' else False

        self.unique = None if self.est_continue else sorted(self.data.unique())
        self.liste = np.array(self.data.tolist())

        if type_var or (info_var is not None and info_var['type_var']) or not self.isinit:
            self._calculer_stats_desc()
            self.creation_classes()
        if self.est_population:
            self.N = self.n
        
        self.isinit = True
        return self


    def infos(self) -> tuple[InfoPop, InfoVar]:
        """
        Retourne les informations sur la population et la variable sous forme de dictionnaire.
        """
        unites_stats = {
            'taille_population': self.N,
            'est_population': self.est_population,
            'unites_stat_complet': self.unites_stat_complet,
            'unites_stat': self.unites_stat,
            'source': self.source,
            'lieu': self.lieu,
            'date': self.date,
        }
        info_variable = {
            'type_var': self.type_var,
            'nom_complet': self.nom_complet,
            'nom_court': self.nom_court,
            'legende': self.legende,
            'unite_mesure': self.unite_mesure,
        }

        return unites_stats, info_variable

    # endregion

    # region COPIES

    def copy(self, sample: int | None = None) -> "Variable":
        data = self.data if sample is None else sample
        unites_stats, info_variable = self.infos()
        copie = Variable(data, self.nom_col).definir(info_pop=unites_stats, info_var=info_variable).creation_classes(bornes=self.bornes, decimales=self.decimales)
        copie.est_continue = self.est_continue
        return copie


    def log(self) -> "Variable":
        data = np.log(self.data)
        unites_stats, info_variable = self.infos()
        return Variable(data).definir(info_pop=unites_stats, info_var=info_variable)


    def echantillon(self, k: int) -> "Variable":
        sample = self.data.sample(k)
        return self.copy(sample)

    # endregion

    # region STATS

    def _calculer_stats_desc(self):
        self.n = self.data.count()
        self.est_population = self.est_population
        self.N = self.n if self.est_population else self.N

        self.min = self.data.min() if self.quantitative else None
        self.max = self.data.max() if self.quantitative else None
        self.etendue = self.max - self.min if self.quantitative else None

        self.moyenne = self.data.mean() if self.quantitative else None
        self.ecart_type = self.data.std(ddof=int(not self.est_population)) if self.quantitative else None
        self.variance = self.data.var(ddof=int(not self.est_population)) if self.quantitative else None
        self.coeff_var = self.ecart_type / self.moyenne if self.quantitative else None

        self.mediane = self.data.median() if self.type_var != 'n' else None
        self.mode = self.data.mode().tolist() if self.type_var != 'c' else None

        self.ecart_interquartile = self.data.quantile(0.75) - self.data.quantile(0.25)
    

    def quartile(self, num):
        if num not in [1,2,3]:
            raise ValueError('Choisir un nombre entre 1 et 3 inclusivement')
        return self.data.quantile(num*0.25)


    def quintile(self, num):
        if num not in [1,2,3,4]:
            raise ValueError('Choisir un nombre entre 1 et 4 inclusivement')
        return self.data.quantile(num*0.2)


    def decile(self, num):
        if num not in range(1,11):
            raise ValueError('Choisir un nombre entre 1 et 10 inclusivement')
        return self.data.quantile(num*0.1)


    def centile(self, num):
        if num not in range(1,101):
            raise ValueError('Choisir un nombre entre 1 et 100 inclusivement')
        return self.data.quantile(num*0.01)
    

    def stats(self, decimales=1):
        types = ["Qualitative nominale", "Qualitative ordinale", "Quantitative discrète", "Quantitative continue"]
        output = [
            f"Statistiques descriptives pour {self.nom_complet}\n",
            f"Type de variable: {types[['n', 'o', 'd', 'c'].index(self.type_var)]}",
            f'Taille: {self.n}\n'
        ]
        dec_type = decimales if self.type_var == 'c' else 0
        if self.type_var in ['n', 'o', 'd']:
            modes = ' et '.join(map(lambda x: form(x, dec_type), self.mode))
            output.append(f'Mode{'s' if len(self.mode)>1 else ''}: {modes}')
        if self.type_var in ['o', 'd', 'c']:
            output.append(f'Mediane: {form(self.mediane, dec_type)}')
            output.append(f'Premier quartile: {form(self.quartile(1), dec_type)}')
            output.append(f'Troisieme quartile: {form(self.quartile(3), dec_type)}')
        if self.quantitative:
            output.append(f'Ecart interquartile: {form(self.ecart_interquartile, dec_type)}\n')
            output.append(f'Minimum: {form(self.min, dec_type)}')
            output.append(f'Maximum: {form(self.max, dec_type)}')
            output.append(f'Etendue: {form(self.etendue, dec_type)}\n')
            output.append(f'Moyenne: {form(self.moyenne, decimales)}')
            output.append(f'Ecart-type: {form(self.ecart_type, decimales)}')
            output.append(f'Coefficient de variation: {form(self.coeff_var, 1, '%')}')
        
        return '\n'.join(output)

    # endregion

    # region FILTRES & PROPORTIONS

    def _filtre(self, *args):
        if self.est_continue:
            if len(args) > 2 or (len(args) > 0 and (args[0][0] not in ['<','>'])) or (len(args) == 2 and ((args[1][0] not in ['<','>']) or (args[0][0] == args[1][0]))):
                raise ValueError("Il doit y avoir au maximum 2 bornes, une inferieur, de la forme '<35' ou '<=35', et une superieure, de la forme '>35' ou '>=35'.")
            elif len(args) == 0:
                return 1
            
            inf = 0 if args[0][0] == '>' else 1

            if len(args) == 1:
                borne_inf = args[0][1:] if inf == 0 else '-inf'
                borne_sup = args[0][1:] if inf == 1 else 'inf'
            else:
                borne_inf = args[inf][1:]
                borne_sup = args[1-inf][1:]
                
            inclus_inf = True if borne_inf[0] == '=' else False
            inclus_sup = True if borne_sup[0] == '=' else False

            borne_inf = float(borne_inf[1:]) if inclus_inf else float(borne_inf)
            borne_sup = float(borne_sup[1:]) if inclus_sup else float(borne_sup)

            data_inf = self.data >= borne_inf if inclus_inf else self.data > borne_inf
            data_sup = self.data <= borne_sup if inclus_sup else self.data < borne_sup
            filtre = data_inf & data_sup
        
        else:
            arg_list = self.data.unique()
            elements = list(args)
            for i, arg in enumerate(args):
                if (arg not in arg_list) and (arg not in self.legende):
                    warnings.warn(f"L'élément {arg} n'est pas une modalité de cette variable.")
                elif arg in self.legende:
                    elements[i] = self.legende.index(arg) + 1
            conditions = [self.data == arg for arg in elements]
            filtre = False
            for condition in conditions:
                filtre = filtre | condition
            
        return filtre
    

    def _filtres_bornes(self, *args):
        if self.est_continue:
            filtres = []
            for inf, sup in zip(args[:-1], args[1:]):
                if inf == float('-inf'):
                    filtres.append(self._filtre(f'<{sup}'))
                elif sup == float('inf'):
                    filtres.append(self._filtre(f'>={inf}'))
                else:
                    filtres.append(self._filtre(f'>={inf}', f'<{sup}'))
        else:
            filtres = [self._filtre(arg) for arg in args]

        return filtres


    def filtre(self, *args):
        return self.data[self._filtre(*args)]
    

    def filtres_bornes(self, *args):
        return [self.data[filtre] for filtre in self._filtres_bornes(*args)]


    def proportion(self, *args):
        return self.filtre(*args).count() / self.n

    # endregion

    # region CLASSES & BORNES

    def format_classes(self, bornes, decimales=0):
        bornes = np.array(bornes)
        self.decimales = decimales
        if decimales < -1:
            niveau = (abs(decimales) + 1) // 3
            bornes = bornes / 10**(niveau * 3)
            if niveau <= 3:
                puissance_unite = ['milliers', 'millions', 'milliards'][niveau-1]
            else:
                puissance_unite = f"E+{3 * niveau}"
            mots_unite = self.unite_mesure.split(' ')
            if len(mots_unite) >= 2 and mots_unite[0] == 'En':
                mots_unite[1] = puissance_unite
                self.unite_mesure = ' '.join(mots_unite)
            else:
                self.unite_mesure = f'En {puissance_unite}{' de ' if self.unite_mesure else ''}{self.unite_mesure}'
            decimales += niveau * 3
        else:
            niveau = 0
            mots_unite = self.unite_mesure.split(' ')
            if len(mots_unite) >= 2 and mots_unite[0] == 'En':
                fin = 3 if len(mots_unite) > 2 and mots_unite[2] == 'de' else 2
                self.unite_mesure = self.unite_mesure.replace(' '.join(mots_unite[0:fin]), '').lstrip()
        
        self.puissance_mille = niveau

        if self.type_var == 'c':
            bornes_format = [f"moins de {form(bornes[1], decimales)}"] if bornes[0] == float('-inf') else [f"[ {form(bornes[0], decimales)} ; {form(bornes[1], decimales)} ["]
            bornes_format += [f"[ {form(inf, decimales)} ; {form(sup, decimales)} [" for inf, sup in zip(bornes[1:-2], bornes[2:-1])]
            bornes_format.append(f"{form(bornes[-2], decimales)} ou plus" if bornes[-1] == float('inf') else f"[ {form(bornes[-2], decimales)} ; {form(bornes[-1], decimales)} [")
        else:
            bornes_format = [f"{bornes[1]-1} ou moins"] if bornes[0] == float('-inf') else [f"{form(bornes[0], decimales)} - {form(bornes[1] - 1, decimales)}"]
            bornes_format += [f"{form(inf, decimales)} - {form(sup - 1, decimales)}" for inf, sup in zip(bornes[1:-2], bornes[2:-1])]
            bornes_format.append(f"{form(bornes[-2], decimales)} ou plus" if bornes[-1] == float('inf') else f"{form(bornes[-2], decimales)} - {form(bornes[-1] - 1, decimales)}")
        
        return bornes_format


    def nouvelles_bornes(self, valeur_depart=None, amplitude=None, valeur_fin=None):
        return self.copy()._creation_bornes(valeur_depart=valeur_depart, amplitude=amplitude, valeur_fin=valeur_fin)[0]


    def _creation_bornes(self, valeur_depart=None, amplitude=None, valeur_fin=None):
        nb_classes_ideal = math.floor(np.log2(self.n/10)) + 5
        i = 0
        est_petit = False
        est_grand = False
        beaux_nombres = np.array([1, 2, 3, 5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, 120, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500]) #, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 75000, 100000, 150000, 200000, 250000, 300000, 400000, 500000, 750000, 1000000])

        if amplitude is None:
            amplitude_ideale = self.etendue / nb_classes_ideal
            while amplitude_ideale < 0.875:
                i += 1
                amplitude_ideale *= 1000
                est_petit = True
            amplitude = beaux_nombres[np.abs(beaux_nombres - amplitude_ideale).argmin()]
            while amplitude_ideale > 8750:
                i -= 1
                amplitude_ideale /= 1000
                est_grand = True
            amplitude = beaux_nombres[np.abs(beaux_nombres - amplitude_ideale).argmin()]
        else:
            while amplitude < 0.875:
                i += 1
                amplitude *= 1000
                est_petit = True
            while amplitude > 8750:
                i -= 1
                amplitude /= 1000
                est_grand = True

        if valeur_depart is None:
            valeur_depart = math.floor(self.min * (1000 ** i) / amplitude) * amplitude
        else:
            valeur_depart *= 1000 ** i

        fin = math.ceil(self.max * (1000 ** i) / amplitude) * amplitude if valeur_fin is None else valeur_fin  * (1000 ** i)

        bornes = list(range(int(valeur_depart), int(fin) + 1, int(amplitude)))
        if valeur_depart > self.min * (1000 ** i):
            bornes.insert(0, float('-inf'))
        if valeur_fin is not None and self.max * (1000 ** i) >= fin:
            bornes.append(float('inf'))
        bornes = np.array(bornes) / (1000 ** i)
        amplitude = np.array(amplitude) / (1000 ** i)

        return bornes, amplitude, i, valeur_depart


    def creation_classes(self, valeur_depart=None, amplitude=None, valeur_fin=None, bornes=None, decimales=None):
        if bornes is not None and self.quantitative:
            self.est_continue = True
        else:
            nb_classes_ideal = math.floor(np.log2(self.n/10)) + 5
            if (self.type_var == 'd') and (len(self.unique) > nb_classes_ideal + 2 or amplitude is not None):
                self.est_continue = True

        if not self.est_continue:
            self.classes = None
            self.bornes = None
            self.milieux = None
            self.frequences = np.array(self.data.value_counts().sort_index())
            self.amplitude = 1
            self.valeur_depart = self.unique[0]
            self.decimales = 0
            self.puissance_mille = 0
        else:
            if bornes is None:
                bornes, amplitude, i, valeur_depart = self._creation_bornes(valeur_depart, amplitude, valeur_fin)
            else:
                i = 0
                bornes = np.array(bornes)
                amplitude = None

            decimales = i * 3 if decimales is None else decimales
            bornes_format = self.format_classes(bornes, decimales)

            if amplitude is None:
                milieux = (bornes[1:] + bornes[:-1]) / 2
                amplitude = max((bornes[2:-1] - bornes[1:-2]))
                if bornes[-1] == float('inf'):
                    milieux[-1] = bornes[-2] + amplitude / 2
            else:
                milieux = bornes[:-1] + amplitude / 2

            if bornes[0] == float('-inf'):
                milieux[0] = bornes[1] - amplitude / 2

            frequences = [serie.count() for serie in self.filtres_bornes(*bornes)]

            self.classes = bornes_format
            self.bornes = bornes
            self.milieux = milieux
            self.frequences = np.array(frequences)
            self.amplitude = amplitude
            self.valeur_depart = valeur_depart

        if self.est_continue:
            self.modalites = self.classes
        elif self.legende:
            self.modalites = self.legende
        else:
            self.modalites = self.unique

        self.pourcentages = self.frequences / self.n * 100
        self.pourcentages_arrondis = np.round(self.frequences / self.n * 100, 1)
        self.pourcentages_cumul = np.array([sum(self.pourcentages_arrondis[:i+1]) for i in range(len(self.pourcentages))])
        return self

    # endregion

    # region TABLEAUX & GRAPHIQUES

    def tableau_frequences(self, valeur_depart=None, amplitude=None, valeur_fin=None, bornes=None, decimales=None, decimales_pourcentages=1):
        if self.quantitative and (valeur_depart is not None or amplitude is not None or valeur_fin is not None):
            decimales = self.decimales if decimales is None else decimales
            copie = self.copy().creation_classes(valeur_depart, amplitude, valeur_fin, decimales=decimales)
            bornes = copie.bornes
        elif self.quantitative and (bornes is not None or decimales is not None):
            bornes = self.bornes if bornes is None else bornes
            decimales = self.decimales if decimales is None else decimales
            copie = self.copy().creation_classes(bornes=bornes, decimales=decimales)
            bornes = copie.bornes
        else:
            copie = self.copy()
            bornes = copie.bornes
            decimales = copie.decimales
            
        modalites = copie.modalites.copy()
        frequences = copie.frequences.copy()
        unite_mesure = copie.unite_mesure
    
        modalites.append('Total')
        pourcentages = np.array(list(map(lambda x: round(x, decimales_pourcentages+2), frequences / copie.n)))
        pourcentages_cumul = np.array(list(map(lambda i: sum(pourcentages[:i+1]), range(len(pourcentages)))))
        
        frequences = list(np.append(frequences, copie.n))
        pourcentages = np.append(pourcentages, sum(pourcentages))
        pourcentages = [form(p, decimales_pourcentages, '%') for p in pourcentages]
        pourcentages_cumul = [form(p, decimales_pourcentages, '%') for p in pourcentages_cumul] + ['']

        for liste in [modalites, frequences, pourcentages, pourcentages_cumul]:
            liste.insert(0, '')
            liste.insert(-1, '')
            liste.append('')

        num_rows = len(modalites)
        unite_stats = copie.unites_stat
        d_unite_stats = "d'" if unite_stats[0].lower() in 'aeiouy' else 'de '
        nom = copie.nom_court
        unite = f"\n({unite_mesure})" if unite_mesure else ''

        if copie.type_var != 'n':
            data = list(zip(['']*num_rows, modalites, ['']*num_rows, frequences, ['']*num_rows, pourcentages, ['']*num_rows, pourcentages_cumul, ['']*num_rows))
            columns = [
                '',
                f"{nom}{unite}",
                '',
                f"Nombre\n{d_unite_stats}{unite_stats}",
                '',
                f"Pourcentage\n{d_unite_stats}{unite_stats}",
                '',
                f"Pourcentage cumulé\n{d_unite_stats}{unite_stats}",
                ''
            ]
        else:
            data = list(zip(['']*num_rows, modalites, ['']*num_rows, frequences, ['']*num_rows, pourcentages, ['']*num_rows))
            columns = [
                '',
                f"{nom}{unite}",
                '',
                f"Nombre\n{d_unite_stats}{unite_stats}",
                '',
                f"Pourcentage\n{d_unite_stats}{unite_stats}",
                ''
            ]

        # Create a figure and axis
        plt.figure()
        fig, ax = plt.subplots(figsize=(6, 3))

        # Hide the axes
        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')

        # Customize the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        # table.auto_set_column_width(col=list(range(len(columns))))  # Adjust column widths

        header_height = 0.23
        table_height = 0.15
        table_width = 0.5

        # Customize cell colors
        for (row, col), cell in table.get_celld().items():
            if col % 2 == 1:
                cell.set_width(table_width)
                if row == 0:  # Header row
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4CAF50')  # Green header
                    cell.set_height(header_height)
                else:
                    cell.set_facecolor('#f2f2f2' if row % 2 == 0 else 'white')  # Alternating row colors
                    cell.set_height(table_height)
                if row == num_rows - 1:
                    cell.set_text_props(weight='bold')
            else:
                cell.set_width(0)
                cell.set_linewidth(3)
                if row == 0:
                    cell.set_height(header_height)
                else:
                    cell.set_height(table_height)
            if row in [1, num_rows -2, num_rows]:
                cell.set_height(0)
                cell.set_linewidth(3)

        title = f"Répartition de {copie.n} {copie.unites_stat} selon {copie.nom_complet}{', ' if copie.lieu else ''}{copie.lieu}{', ' if copie.date else ''}{copie.date}"
        if len(title) > 82:
            line_break = 0
            for i in range(82,0,-1):
                if title[i] == ' ':
                    line_break = i
                    break
            title = title[:line_break] + '\n' + title[line_break+1:]

        num_cols = 7 if copie.type_var == 'n' else 9
        for i in range(num_cols):
            if i % 2 == 0:
                cell = table.add_cell(-1, i, width=0, height=0, text='')
                cell2 = table.add_cell(num_rows + 1, i, width=0, height=0.01, text='')
            else:
                cell = table.add_cell(-1, i, width=table_width, height=0, text='')
                cell2 = table.add_cell(num_rows + 1, i, width=table_width, height=0.01, text='')
            cell.set_linewidth(3)
            cell2.set_linewidth(3)
            cell2.visible_edges = 'T'
            
        col_title = num_cols // 2
        title_width = table_width if copie.type_var == 'n' else 0
        title_cell = table.add_cell(-2, col_title, width=title_width, height=header_height, text=title)
        title_cell.set_text_props(weight='bold', fontsize=14, ha='center', va='center')
        title_cell.visible_edges = ''
        
        erreur_arrondis = "*Le total des pourcentages n'est pas 100,0% en raison des arrondis." if (pourcentages[-2] != '100,0%') else ""
        source = f"Source: {copie.source}" if copie.source else ''
        footnote_text = f"{source}{'\n' if copie.source and erreur_arrondis else ''}{erreur_arrondis}"
    
        if footnote_text:
            footnote = table.add_cell(num_rows + 2, 1, width=table_width, height=header_height, text=footnote_text)
            footnote.set_linestyle('')
            footnote.set_text_props(fontsize='8', fontstyle='italic', ha='left', va='bottom', color='gray')

        return table


    def histo(self, valeur_depart=None, amplitude=None, valeur_fin=None, pourcentage=True, rotation_x=0, etiquette_barres=False, bornes=None, decimales=None, decimales_pourcentages=1):
        if self.quantitative and (valeur_depart is not None or amplitude is not None or valeur_fin is not None):
            decimales = self.decimales if decimales is None else decimales
            copie = self.copy().creation_classes(valeur_depart, amplitude, valeur_fin, decimales=decimales)
            bornes = copie.bornes
        elif self.quantitative and (bornes is not None or decimales is not None):
            bornes = self.bornes if bornes is None else bornes
            decimales = self.decimales if decimales is None else decimales
            copie = self.copy().creation_classes(bornes=bornes, decimales=decimales)
            bornes = copie.bornes
        else:
            copie = self.copy()
            bornes = copie.bornes
            decimales = copie.decimales
            
        y = copie.pourcentages if pourcentage else copie.frequences
        x = copie.milieux if copie.est_continue else copie.unique
        amplitude = copie.amplitude

        decimales_etiquettes = copie.decimales + copie.puissance_mille * 3

        if copie.est_continue:
            x_grad = copie.bornes
            x_etiquettes = [form(num, decimales_etiquettes) for num in x_grad / (1000 ** copie.puissance_mille)]
            if x_grad[0] == float('-inf'):
                x_etiquettes[0] = r'$-\infty$'
                x_grad[0] = x_grad[1] - amplitude
            if x_grad[-1] == float('inf'):
                x_etiquettes[-1] = r'$\infty$'
                x_grad[-1] = x_grad[-2] + amplitude
        else:
            x_grad = x
            x_etiquettes = copie.legende if copie.legende else x_grad.copy()


        if copie.qualitative:
            largeur_barre = 0.5
        elif copie.est_continue:
            largeur_barre = amplitude
        else:
            largeur_barre = 0.1
        
        plt.figure()
        plt.bar(x, y, width=largeur_barre, color='skyblue', edgecolor='black')

        # Customize the graph
        plt.title(f"Répartition de {copie.n} {copie.unites_stat} selon {copie.nom_complet}{', ' if copie.lieu else ''}{copie.lieu}{', ' if copie.date else ''}{copie.date}")
        titre_unites = f" ({copie.unite_mesure})" if copie.unite_mesure else ''
        plt.xlabel(f"{copie.nom_court}{titre_unites}")
        plt.xticks(x_grad, x_etiquettes, rotation=rotation_x)
        if copie.est_continue:
            plt.xlim(x_grad[0], x_grad[-1])

        est_voyelle = copie.unites_stat[0] in 'aeiouy'
        d_unite_stats = "d'" if est_voyelle else 'de '
        plt.ylabel(f"{'Pourcentage' if pourcentage else 'Nombre'} {d_unite_stats}{copie.unites_stat}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if etiquette_barres:
            somme = sum([round(i / 100, decimales_pourcentages) for i in y])
            pad = max(y) /50
            for position, hauteur in zip(x, y):
                etiquette = form(hauteur / 100, decimales_pourcentages, '%') if pourcentage else str(hauteur)
                plt.text(position, hauteur + pad, etiquette, ha='center', fontsize=8)
            # Add a note at the bottom of the plot
            erreur_arrondis = "Le total des pourcentages n'est pas 100,0% en raison des arrondis." if pourcentage and (somme != 100) else ""
            if copie.source or erreur_arrondis:
                text_height = -0.05 if rotation_x == 0 else -0.1
                source = f"Source: {copie.source}" if copie.source else ''
                footnote_text = f"{source}{'\n' if copie.source and erreur_arrondis else ''}{erreur_arrondis}"
                plt.figtext(0.5, text_height, footnote_text, ha='center', fontsize=8, fontstyle='italic', color='gray')


    def barres_verticales(self, pourcentage=True, rotation_x=0, etiquette_barres=False, decimales_pourcentages=1):
        return self.histo(pourcentage=pourcentage, rotation_x=rotation_x, etiquette_barres=etiquette_barres, decimales_pourcentages=decimales_pourcentages)
    
    
    def batons(self, pourcentage=True, rotation_x=0, etiquette_barres=False, decimales_pourcentages=1):
        return self.histo(pourcentage=pourcentage, rotation_x=rotation_x, etiquette_barres=etiquette_barres, decimales_pourcentages=decimales_pourcentages)
    
    
    def barres_horizontales(self, pourcentage=True, rotation_x=0, etiquette_barres=False, decimales_pourcentages=1):
        if self.quantitative:
            return "Non valide pour des variables quantitatives"
        
        y = self.pourcentages if pourcentage else self.frequences
        x = self.unique

        x_etiquettes = self.legende if self.legende else x

        plt.figure()
        plt.barh(x, y, height=0.5, color='skyblue', edgecolor='black')

        # Customize the graph
        plt.title(f'Répartition de {self.n} {self.unites_stat} selon {self.nom_complet}{', ' if self.lieu else ''}{self.lieu}{', ' if self.date else ''}{self.date}')
        plt.ylabel(f"{self.nom_court}")
        plt.yticks(x, x_etiquettes, rotation=rotation_x)

        est_voyelle = self.unites_stat[0] in 'aeiouy'
        d_unite_stats = "d'" if est_voyelle else 'de '
        plt.ylabel(f"{'Pourcentage' if pourcentage else 'Nombre'} {d_unite_stats}{self.unites_stat}")
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        if etiquette_barres:
            somme = sum([round(i / 100, decimales_pourcentages) for i in y])
            pad = max(y) / 50
            for position, hauteur in zip(x, y):
                etiquette = form(hauteur / 100, decimales_pourcentages, '%') if pourcentage else str(hauteur)
                plt.text(hauteur + 0.3, position, etiquette, ha='left', fontsize=8)
            # Add a note at the bottom of the plot
            erreur_arrondis = "Le total des pourcentages n'est pas 100,0% en raison des arrondis." if pourcentage and (somme != 100) else ""
            if self.source or erreur_arrondis:
                text_height = -0.05 if rotation_x == 0 else -0.1
                source = f"Source: {self.source}" if self.source else ''
                footnote_text = f"{source}{'\n' if self.source and erreur_arrondis else ''}{erreur_arrondis}"
                plt.figtext(0.5, text_height, footnote_text, ha='center', fontsize=8, fontstyle='italic', color='gray')

    
    def diagramme_circulaire(self, decimales_pourcentages=1):
        if self.type_var != 'n':
            return "Valide seulement pour des variables qualitatives nominales"
        
        y = self.pourcentages
        x = self.unique
        x_etiquettes = self.legende if self.legende else x

        def formattage(perc):
            return form(perc / 100, decimales_pourcentages, '%')

        plt.figure()
        plt.pie(y, labels=x_etiquettes, autopct=formattage)

        # Customize the chart
        plt.title(f"Répartition de {self.n} {self.unites_stat} selon {self.nom_complet}{', ' if self.lieu else ''}{self.lieu}{', ' if self.date else ''}{self.date}")
        plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle

        # Add a note at the bottom of the plot
        erreur_arrondis = "Le total des pourcentages n'est pas 100,0% en raison des arrondis." if sum(np.round(y, decimals=decimales_pourcentages)) != 100 else ""
        source = f"Source: {self.source}" if self.source else ''
        footnote_text = f"{source}{'\n' if self.source and erreur_arrondis else ''}{erreur_arrondis}"
        if self.source or erreur_arrondis:
            plt.figtext(0.5, 0.01, footnote_text, ha='center', fontsize=8, fontstyle='italic', color='gray')

        # Show the chart
        plt.show()
    

    def polygone(self, valeur_depart=None, amplitude=None, valeur_fin=None, pourcentage=True, rotation_x=0, etiquette_barres=False, bornes=None, decimales=None, decimales_pourcentages=1):
        if self.quantitative and (valeur_depart is not None or amplitude is not None or valeur_fin is not None):
            decimales = self.decimales if decimales is None else decimales
            copie = self.copy().creation_classes(valeur_depart, amplitude, valeur_fin, decimales=decimales)
            bornes = copie.bornes
        elif self.quantitative and (bornes is not None or decimales is not None):
            bornes = self.bornes if bornes is None else bornes
            decimales = self.decimales if decimales is None else decimales
            copie = self.copy().creation_classes(bornes=bornes, decimales=decimales)
            bornes = copie.bornes
        elif self.est_continue:
            copie = self.copy()
            bornes = copie.bornes
            decimales = copie.decimales
        else:
            return "Valide seulement pour des variables quantitatives continues."

        if (bornes[0] == float('-inf')) or (bornes[-1] == float('inf')):
            return "Valide seulement pour des classes fermées, pas de bornes infinies."

        decimales_etiquettes = copie.decimales + copie.puissance_mille * 3

        y = np.pad(copie.pourcentages if pourcentage else copie.frequences, (1, 1), constant_values=0)
        x = copie.milieux
        amplitude = copie.amplitude
        x_inf = max(0, x[0] - amplitude) if copie.min >= 0 else x[0] - amplitude
        x = np.insert(x, 0, x_inf)
        x = np.append(x, x[-1] + amplitude)

        x_grad = copie.bornes
        x_grad_inf = max(0, x_grad[0] - amplitude) if copie.min >= 0 else x_grad[0] - amplitude
        x_grad = np.insert(x_grad, 0, x_grad_inf)
        x_grad = np.append(x_grad, x_grad[-1] + amplitude)
        x_etiquettes = [form(num, decimales_etiquettes) for num in x_grad / (1000 ** copie.puissance_mille)]
        
        plt.figure()
        plt.plot(x, y, marker='o', linestyle='-', color='blue')

        # Customize the graph
        plt.title(f"Répartition de {copie.n} {copie.unites_stat} selon {copie.nom_complet}{', ' if copie.lieu else ''}{copie.lieu}{', ' if copie.date else ''}{copie.date}")
        titre_unites = f" ({copie.unite_mesure})" if copie.unite_mesure else ''
        plt.xlabel(f"{copie.nom_court}{titre_unites}")
        plt.xticks(x_grad, x_etiquettes, rotation=rotation_x)
        plt.xlim(x_grad[0] - amplitude / 3, x_grad[-1])

        est_voyelle = copie.unites_stat[0] in 'aeiouy'
        d_unite_stats = "d'" if est_voyelle else 'de '
        plt.ylabel(f"{'Pourcentage' if pourcentage else 'Nombre'} {d_unite_stats}{copie.unites_stat}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if etiquette_barres:
            somme = sum([round(i / 100, decimales_pourcentages) for i in y])
            pad = max(y) / 50
            for position, hauteur, i in zip(x[1:-1], y[1:-1], range(1, len(y)-1)):
                etiquette = form(hauteur / 100, decimales_pourcentages, '%') if pourcentage else str(hauteur)
                if hauteur > y[i-1] and hauteur > y[i+1]:
                    pos_etiq_x = 'center'
                    pos_etiq_y = 1
                elif hauteur < y[i-1] and hauteur < y[i+1]:
                    pos_etiq_x = 'center'
                    pos_etiq_y = -1
                elif hauteur > y[i-1] and hauteur < y[i+1]:
                    pos_etiq_x = 'right'
                    pos_etiq_y = 1
                elif hauteur < y[i-1] and hauteur > y[i+1]:
                    pos_etiq_x = 'left'
                    pos_etiq_y = 1
                plt.text(position, hauteur + pos_etiq_y * pad, etiquette, ha=pos_etiq_x, fontsize=8)
            # Add a note at the bottom of the plot
            erreur_arrondis = "Le total des pourcentages n'est pas 100,0% en raison des arrondis." if pourcentage and (somme != 100) else ""
            if copie.source or erreur_arrondis:
                text_height = -0.05 if rotation_x == 0 else -0.1
                source = f"Source: {copie.source}" if copie.source else ''
                footnote_text = f"{source}{'\n' if copie.source and erreur_arrondis else ''}{erreur_arrondis}"
                plt.figtext(0.5, text_height, footnote_text, ha='center', fontsize=8, fontstyle='italic', color='gray')
    

    def ogive(self, valeur_depart=None, amplitude=None, valeur_fin=None, rotation_x=0, etiquette_barres=False, bornes=None, decimales=None, decimales_pourcentages=1):
        if self.quantitative and (valeur_depart is not None or amplitude is not None or valeur_fin is not None):
            decimales = self.decimales if decimales is None else decimales
            copie = self.copy().creation_classes(valeur_depart, amplitude, valeur_fin, decimales=decimales)
            bornes = copie.bornes
        elif self.quantitative and (bornes is not None or decimales is not None):
            bornes = self.bornes if bornes is None else bornes
            decimales = self.decimales if decimales is None else decimales
            copie = self.copy().creation_classes(bornes=bornes, decimales=decimales)
            bornes = copie.bornes
        elif self.est_continue:
            copie = self.copy()
            bornes = copie.bornes
            decimales = copie.decimales
        else:
            return "Valide seulement pour des variables quantitatives continues."

        if (bornes[0] == float('-inf')) or (bornes[-1] == float('inf')):
            return "Valide seulement pour des classes fermées, pas de bornes infinies."

        decimales_etiquettes = copie.decimales + copie.puissance_mille * 3
        
        y = np.insert(copie.pourcentages_cumul, 0, 0)
        x = copie.bornes
        x_etiquettes = [form(num, decimales_etiquettes) for num in x / (1000 ** copie.puissance_mille)]

        amplitude = copie.amplitude
        
        plt.figure()
        plt.plot(x, y, marker='o', linestyle='-', color='blue')

        # Customize the graph
        plt.title(f"Répartition cumulée de {copie.n} {copie.unites_stat} selon {copie.nom_complet}{', ' if copie.lieu else ''}{copie.lieu}{', ' if copie.date else ''}{copie.date}")
        titre_unites = f" ({copie.unite_mesure})" if copie.unite_mesure else ''
        plt.xlabel(f"{copie.nom_court}{titre_unites}")
        plt.xticks(x, x_etiquettes, rotation=rotation_x)
        plt.xlim(x[0] - amplitude / 3, x[-1] + amplitude / 5)

        est_voyelle = copie.unites_stat[0] in 'aeiouy'
        d_unite_stats = "d'" if est_voyelle else 'de '
        plt.ylabel(f"Pourcentage {d_unite_stats}{copie.unites_stat}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if etiquette_barres:
            for position, hauteur in zip(x[1:], y[1:]):
                etiquette = form(hauteur / 100, decimales_pourcentages, '%')
                plt.text(position, hauteur + 1, etiquette, ha='right', fontsize=8)
            # Add a note at the bottom of the plot
            erreur_arrondis = "Le total des pourcentages n'est pas 100,0% en raison des arrondis." if (round(y[-1], decimales_pourcentages) != 100.0) else ""
            if copie.source or erreur_arrondis:
                text_height = -0.05 if rotation_x == 0 else -0.1
                source = f"Source: {copie.source}" if copie.source else ''
                footnote_text = f"{source}{'\n' if copie.source and erreur_arrondis else ''}{erreur_arrondis}"
                plt.figtext(0.5, text_height, footnote_text, ha='center', fontsize=8, fontstyle='italic', color='gray')

    
    def visuel(self, valeur_depart=None, amplitude=None, valeur_fin=None, pourcentage=True, rotation_x=0, etiquette_barres=False, bornes=None, decimales=None, decimales_pourcentages=1):
        self.tableau_frequences(valeur_depart, amplitude, valeur_fin, bornes, decimales, decimales_pourcentages)

        if self.type_var == 'n':
            self.diagramme_circulaire(decimales_pourcentages)
            plt.show()
            self.barres_verticales(pourcentage, rotation_x, etiquette_barres, decimales_pourcentages)
            plt.show()
            self.barres_horizontales(pourcentage, rotation_x, etiquette_barres, decimales_pourcentages)
            plt.show()
        elif self.type_var == 'o':
            self.barres_verticales(pourcentage, rotation_x, etiquette_barres, decimales_pourcentages)
            plt.show()
            self.barres_horizontales(pourcentage, rotation_x, etiquette_barres, decimales_pourcentages)
            plt.show()
        elif self.type_var == 'd':
            self.batons(pourcentage, rotation_x, etiquette_barres, decimales_pourcentages) 
            plt.show()
        elif self.type_var == 'c':
            self.histo(valeur_depart, amplitude, valeur_fin, pourcentage, rotation_x, etiquette_barres, bornes, decimales, decimales_pourcentages)
            plt.show()
            self.polygone(valeur_depart, amplitude, valeur_fin, pourcentage, rotation_x, etiquette_barres, bornes, decimales, decimales_pourcentages)
            plt.show()
            self.ogive(valeur_depart, amplitude, valeur_fin, rotation_x, etiquette_barres, bornes, decimales, decimales_pourcentages)
            plt.show()

    
    # endregion

    # region INTERVALLES DE CONFIANCE

    def intervalle_confiance_moyenne(self, nc=95, decimales=1, remise=True):
        if self.qualitative:
            return "Impossible à calculer, puisque la variable doit être quantitative"
        
        m, s, n = self.moyenne, self.ecart_type, self.n
        alpha = 1-nc/100
        normale = True if self.n >= 30 else False
        cote = norm.ppf(1-alpha/2) if normale else t.ppf(1-alpha/2, self.n-1)

        erreur_type = s / np.sqrt(n)
        taux_sondage = self.n / self.N if (self.N is not None) and self.n >= 30 else 0
        facteur_correction = 1 if remise or (taux_sondage <= 0.05) else np.sqrt((self.N - self.n) / (self.N - 1))

        marge_erreur = cote * erreur_type * facteur_correction
        borne_inf = m - marge_erreur
        borne_sup = m + marge_erreur

        decimales_ic = decimales if self.type_var == 'c' else 0
        decimales_rep = max(1, decimales)

        output = [
            f"Intervalle de confiance pour une moyenne de {form(m, decimales_rep)} avec un niveau de confiance de {nc}%",
            f"    {"z_{alpha/2}" if normale else "t_{alpha/2 ; n-1}"} = {form(cote, 3)}",
            f"    Erreur-type: {form(erreur_type, decimales_rep)}",
            f"    Facteur de correction: {"N/A" if facteur_correction == 1 else f"{form(facteur_correction, 3)} car SANS remise et taux de sondage = {form(taux_sondage,1,'%', 'sup')} > 5%"}",
            f"    Marge d'erreur: {form(marge_erreur, decimales_rep)}",
            f"    Intervalle de confiance: [ {form(borne_inf, decimales_ic, sens='plancher')} ; {form(borne_sup, decimales_ic, sens='plafond')} ]"
        ]

        print(*output, sep='\n')

        return {
            'z': cote,
            'erreur_type': erreur_type,
            'facteur_correction': facteur_correction,
            'marge_erreur': marge_erreur,
            'borne_inf': borne_inf,
            'borne_sup': borne_sup
        }
    

    def intervalle_confiance_proportion(self, p, nc=95, decimales=1, remise=True):
        n = self.n
        alpha = 1-nc/100
        z_alpha_2 = norm.ppf(1-alpha/2)

        erreur_type = np.sqrt(p * (1-p) / n)
        taux_sondage = self.n / self.N if self.N is not None else 0
        facteur_correction = 1 if remise or (taux_sondage <= 0.05) else np.sqrt((self.N - self.n) / (self.N - 1))

        marge_erreur = z_alpha_2 * erreur_type * facteur_correction
        borne_inf = p - marge_erreur
        borne_sup = p + marge_erreur

        output = [
            f"Intervalle de confiance pour une proportion de {form(p, 1, '%')} avec un niveau de confiance de {nc}%",
            f"    z_{{alpha/2}} = {form(z_alpha_2, 3)}",
            f"    Erreur-type: {form(erreur_type, decimales + 2)}",
            f"    Facteur de correction: {"N/A" if facteur_correction == 1 else f"{form(facteur_correction, 3)} car SANS remise et taux de sondage = {form(taux_sondage,1,'%', 'sup')} > 5%"}",
            f"    Marge d'erreur: {form(marge_erreur, decimales, '%')}",
            f"    Intervalle: [ {form(borne_inf, decimales, '%', 'plancher')} ; {form(borne_sup, decimales, '%', 'plafond')} ]"
        ]

        print(*output, sep='\n')

        return {
            'z': z_alpha_2,
            'erreur_type': erreur_type,
            'facteur_correction': facteur_correction,
            'marge_erreur': marge_erreur,
            'borne_inf': borne_inf,
            'borne_sup': borne_sup
        }

    # endregion

    # region TESTS HYPOTHESES AVEC VALEUR DE REFERENCE

    def test_hypo_moyenne_reference(self, moyenne_reference, seuil=0.05, lateralite='bilateral', decimales=3):
        if self.qualitative:
            return "Impossible à effectuer le test, puisque la variable doit être quantitative"
        
        h0 = f'la vrai moyenne est la même que la moyenne de référence, soit {moyenne_reference}'
        h1 = f'la moyenne est {['différente de', 'plus petite que', 'plus grande que'][['bilateral', 'gauche', 'droite'].index(lateralite)]} la moyenne de référence de {moyenne_reference}'

        alpha = seuil / 2 if lateralite == 'bilateral' else seuil
        z = (self.moyenne - moyenne_reference) * np.sqrt(self.n) / self.ecart_type
        z_critique = norm.ppf(1-alpha) if self.n >= 30 else t.ppf(1-alpha, self.n - 1)
        
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

        decimales_moyenne = 1 if self.decimales < 1 else self.decimales

        output = [
            f"La moyenne de l'échantillon est: {form(self.moyenne, self.decimales)}",
            f"Valeur critique: {form(z_critique, 3)} obtenue avec la loi {'normale' if self.n >= 30 else ' de Student'}",
            f"La statistique calculée est: {form(z, decimales)}",
            f"Puisque {comparaison}, {decision} voulant que {h0}.",
            f"{conclusion} {h1}."
        ]

        if z < 0 and lateralite == 'droite':
            output.insert(0, "Attention, peut-être que la latéralité devrait être gauche ou bilatérale, au lieu de droite.")
        elif z > 0 and lateralite == 'gauche':
            output.insert(0, "Attention, peut-être que la latéralité devrait être droite ou bilatérale, au lieu de gauche.")

        if self.n < 30:
            output.insert(0, 'Attention, la variable doit sembler suivre une loi normale pour que le test soit valide, puisque n < 30.')

        return '\n'.join(output)


    def test_hypo_proportion_reference(self, p, proportion_reference, seuil=0.05, lateralite='bilateral', decimales=3):
        if self.n < 30:
            return 'Le test est invalide puisque n < 30.'
        elif self.n * p < 5:
            return "Le test est invalide puisque np < 5."
        elif self.n * (1 - p) < 5:
            return "Le test est invalide puisque n(1-p) < 5."

        h0 = f'la vrai proportion est la même que la proportion de référence, soit {form(proportion_reference, 1, '%')}'
        h1 = f'la proportion est {['différente de', 'plus petite que', 'plus grande que'][['bilateral', 'gauche', 'droite'].index(lateralite)]} la moyenne de référence de {form(proportion_reference, 1, '%')}'

        alpha = seuil / 2 if lateralite == 'bilateral' else seuil
        z = (p - proportion_reference) / np.sqrt(p * (1 - p) / self.n)
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
            f"La proportion de l'échantillon est: {form(p, 1, '%')}",
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

    # endregion

# FIN