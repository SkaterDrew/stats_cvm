import pandas as pd
from pandas import ExcelWriter

def split_date_heure(df, date_col):
    """
    Split the datetime column into separate date and time columns.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    column_index = df.columns.get_loc(date_col)
    df.insert(column_index + 1, 'Date', df[date_col].dt.date)
    df.insert(column_index + 2, 'Heure', df[date_col].dt.time)
    return df.drop(columns=[date_col])


def creer_fichier_avec_filtres(df, filtre, xlsx_file, date_time_col=None, filtre_moyennes=None):
    """
    Create a new Excel file with separate sheets for each unique value in the specified column.
    Each sheet will contain the rows of the original DataFrame that correspond to that unique value.
    """
    if date_time_col is not None and date_time_col in df.columns:
        df = split_date_heure(df, date_time_col)

    # Get the unique values in the specified column
    elements = df[filtre].unique()
    # Create a new filtered DataFrame for each unique value
    df_elements = {element: df[df[filtre] == element] for element in elements}
    # If a filter for averages is provided, apply it
    if filtre_moyennes is not None:
        for element in elements:
            df_elements[element] = grouper_avec_moyennes(df_elements[element], filtre_moyennes)
    # Create a new Excel file with separate sheets for each unique value
    filtered_file = xlsx_file.replace('.xlsx', f'_{filtre}.xlsx')
    with ExcelWriter(filtered_file, engine="openpyxl") as writer:
        for element in elements:
            # Write each DataFrame to a separate sheet
            df_elements[element].to_excel(writer, sheet_name=element, index=False)
        # Save the file (automatically handled by the context manager)
    print(f"Le fichier {filtered_file} a été créé avec succès.")
    return df_elements


def grouper_avec_moyennes(df, filtre):
    """
    Grouper les données selon les valeurs unique du filtre et prendre la moyenne pour chacune des variables quantitatives, pour chaque valeur.
    """
    return df.groupby(filtre).mean(numeric_only=True).reset_index()