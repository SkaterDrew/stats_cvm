import math

def form(nombre, decimales=1, forme='', sens=''):
    if nombre == float('-inf'):
        return ' ou moins'
    elif nombre == float('inf'):
        return ' ou plus'

    if forme == 'e':
        chiffres = len(str(nombre).replace('.', '')) - 1
        mantisse, expo = f"{nombre:.{chiffres}e}".split('e')
        nombre = float(mantisse)
        expo = 'e' + expo

    place = 10**(decimales + 2) if forme == '%' else 10**decimales

    if sens == 'plafond':
        nombre = math.ceil(nombre * place) / place
    elif sens == 'plancher':
        nombre = math.floor(nombre * place) / place
    else:
        nombre = round(nombre * place) / place

    d = 0 if decimales < 0 else decimales

    if forme == '%':
        output = f"{nombre:,.{d}%}"
    elif forme == 'e':
        output = f"{float(str(nombre) + expo):.{d}e}"
    else:
        output = f"{nombre:,.{d}f}"
    return output.replace(',', ' ').replace('.', ',')