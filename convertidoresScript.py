from nbconvert import PythonExporter
from nbformat import read, NO_CONVERT


def convertir_ipynb_a_py(ruta_ipynb, ruta_py):
    # Crear un exportador de Python
    exporter = PythonExporter()

    # Leer el archivo .ipynb
    with open(ruta_ipynb, 'r', encoding='utf-8') as archivo_ipynb:
        notebook = read(archivo_ipynb, NO_CONVERT)

    # Convertir el archivo .ipynb a un script .py
    (codigo_python, _) = exporter.from_notebook_node(notebook)

    # Guardar el código Python en un archivo .py
    with open(ruta_py, 'w', encoding='utf-8') as archivo_py:
        archivo_py.write(codigo_python)


# Rutas de entrada y salida
ruta_archivo_ipynb = 'XGBoost.ipynb'
ruta_archivo_py = 'XGBoost.py'

# Convertir el archivo .ipynb a un script .py
convertir_ipynb_a_py(ruta_archivo_ipynb, ruta_archivo_py)
