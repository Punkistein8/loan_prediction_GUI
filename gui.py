import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import csv

from LogisticRegression import obtener_prediccion_logistic_regression
from DecisionTree import obtener_prediccion_decision_tree
from RandomForest import obtener_prediccion_random_forest
from XGBoost import obtener_prediccion_xgboost


def guardar_valores():
    # Obtener los valores de la interfaz gráfica
    valor1 = 'LP00431'
    valor2 = entry2.get()
    valor3 = entry3.get()
    valor4 = entry4.get()
    valor5 = entry5.get()
    valor6 = entry6.get()
    valor7 = entry7.get()
    valor8 = entry8.get()
    valor9 = entry9.get()
    valor10 = entry10.get()
    valor11 = entry11.get()
    valor12 = entry12.get()

    valorModelo = entryModelos.get()

    if not valor2 or not valor3 or not valor4 or not valor5 or not valor6 or not valor7 or not valor8 or not valor9 or not valor10 or not valor11 or not valor12 or not valorModelo:
        messagebox.showerror("Error", "Todos los campos son requeridos")
        return

    try:
        valor7 = float(valor7)
        valor8 = float(valor8)
        valor9 = float(valor9)
    except ValueError:
        messagebox.showerror(
            "Error", "Los campos ApplicantIncome, CoapplicantIncome y LoanAmount deben ser numéricos")
        return

    if valor7 <= 50 or valor8 < 0 or valor9 <= 50:
        messagebox.showerror(
            "Error", "Los campos ApplicantIncome, CoapplicantIncome y LoanAmount deben ser mayores a 50")
        return

    # Crear una lista con los valores
    valores = [valor1, valor2, valor3, valor4, valor5, valor6,
               valor7, valor8, valor9, valor10, valor11, valor12]

    # Abrir el archivo CSV en modo escritura
    with open('testGUI.csv', 'w', newline='') as archivo_csv:
        # Crear el escritor CSV
        escritor_csv = csv.writer(archivo_csv)

        # Escribir los encabezados
        escritor_csv.writerow(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                              'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

        # Escribir los valores en la primera fila
        escritor_csv.writerow(valores)

    # Obtener la prediccion
    predecir_por_modelo(valorModelo)


def predecir_por_modelo(modeloElegido):
    if modeloElegido == 'Regresión Logística':
        prediccion = obtener_prediccion_logistic_regression()
    elif modeloElegido == 'Árbol de decisión':
        prediccion = obtener_prediccion_decision_tree()
    elif modeloElegido == 'Random Forest':
        prediccion = obtener_prediccion_random_forest()
    elif modeloElegido == 'XGBoost':
        prediccion = obtener_prediccion_xgboost()

    print('la prediccion es ', prediccion)
    if prediccion == 'Aprobado':
        messagebox.showinfo(
            "Predicción", "El préstamo será aprobado")
        etiqueta_resultado.config(text=prediccion, bg='green', fg='white')
    else:
        messagebox.showinfo(
            "Predicción", "El préstamo será rechazado")
        etiqueta_resultado.config(text=prediccion, bg='red', fg='white')


# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Sistema de predicción de préstamos")
ventana.geometry("400x720")
espaciado = 6

opcionesModelos = ['Regresión Logística', 'Árbol de decisión', 'Random Forest', 'XGBoost']
labelModelos = tk.Label(ventana, text="Modelo de predicción: ", font=("Arial Bold", 12, 'bold'))
entryModelos = ttk.Combobox(
    ventana, values=opcionesModelos, state='readonly')
labelModelos.pack(pady=(espaciado, 0))
entryModelos.pack(pady=(0, espaciado))

ttk.Separator(ventana, orient=tk.HORIZONTAL).pack(fill=tk.X)

opciones2 = ['Male', 'Female']
label2 = tk.Label(ventana, text="Género: ")
entry2 = ttk.Combobox(ventana, values=opciones2, state='readonly')
label2.pack(pady=(espaciado, 0))
entry2.pack(pady=(0, espaciado))

opciones3 = ['Yes', 'No']
label3 = tk.Label(ventana, text="¿Casado?: ")
entry3 = ttk.Combobox(ventana, values=opciones3, state='readonly')
label3.pack(pady=(espaciado, 0))
entry3.pack(pady=(0, espaciado))

opciones4 = ['0', '1', '2', '3']
label4 = tk.Label(ventana, text="Dependientes: ")
entry4 = ttk.Combobox(ventana, values=opciones4, state='readonly')
label4.pack(pady=(espaciado, 0))
entry4.pack(pady=(0, espaciado))

opciones5 = ['Graduate', 'Not Graduate']
label5 = tk.Label(ventana, text="Educación: ")
entry5 = ttk.Combobox(ventana, values=opciones5, state='readonly')
label5.pack(pady=(espaciado, 0))
entry5.pack(pady=(0, espaciado))

opciones6 = ['Yes', 'No']
label6 = tk.Label(ventana, text="¿Autoempleado?: ")
entry6 = ttk.Combobox(ventana, values=opciones6, state='readonly')
label6.pack(pady=(espaciado, 0))
entry6.pack(pady=(0, espaciado))

label7 = tk.Label(ventana, text="Ingresos mensuales del solicitante: ")
entry7 = tk.Entry(ventana)
label7.pack(pady=(espaciado, 0))
entry7.pack(pady=(0, espaciado))

label8 = tk.Label(ventana, text="Ingresos mensuales del co-solicitante: ")
entry8 = tk.Entry(ventana)
label8.pack(pady=(espaciado, 0))
entry8.pack(pady=(0, espaciado))

label9 = tk.Label(ventana, text="Monto del préstamo: ")
entry9 = tk.Entry(ventana)
label9.pack(pady=(espaciado, 0))
entry9.pack(pady=(0, espaciado))

opciones10 = ['360', '180', '120']
label10 = tk.Label(ventana, text="Plazo del préstamo: ")
entry10 = ttk.Combobox(ventana, values=opciones10, state='readonly')
label10.pack(pady=(espaciado, 0))
entry10.pack(pady=(0, espaciado))

opciones11 = ['1', '0']
label11 = tk.Label(ventana, text="Historial crediticio: ")
entry11 = ttk.Combobox(ventana, values=opciones11, state='readonly')
label11.pack(pady=(espaciado, 0))
entry11.pack(pady=(0, espaciado))

opciones12 = ['Rural', 'Semiurban', 'Urban']
label12 = tk.Label(ventana, text="Zona de la propiedad: ")
entry12 = ttk.Combobox(ventana, values=opciones12, state='readonly')
label12.pack(pady=(espaciado, 0))
entry12.pack(pady=(0, espaciado))


# Crear el botón para guardar los valores
boton_guardar = tk.Button(ventana, text="Consultar", command=guardar_valores)
boton_guardar.pack()

# Etiqueta para la prediccion
etiqueta_resultado = tk.Label(
    ventana, text="Resultado", font=("Arial", 18, 'bold'))
etiqueta_resultado.pack(pady=(espaciado, 0))

# Iniciar el bucle de eventos
ventana.mainloop()
