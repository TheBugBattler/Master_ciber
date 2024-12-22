# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 21:28:34 2024

@author: Pablo D
"""
import pandas as pd

def gestionar_datos():  # gran función donde tendremos el resto
    pd.set_option('display.max_columns', None)  # Configuración para mostrar todas las columnas del DataFrame al imprimirlo
    pd.set_option('display.max_rows', None)  # Configuración para mostrar todas las filas del DataFrame al imprimirlo

    def cargar_fichero():  # Función para cargar el archivo, tenemos que escribir el nombre con la extensión
        nombre_fichero = input("Nombre del archivo (con extensión): ")
        if nombre_fichero.endswith(".csv"): #caso de que el archivo sea csv
            data = pd.read_csv(nombre_fichero)
        elif nombre_fichero.endswith(".xml"): #caso de que el archivo sea xml
            data = pd.read_xml(nombre_fichero)
        elif nombre_fichero.endswith(".json"): #caso archivo json 
            data = pd.read_json(nombre_fichero)
        else:
            print("Formato no soportado.")  # En caso de tener otro formato definimos error 
            return None
        print(f"Archivo '{nombre_fichero}' cargado con {len(data)} registros.")  # Imprimimos el nombre del fichero y el tamaño
        return data

    def imprimir_data(data):  # Función para imprimir el DataFrame completo
        print("\nDatos completos:")
        print(data)

    def filtrar_datos(data):  # Función para aplicar filtros temporales (no se sobreescribe el dataframe original) 
        print("\nColumnas disponibles:", list(data.columns))  # Imprimimos las columnas disponibles en el dataset 
        columna = input("¿Qué columna quieres filtrar? ")  # Preguntamos sobre qué columna queremos aplicar el filtro
        if columna not in data.columns:  # Caso de que la columna no exista ponemos que no existe
            print("Columna no encontrada.")
            return
        print("Filtros: 1) >  2) <  3) =  4) Valor asociado (caso string o booleano)  5) Cancelar")  # Definimos los filtros
        opcion = input("Elige el filtro: ")  # Pedimos qué filtro usar
        if opcion == "5":  # Para cancelar la operación
            return
        valor = input("Valor para el filtro: ")  # Valor para aplicar el filtro

        if opcion == "4":  # Caso string o booleano
            resultado = data[data[columna].astype(str) == valor]  # Convertimos los valores booleanos a cadenas para manejar filtro 4
        elif opcion in {"1", "2", "3"}:
            operadores = {"1": ">", "2": "<", "3": "=="}
            try:
                valor = float(valor)  # Intentamos convertir a número
                resultado = data.query(f"{columna} {operadores[opcion]} @valor")
            except ValueError:
                print("El valor debe ser numérico para este filtro.")
                return
        else:
            print("Filtro no válido.")
            return
        
        if resultado.empty:  # Verificamos si el DataFrame filtrado está vacío
            print("\nNo hay registros que coincidan con el filtro.")
            return
        print("\nDatos después del filtro:")  # Mostramos el DataFrame después de aplicar el filtro
        print(resultado)

        exportar = input("\u00bfDeseas exportar los datos filtrados? (s/n): ") #Añadimos una opcion para elegir si exportar el dataframe con ese filtro aplicado
        if exportar.lower() == "s": #Lo pasamos a minuscula por si lo escribe en mayuscula 
            exportar_datos(resultado) #Exportamos haciendo uso de la funcion que definimos más adelante

    def gestionar_registros(data):  # Función para agregar, eliminar o modificar los datos
        print("\nOpciones: 1) Insertar  2) Borrar  3) Actualizar  4) Volver") #Presentamos las diferentes opciones
        opcion = input("Elige una opción: ")
        if opcion == "4": #Opcion para volver hacia atrás
            return data
        if opcion == "1":  # Insertar nuevo registro
            nuevo = {col: input(f"{col}: ") for col in data.columns} #Creamos diccionario clave columna del dataframe y valor entrada del usuario
            data = pd.concat([data, pd.DataFrame([nuevo])], ignore_index=True) #Añadimos nuevo registro al final del dataframe
            print("\nRegistro añadido. Datos actualizados:")
            print(data) #Imprimimos dataframe actualizado
        elif opcion == "2":  # Borrar registro
            print("\nElige el índice de la fila que deseas borrar:") #Pedimos el índice de la fila del registro que queremos borrar
            print(data.reset_index()) #Imprimimos los índices del dataframe
            indice = int(input("Índice de la fila a borrar: ")) 
            if 0 <= indice < len(data): #Comprobamos que índice está dentro del rango válido 
                data = data.drop(index=indice).reset_index(drop=True) #Eliminamos fila y reorganizamos índices
                print("\nRegistro eliminado. Datos actualizados:")
                print(data) #Imprimimos el dataframe actualizado
            else: #Caso de que el índice esté fuera del rango 
                print("Índice fuera de rango.")
        elif opcion == "3":  # Actualizar un registro
            print("\nElige el índice de la fila que deseas modificar:")#Vamos a pedir índice para la fila que queremos modificar
            print(data.reset_index()) #Mostramos índices 
            indice = int(input("Índice de la fila: ")) #Ingresamos índice de la fila a modificar
            if 0 <= indice < len(data): #Comprobamos que índice esté dentro del rango válido 
                columna = input("¿Qué columna deseas actualizar? ") #Preguntamos qué columna actualizar
                if columna in data.columns: #caso de que la columna se encuentre
                    nuevo_valor = input(f"Nuevo valor para '{columna}': ")  #Pedimos nuevo valor 
                    if data[columna].dtype in ['float64', 'int64']: #Comprobamos si la columna es numérica
                        nuevo_valor = float(nuevo_valor) #Pasamos a float 
                    data.at[indice, columna] = nuevo_valor #actualizamos valor del dataframe
                    print("\nRegistro actualizado. Datos actualizados:")
                    print(data) #Mostramos dataframe actualizado
                else:
                    print("Columna no encontrada.") #Caso de no estar la columna
            else:
                print("Índice fuera de rango.") #Que el índice esté fuera de rango
        return data

    def modificar_estructura(data):  # Función para modificar la estructura del DataFrame (añadir/eliminar columnas)
        print("\nOpciones: 1) Eliminar columna  2) Añadir columna nueva  3) Volver") #Presentamos las opciones disponibles
        opcion = input("Elige una opción: ")
        if opcion == "3": #Volver hacia atrás
            return data
        if opcion == "1": #Caso eliminar columna
            print("\nColumnas disponibles:", list(data.columns)) #Mostramos las columnas disponibles
            columna = input("Nombre de la columna a eliminar: ") #nombre de la columna que queremos eliminar 
            if columna in data.columns:
                data = data.drop(columns=[columna]) #Eliminamos la columna 
                print(f"\nColumna '{columna}' eliminada. Datos actualizados:")
                print(data) #Mostramos el dataframe actualizado
            else:
                print("Columna no encontrada.") #caso de que no esté la columna
        elif opcion == "2": #Caso añadir nueva columna con datos vacíos 
            nueva_columna = input("Nombre de la nueva columna: ") 
            if nueva_columna not in data.columns: #en caso de que no exista ya la columna a crear
                data[nueva_columna] = None #añadimos columna con valores none
                print(f"\nColumna '{nueva_columna}' añadida con valores vacíos. Datos actualizados:")
                print(data) #mostramos el dataframe actualizado
            else: #caso de que la columna ya exista
                print("La columna ya existe.")
        return data

    def exportar_datos(data):  # Función para exportar el fichero en el formato deseado
        print("\nFormatos: 1) CSV  2) XML  3) JSON  4) Volver") #mostramos las extensiones disponibles
        opcion = input("Formato de salida: ") 
        if opcion == "4":#volver hacia atrás 
            return
        nombre = input("Nombre del archivo (sin extensión): ") #Pedimos el nombre del archivo sin la extensión 
        if opcion == "1": #caso csv
            data.to_csv(nombre + ".csv", index=False)
        elif opcion == "2": #casp xml
            data.to_xml(nombre + ".xml", index=False)
        elif opcion == "3": #caso json, formato orient 
            data.to_json(nombre + ".json", orient="records")
        print(f"\nArchivo guardado como {nombre}.")

    # Flujo principal
    data = cargar_fichero() 
    if data is None: #caso de que no exista 
        return
    print("\nDatos cargados:") 
    print(data) #imprimimos datos
    while True: #Iniciamos bucle para el menú principal del programa
        print("\nOpciones: 1) Filtrar  2) Gestionar registros  3) Modificar estructura  4) Imprimir datos  5) Exportar  6) Salir") #Mostramos las opciones disponibles 
        opcion = input("Elige una opción: ") #Asociamos las diferentes funciones que hemos definido con un valor numérico
        if opcion == "1":
            filtrar_datos(data)
        elif opcion == "2":
            data = gestionar_registros(data)
        elif opcion == "3":
            data = modificar_estructura(data)
        elif opcion == "4":
            imprimir_data(data)
        elif opcion == "5":
            exportar_datos(data)
        elif opcion == "6":
            print("El programa se ha cerrado.") #Cierre del programa 
            break
        else:
            print("Opción no válida.") #Caso de error al elegir opción 

gestionar_datos() #cargamos función principal para ejecutar programa



