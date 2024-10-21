import pandas as pd

# Cargar el archivo Excel
# Cambia 'ruta_del_archivo.xlsx' por la ruta a tu archivo
archivo_excel = 'dataset/Mapper.xlsx'

# Leer el archivo Excel
df = pd.read_excel(archivo_excel)

# Filtrar solo las columnas necesarias
df_filtrado = df[['Image', 'Constellation']]

# Procesar la columna 'Constellation' para quitar lo que hay después del guion
df_filtrado['Constellation'] = df_filtrado['Constellation'].str.split('–').str[0].str.strip()

# Guardar el resultado en un nuevo archivo Excel
# Cambia 'resultado.xlsx' por el nombre que deseas para el nuevo archivo
df_filtrado.to_excel('resultado.xlsx', index=False)

print("Los datos han sido procesados y guardados en 'resultado.xlsx'")
