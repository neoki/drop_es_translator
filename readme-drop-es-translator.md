# DROP ES Translator: Traductor GPU-Optimizado para DROP

Este script (`drop_es_translator.py`) está diseñado para traducir el dataset DROP de inglés a español utilizando un modelo de traducción neuronal y aceleración GPU.

## Requisitos

- Python 3.7+
- CUDA 12.1+ (para uso de GPU)
- Una GPU NVIDIA compatible con CUDA

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/tu-usuario/tu-repositorio.git
   cd tu-repositorio
   ```

2. Crea un entorno virtual:
   ```
   python -m venv venv
   ```

3. Activa el entorno virtual:
   - En Windows:
     ```
     venv\Scripts\activate
     ```
   - En macOS y Linux:
     ```
     source venv/bin/activate
     ```

4. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

1. Asegúrate de que tu archivo de entrada (por defecto 'drop_dataset_dev.json') esté en el mismo directorio que el script.

2. Ejecuta el script:
   ```
   python drop_es_translator.py
   ```

3. El script generará un archivo de salida con el sufijo '_GPU_Optimized_ES.json'.

## Configuración

- Puedes ajustar el tamaño del batch modificando el parámetro `batch_size` en la función `translate_dataset` dentro de `drop_es_translator.py`.
- Si experimentas problemas de memoria, intenta reducir el tamaño del batch.

## Resultados

El script proporciona:
- Traducciones de los pasajes y preguntas del dataset DROP.
- Logs detallados del proceso de traducción, incluyendo uso de GPU y progreso.
- Un archivo JSON de salida con la estructura original de DROP, pero con el contenido traducido al español.

## Adaptabilidad

Este script está diseñado específicamente para el formato del dataset DROP. Para usarlo con otros datasets o tipos de datos, se necesitarían modificaciones en la estructura de carga y procesamiento de datos.

## Notas

- El rendimiento puede variar dependiendo de tu hardware GPU.
- Asegúrate de tener suficiente espacio en disco para el archivo de salida.
- El proceso puede tardar dependiendo del tamaño del dataset de entrada.

## Solución de problemas

Si encuentras errores relacionados con CUDA o memoria, prueba lo siguiente:
1. Reduce el tamaño del batch en el script.
2. Asegúrate de que no hay otras aplicaciones utilizando la GPU.
3. Verifica que tienes la última versión de los drivers de NVIDIA y CUDA toolkit instalados.

Para cualquier otro problema, revisa los logs detallados que proporciona el script.

## Contacto

Si tienes preguntas o encuentras problemas, por favor abre un issue en este repositorio.
