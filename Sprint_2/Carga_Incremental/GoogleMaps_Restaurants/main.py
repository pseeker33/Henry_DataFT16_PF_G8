{\rtf1\ansi\ansicpg1252\deff0\nouicompat\deflang3082{\fonttbl{\f0\fnil\fcharset0 Calibri;}}
{\*\generator Riched20 10.0.19041}\viewkind4\uc1 
\pard\sa200\sl276\slmult1\f0\fs22\lang10 from bigquery_uploader import load_df_to_bigquery\par
from restaurantes import obtener_y_guardar_lugares\par
from schema import SCHEMA\par
\par
def main(request):\par
    try:\par
        # PROJECT VARIABLES\par
        BIGQUERY_PROJECT_ID = 'aerial-resolver-237012'\par
        BIGQUERY_DATASET_ID = 'BQ_Google'\par
        BIGQUERY_TABLE_NAME = 'restaurantes_google'\par
        PARTITION_FIELD = 'load_timestamp'\par
        APPEND_DATA_ON_BIGQUERY = True\par
\par
        # Coordenadas y API Key\par
        coordenadas = \{\par
            # ... tus coordenadas ...\par
        \}\par
        api_key = 'tu-api-key'\par
\par
        # Obtener datos de lugares y rese\'f1as\par
        df_lugares, df_resenas_usuarios = obtener_y_guardar_lugares(coordenadas, api_key)\par
\par
        # Verificar si el DataFrame de lugares no est\'e1 vac\'edo\par
        if df_lugares is None or len(df_lugares) == 0:\par
            print('No hay contenido, el DataFrame de lugares est\'e1 vac\'edo')\par
            return ('No hay contenido, el DataFrame de lugares est\'e1 vac\'edo', 204)\par
\par
        # Guardar en Google Big Query\par
        print("Guardando datos en Google Big Query....")\par
        http_status = load_df_to_bigquery(\par
            project_id=BIGQUERY_PROJECT_ID,\par
            dataset_id=BIGQUERY_DATASET_ID,\par
            table_name=BIGQUERY_TABLE_NAME,\par
            df=df_lugares,\par
            schema=SCHEMA,\par
            partition_field=PARTITION_FIELD,\par
            append=APPEND_DATA_ON_BIGQUERY\par
        )\par
\par
        # Verificar el estado HTTP de la operaci\'f3n\par
        if http_status == 200:\par
            return ('\'a1\'c9xito!', http_status)\par
        else:\par
            return ("Error. Por favor, revisa el panel de registro", http_status)\par
\par
    except Exception as e:\par
        error_message = "Error al cargar datos: \{\}".format(e)\par
        print('[ERROR] ' + error_message)\par
        return (error_message, '400')\par
}
 