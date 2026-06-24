# Importación del driver oficial de Neo4j
from neo4j import GraphDatabase

# Importación de la librería pandas para manipulación del CSV
import pandas as pd

# URI de conexión hacia Neo4j mediante el protocolo Bolt
URI = "neo4j://127.0.0.1:7687"

# Credenciales de autenticación de la base de datos
AUTH = ("neo4j", ".L/7?#KCSVW8")

# Creación del driver de conexión con Neo4j
driver = GraphDatabase.driver(URI, auth=AUTH)

# Lectura del archivo CSV que contiene los eventos procesados
df = pd.read_csv("predictions_for_neo4j.csv")

# Reemplazo de valores NaN por cadenas vacías
# para evitar errores durante la carga
df = df.fillna("")

# Conversión del DataFrame a una lista de diccionarios
# para facilitar el procesamiento por lotes
rows = df.to_dict("records")

# Función encargada de crear índices
# para optimizar las consultas Cypher
def create_indexes(session):

    # Índice sobre la propiedad ip del nodo IPAddress
    session.run("""
        CREATE INDEX ip_index IF NOT EXISTS
        FOR (n:IPAddress)
        ON (n.ip)
    """)

    # Índice sobre la propiedad community_id del nodo Connection
    session.run("""
        CREATE INDEX conn_index IF NOT EXISTS
        FOR (c:Connection)
        ON (c.community_id)
    """)

# Función encargada de cargar un lote de registros
# en la base de datos Neo4j
def load_batch(tx, batch):

    tx.run("""

    // Recorrido del lote de registros recibido
    UNWIND $batch AS row

    // Creación o reutilización del nodo IP origen
    MERGE (src:IPAddress {
        ip: toString(row.src_ip_zeek)
    })

    // Asignación de la propiedad que indica
    // si la IP pertenece a la red local
    SET src.is_local = row.local_orig

    // Creación o reutilización del nodo IP destino
    MERGE (dst:IPAddress {
        ip: toString(row.dest_ip_zeek)
    })

    // Asignación de la propiedad local de la IP destino
    SET dst.is_local = row.local_resp

    // Creación o reutilización del nodo Port
    MERGE (p:Port {
        port_number: toString(row.dest_port_zeek),
        service: toString(row.service)
    })

    // Creación del nodo Connection
    // que representa un evento de red
    CREATE (c:Connection {

        // Identificador único de la conexión
        community_id: toString(row.community_id),

        // Dirección IP origen
        src_ip: toString(row.src_ip_zeek),

        // Dirección IP destino
        dest_ip: toString(row.dest_ip_zeek),

        // Puerto origen
        src_port: toString(row.src_port_zeek),

        // Puerto destino
        dest_port: toString(row.dest_port_zeek),

        // Protocolo utilizado
        proto: toString(row.proto),

        // Servicio identificado
        service: toString(row.service),

        // Etiqueta predicha por el modelo
        predicted_label: toString(row.predicted_label),

        // Duración de la conexión
        duration: toFloat(row.duration),

        // Bytes enviados por el origen
        orig_bytes: toFloat(row.orig_bytes),

        // Bytes enviados por el destino
        resp_bytes: toFloat(row.resp_bytes),

        // Paquetes enviados por el origen
        orig_pkts: toFloat(row.orig_pkts),

        // Paquetes enviados por el destino
        resp_pkts: toFloat(row.resp_pkts),

        // Estado de la conexión
        conn_state: toString(row.conn_state),

        // Bytes perdidos o no contabilizados
        missed_bytes: toFloat(row.missed_bytes),

        // Marca temporal del evento
        datetime: toString(row.datetime)

    })

    // Creación o reutilización del nodo AttackPhase
    MERGE (ap:AttackPhase {

        // Etiqueta táctica detectada
        label: toString(row.predicted_label),

        // Táctica asociada al marco MITRE ATT&CK
        mitre_tactic: toString(row.label_tactic)

    })

    // Relación entre la conexión y la IP origen
    CREATE (c)-[:ORIGINATES_FROM]->(src)

    // Relación entre la conexión y la IP destino
    CREATE (c)-[:TARGETS]->(dst)

    // Relación entre la conexión y el puerto utilizado
    CREATE (c)-[:USES_PORT]->(p)

    // Relación de clasificación táctica
    CREATE (c)-[:CLASSIFIED_AS {

        // Etiqueta predicha por el modelo
        predicted_label: toString(row.predicted_label),

        // Etiqueta real del dataset
        real_label: toString(row.real_label)

    }]->(ap)

    """, batch=batch)

# Apertura de una sesión Neo4j
with driver.session() as session:

    # Mensaje informativo
    print("Creando índices...")

    # Creación de índices
    create_indexes(session)

    # Tamaño de lote utilizado para la carga
    batch_size = 1000

    # Mensaje informativo
    print("Cargando datos...")

    # Recorrido del dataset por lotes
    for i in range(0, len(rows), batch_size):

        # Selección del lote actual
        batch = rows[i:i + batch_size]

        # Ejecución de la carga del lote
        session.execute_write(load_batch, batch)

        # Mensaje de progreso
        print(f"Lote {i//batch_size + 1} cargado")

    # Mensaje informativo
    print("Creando relaciones PRECEDES...")

    # Creación de relaciones temporales
    # entre conexiones consecutivas
    session.run("""

    // Obtención de conexiones por IP origen
    MATCH (src:IPAddress)<-[:ORIGINATES_FROM]-(c:Connection)

    // Ordenamiento cronológico
    WITH src, c
    ORDER BY c.datetime

    // Agrupación de conexiones por IP
    WITH src, collect(c) AS connections

    // Recorrido de las conexiones consecutivas
    UNWIND range(0, size(connections)-2) AS i

    // Selección de pares consecutivos
    WITH
        connections[i] AS c1,
        connections[i+1] AS c2

    // Creación de la relación temporal PRECEDES
    CREATE (c1)-[:PRECEDES {

        // Diferencia temporal en segundos
        time_delta: duration.between(
            datetime(c1.datetime),
            datetime(c2.datetime)
        ).seconds

    }]->(c2)

    """)

# Cierre de la conexión con Neo4j
driver.close()

# Mensaje final de confirmación
print("Carga completada.")
