# Sistema Inteligente SSOMA
Visión Artificial · XAI · Riesgo Inteligente

SSOMA_RUNTIME_v001 conserva los algoritmos y UI de SSOMA_APPLICATION_RC1.
No incluye datasets, experimentos, historiales, entornos ni secretos.

## Requisitos
Python 3.12, Linux CPU x86_64 o Ubuntu ARM64. En Windows, el launcher usa WSL2
y el entorno Python Linux; no se declara soporte Windows nativo. ARM64 tiene
dependencias auditadas, pero ejecución nativa pendiente. Sin GPU obligatoria.

## Instalación
Desde esta carpeta en Linux/WSL:
```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.deploy.txt
```
Para ARM64 usar deploy/oci/requirements.arm64.lock.txt con --require-hashes.
El entorno se crea después de instalar; no forma parte de esta distribución.
Puede reutilizar un entorno externo compatible mediante SSOMA_PYTHON.

## Inicio rápido
Linux: `bash start_ssoma.sh`. Windows: `./start_ssoma.ps1`.
Windows con entorno Linux externo: `./start_ssoma.ps1 -LinuxPython /ruta/venv/bin/python`.
Si el entorno pertenece a otra distribución WSL, agregar `-Distribution Ubuntu-24.04`
o el nombre correspondiente; sin ese parámetro se usa la distribución predeterminada.
Abra http://127.0.0.1:8000. Copiar .env.example a .env es opcional.
Para verificación sin análisis: `python scripts/self_check.py`, o
`./start_ssoma.ps1 -SelfCheck`. No cambia pesos ni deja resultados de prueba.

## Uso
Nuevo análisis permite cargar un video. Al finalizar están disponibles Monitor,
Análisis, Dashboard, evidencias temporales y explicación XAI. La cola admite un
análisis activo. Resultados científicos son apoyo para revisión SSOMA; no se
declaran accidente, incumplimiento ni exactitud final. Goggles/boots mantienen
SUPPORT_ONLY. No se sustituye la validación experta pendiente.

## Estructura
- web/: interfaz y presentación originales, sin rediseño.
- src/detection/: inferencia, tracking/asociación, eventos, pose, fuzzy y XAI.
- deployment/adapter_v001/: serving, seguridad, cola y persistencia.
- config/: parámetros científicos intactos y registry relativo a MODEL_DIR.
- models/: manifiestos de siete modelos, ocho archivos requeridos localmente; gloves necesita además su
  estado de arquitectura. El baseline compartido se guarda una sola vez.
- data/uploads y data/results: inicialmente vacíos, almacenamiento generado.
- branding/: placeholder original; PNG definitivos pendientes.
- scripts/: arranque/verificación; deploy/: administración OCI y túnel.

## Variables de entorno
SSOMA_DATA_DIR, SSOMA_MODEL_DIR, SSOMA_UPLOAD_DIR y SSOMA_RESULT_DIR aceptan rutas
relativas a esta raíz o absolutas del host. SSOMA_DEVICE=auto/cpu; CPU por defecto
si no hay acelerador compatible. El launcher no depende del proyecto de origen.
SSOMA_ACCESS_TOKEN es opcional sólo para localhost, obligatorio al publicar.
La API admite Bearer; el navegador, Basic con cualquier usuario y token como
contraseña. Guardar el secreto externamente; no añadirlo a este paquete.

## Límites de demo
La plantilla selecciona TINY=2 s/1 MiB; STANDARD=5 s/2 MiB y LOCAL_FULL=600 s/250
MiB requieren configurar límites conscientemente. Sin .env se conservan los
límites locales del adaptador: 10 s/25 MiB. Persisten hasta 10 análisis, 24 horas,
un activo y dos en cola. No hay limpieza automática: use el comando cleanup del
adaptador en modo dry-run antes de autorizar --apply. DELETE protege entradas
activas y rutas ajenas.

## Deployment
deploy/oci/README_DEPLOY.md describe bootstrap, systemd, modelo ARM64 y benchmark.
No se ha desplegado ni iniciado un túnel. Guardar datos en volumen persistente.
El registry usa filenames relativos a SSOMA_MODEL_DIR (./models por defecto).
Las firmas de pesos y fuentes científicas se comprueban al arrancar.
distribution_manifest.csv enumera cada archivo publicado y razón; su propia fila excluye
el SHA256 para evitar una referencia circular. Licencias: THIRD_PARTY_NOTICES.md.

## Arquitectura y características
Video offline → detección PPE/contexto → tracking y asociación por persona →
evidencia temporal y pose → eventos → riesgo fuzzy → explicación XAI y revisión
visual. FastAPI sirve la interfaz, SQLite conserva la cola y un worker aislado
procesa un análisis a la vez. No hay entrenamiento en este repositorio.

## Modelos requeridos localmente
**Este repositorio no incluye pesos y no puede analizar videos sin ellos.**
Los ocho archivos están clasificados PUBLIC_REDISTRIBUTION_UNCLEAR y marcados
DISTRIBUTION_REQUIRED_LOCALLY. No se afirma que su distribución esté prohibida:
falta verificar la autorización completa para esta publicación.

Coloque copias obtenidas legítimamente de los checkpoints exactos en `models/`
(o en SSOMA_MODEL_DIR). No sustituya pesos ni cambie parámetros para pasar firmas:

| Archivo requerido | Ubicación predeterminada |
|---|---|
| PERSON_B.pt | models/PERSON_B.pt |
| VEST_B.pt | models/VEST_B.pt |
| GLOVES_B.pt | models/GLOVES_B.pt |
| GLOVES_B_architecture.pt | models/GLOVES_B_architecture.pt |
| PPE_BASELINE.pt | models/PPE_BASELINE.pt |
| HAZARD.pt | models/HAZARD.pt |
| CONTEXT.pt | models/CONTEXT.pt |
| POSE.pt | models/POSE.pt |

Los tamaños y SHA256 esperados están en models/model_manifest.json; el arranque
los verifica. La procedencia oficial de pose está en models/pose_v002/source_metadata.json.
No se descargan pesos automáticamente. El estado de arquitectura de gloves es
obligatorio. Hasta resolver licencias no se publica GitHub Release.

## Seguridad
No versionar .env, claves, tokens, uploads, resultados ni bases SQLite. Mantener
el servidor en loopback por defecto; acceso externo requiere autenticación y TLS
mediante la configuración de deployment. Definir un token propio fuera de Git.
Una revisión de secretos no equivale a una auditoría de seguridad exhaustiva.

## Estado científico y licencia
Aplicación funcional SSOMA_APPLICATION_RC1; selección de desarrollo, sin claim
final GOLD/TEST. Boots y goggles son SUPPORT_ONLY. La accuracy del FallEngine no
ha sido evaluada formalmente; FALL_CANDIDATE no confirma un accidente. La validación
experta del motor fuzzy sigue pendiente. Es apoyo a decisión SSOMA y revisión humana,
no certificación de cumplimiento. Procesamiento offline, no vigilancia en tiempo real.

La licencia global del proyecto sigue sin decidirse. Publicar el código no concede
una licencia global de reutilización ni resuelve las obligaciones de dependencias.
Consultar THIRD_PARTY_NOTICES.md antes de redistribuir o desplegar públicamente.

## Verificación del repositorio sin pesos
`python scripts/public_repo_check.py` comprueba estructura, sintaxis, hashes y
exclusiones sin cargar modelos. No sustituye `python scripts/self_check.py`, que
requiere los ocho pesos y el entorno Linux compatible. La validación ARM64 nativa
sigue pendiente.
