# Avisos de terceros

Esta publicación conserva lógica científica; los pesos se mantienen sólo localmente. Ultralytics y sus
modelos YOLO utilizan AGPL-3.0 según la licencia upstream y la metadata del modelo
pose; existe una alternativa Enterprise, cuya contratación no se presume.
Se adjunta el texto de licencia de Ultralytics 8.3.203. El repositorio no distribuye pesos ni certifica condiciones para una futura redistribución o servicio público.
Fuente: https://github.com/ultralytics/ultralytics/blob/v8.3.203/LICENSE
Política de modelos: https://www.ultralytics.com/license

Las dependencias se instalan mediante pip; sus propios archivos LICENSE/NOTICE
acompañan sus paquetes. No se incluyen sus binarios ni entornos en esta carpeta.
Identificación obtenida de los metadatos instalados, sin inventar licencias:

| Componente | Versión validada CPU x86 | Licencia declarada |
|---|---|---|
| ultralytics | 8.3.203 | AGPL-3.0 |
| torch | 2.9.1+cpu | BSD-3-Clause |
| torchvision | 0.24.1+cpu | BSD |
| opencv-python | 4.11.0.86 | Apache 2.0 |
| numpy | 1.26.4 | Copyright (c) 2005-2023, NumPy Developers. (texto completo incluido por pip) |
| scipy | 1.17.1 | Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. (texto completo incluido por pip) |
| pillow | 12.3.0 | MIT-CMU |
| fastapi | 0.141.1 | MIT |
| uvicorn | 0.53.0 | BSD-3-Clause |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause |

FFmpeg es una dependencia ejecutable externa: su licencia y opciones de build
deben conservarse con la instalación concreta (imageio-ffmpeg o paquete Ubuntu).
La licencia del wrapper imageio-ffmpeg no sustituye la del ejecutable FFmpeg.

No se distribuyen datasets. No se deduce una licencia adicional de los datasets
de entrenamiento para los checkpoints propios: esa atribución específica no se
revalidó en esta copia. Se conserva el aviso AGPL conocido de sus bases YOLO.
No se afirma una licencia Enterprise ni una autorización nueva sobre datos.
El placeholder de branding procede del artefacto original; no es el logo final.

## Auditoría de publicación v001
Los ocho pesos se excluyen de Git; véase la clasificación individual y justificación
en models/model_manifest.json. La licencia upstream AGPL-3.0 está documentada,
pero no se ha demostrado el cumplimiento de todas las condiciones de redistribución
de esta entrega. No se declara DO_NOT_REDISTRIBUTE sin evidencia de prohibición.
La procedencia oficial de POSE sí está registrada; eso no verifica por sí solo
el cumplimiento de la distribución completa. No se presume licencia Enterprise.
La política upstream consultada es https://www.ultralytics.com/license.
No se asigna una licencia global al proyecto. Su decisión y revisión de compatibilidad
permanecen pendientes; este aviso no autoriza reutilización ni deployment público.
