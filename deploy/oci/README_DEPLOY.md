# OCI ARM64

Destino preparado: Ubuntu 24.04 ARM64, A1.Flex 2 OCPU/12 GB, almacenamiento
persistente. Confirmar elegibilidad Always Free y cuotas antes de provisionar.
La preparación no constituye ejecución nativa ARM64 verificada.

1. Instalar esta carpeta en /opt/ssoma/app. Los pesos ya incluidos pueden usarse
   allí mediante SSOMA_MODEL_DIR=/opt/ssoma/app/models, sin duplicarlos.
2. Ejecutar bootstrap_ssoma.sh sólo en una VM expresamente autorizada, indicando
   SSOMA_MODEL_DIR si se conserva esa ubicación. Crea venv, instala el lock ARM64
   con hashes, configura systemd y verifica modelos. No inicia el servicio.
3. Completar /etc/ssoma/ssoma.env fuera del paquete, con token privado y bind
   loopback. La plantilla y el servicio no contienen secretos. Los paths pueden
   configurarse por env. Conservar permisos de lectura del usuario ssoma.
4. Verificar verify_arm64.py --models y ejecutar un E2E nativo antes de publicar.
   benchmark_oci.py requiere --clip y --reference: ambos archivos de validación
   deben suministrarse externamente. La distribución no contiene videos ni
   resultados históricos. --ui permite probar la interfaz mediante SSH forwarding.
5. Iniciar explícitamente ssoma.service. Mantener SSH restringido; no exponer el
   puerto 8000, SQLite ni los directorios. No se necesita Nginx/Caddy inicialmente.
6. Cloudflare Quick Tunnel es opcional y temporal: instalar cloudflared ARM64
   oficial, verificar su checksum y ejecutar el script sólo tras el benchmark.
   Exige token y marker ARM64 satisfactorio. No garantiza URL estable ni producción.

Persistencia: DATA_DIR/jobs.sqlite3, uploads/results/evidence; modelos de sólo
lectura. Una tarea activa y dos en cola, diez registros y 24 horas de retención.
Cleanup siempre empieza en dry-run, sin tareas de borrado automáticas.
TINY 2s/1MiB; ampliar a STANDARD 5s/2MiB sólo tras medir OCI. El benchmark
externo de 60 frames/2.002s usa una admisión aislada de 2.01s; la demo pública
mantiene 2s. No cambia algoritmos, weights ni thresholds.
