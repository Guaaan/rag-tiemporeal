# Realtime Assistant - README Oficial

## 1. Descripción general

Realtime Assistant es un asistente de voz en tiempo real basado en la API GPT-4o de OpenAI, diseñado para ofrecer experiencias multimodales de conversación natural, integrando voz y texto, con capacidades de búsqueda, grounding y ejecución de herramientas externas. El proyecto está optimizado para ejecutarse tanto localmente como en contenedores Docker y puede integrarse fácilmente con servicios de Azure. Sus principales casos de uso incluyen atención al cliente, educación, traducción en tiempo real y automatización de tareas por voz.

## 2. Arquitectura del Proyecto

- **app.py**: Orquestador principal. Gestiona la interacción con Chainlit, la autenticación con Azure AD, la selección de voz, la lógica de conversación y la integración de herramientas.
- **azure_tts.py**: Cliente para Azure Text-to-Speech. Permite la síntesis de voz en múltiples idiomas y la configuración dinámica de voces.
- **tools.py**: Utilidades para búsqueda en Azure Cognitive Search, integración con Net2Phone, manejo de tokens y logs.
- **realtime/**: Cliente para la API Realtime de OpenAI. Maneja la comunicación WebSocket, el procesamiento de audio y la detección de actividad de voz (VAD).
- **VAD/vad_iterator.py**: Implementación de Voice Activity Detection para segmentar audio en tiempo real y mejorar la experiencia conversacional.

## 3. Configuración del archivo `.env`

Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido y reemplaza los valores según tu configuración:

```
AZURE_OPENAI_API_KEY=XXXX
# Tu clave de API de Azure OpenAI

AZURE_OPENAI_ENDPOINT=wss://xxxx.openai.azure.com/
# Endpoint de Azure OpenAI

AZURE_OPENAI_DEPLOYMENT=gpt-4o-realtime-preview
# Nombre del deployment del modelo GPT-4o

AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION=2024-10-01-preview
# Versión del deployment (por defecto, no es necesario cambiarlo)

AZURE_SEARCH_ENDPOINT=your_azure_search_endpoint
# Endpoint de Azure Cognitive Search

AZURE_SEARCH_KEY=your_azure_search_key
# Clave de Azure Cognitive Search

INDEX_NAME=your_index_name
# Nombre del índice de búsqueda

CHAINLIT_AUTH_SECRET=your_chainlit_auth_secret
# Secreto de autenticación para Chainlit (usa `chainlit create-secret` para generarlo)

AZURE_SPEECH_KEY=your_azure_speech_key
# Clave de Azure Speech (Text-to-Speech)

AZURE_SPEECH_REGION=your_azure_speech_region
# Región de Azure Speech

# Opcionales para autenticación Azure AD
AZURE_CLIENT_ID=your_azure_client_id
AZURE_TENANT_ID=your_azure_tenant_id
AZURE_CLIENT_SECRET=your_azure_client_secret
REDIRECT_URI=https://your-redirect-uri

# Opcionales para Net2Phone
NET2PHONE_CLIENT_ID=your_n2p_client_id
NET2PHONE_API_BASE_URL=https://api.n2p.io
NET2PHONE_USERNAME=tu_usuario
NET2PHONE_PASSWORD=tu_contraseña
```

## 4. Instalación y ejecución

### Requisitos previos
- Suscripción activa de [Azure](https://azure.microsoft.com/en-gb/free/)
- [VS Code](https://code.visualstudio.com/)
- [Docker](https://www.docker.com/)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- [Azure OpenAI](https://azure.microsoft.com/en-us/services/cognitive-services/openai/)
- Python 3.11 o superior
- (Opcional) [Azure Container Registry](https://docs.microsoft.com/en-us/azure/container-registry/)

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Ejecución local

```bash
chainlit run app.py -w
```

Accede a la aplicación en [http://localhost:8000/](http://localhost:8000/)

### Ejecución en Docker

1. Ajusta el archivo `build-docker-image.sh` según la arquitectura de tu máquina (linux/arm64 o linux/amd64).
2. Construye la imagen:
   ```bash
   ./build-docker-image.sh
   ```
3. Ejecuta el contenedor:
   ```bash
   ./run-docker-image.sh
   ```
4. Accede a la aplicación en [http://localhost:8000/](http://localhost:8000/)

### (Opcional) Publicar en Azure Container Registry

1. Actualiza `variables.sh` con los datos de tu registro e imagen.
2. Inicia sesión en Azure:
   ```bash
   az login
   ```
3. Publica la imagen:
   ```bash
   ./push-docker-image.sh
   ```

## 5. Uso general

- Puedes interactuar con el asistente tanto por voz como por texto.
- El sistema soporta cambio de idioma y selección de voz.
- Permite búsquedas en bases de conocimiento y ejecución de herramientas externas (Net2Phone, grounding, etc.).
- Visualiza en tiempo real si el asistente está escuchando o hablando.

### Ejemplo de flujo de conversación
1. El usuario inicia la conversación por voz o texto.
2. El asistente responde usando la voz seleccionada y puede citar fuentes o realizar llamadas si se solicita.
3. El usuario puede pedir búsquedas, grounding o realizar una llamada de emergencia.

### Herramientas disponibles
- **search_knowledge_base**: Busca información en la base de conocimientos.
- **report_grounding**: Devuelve detalles de las fuentes citadas.
- **make_phone_call**: Realiza una llamada al contacto de emergencia usando Net2Phone.

## 6. Estructura de carpetas y archivos

- `app.py`: Lógica principal y punto de entrada.
- `azure_tts.py`: Cliente de síntesis de voz.
- `tools.py`: Utilidades y herramientas externas.
- `realtime/`: Cliente de API Realtime y procesamiento de audio.
- `VAD/`: Detección de actividad de voz.
- `public/`: Archivos estáticos para la interfaz.
- `requirements.txt`: Dependencias Python.
- `Dockerfile`, `compose.yaml`: Contenedores y despliegue.
- Scripts `.sh`: Automatización de build, run y push.

## 7. Contribución y soporte

- Para contribuir, abre un issue o pull request en el repositorio.
- Para soporte, consulta la documentación oficial de Azure y OpenAI, o contacta al mantenedor del proyecto.

---

© 2025 Realtime Assistant. Proyecto open source bajo licencia MIT.
