# Documentación Net2Phone - Integración en Realtime Assistant

Esta guía describe cómo funciona la integración con Net2Phone en el archivo `tools.py` y cómo configurar correctamente las variables de entorno y el flujo para realizar llamadas telefónicas desde el asistente.

## ¿Qué es Net2Phone?
Net2Phone es una plataforma de telefonía que permite realizar llamadas programáticamente mediante su API. En este proyecto, se utiliza para realizar llamadas de emergencia o contacto directo desde el asistente de voz.

## Flujo de autenticación y llamada
1. **Autenticación OAuth2**: Se obtiene un token de acceso usando el método `grant_type=password`.
2. **Caché de token**: El token se almacena en memoria y en disco (`/tmp/n2p_token.json`) para evitar solicitarlo repetidamente.
3. **Realización de llamada**: Se envía una petición POST a la API de Net2Phone con el número y nombre de contacto.

## Variables de entorno necesarias
Asegúrate de definir las siguientes variables en tu archivo `.env`:

```
NET2PHONE_CLIENT_ID=tu_client_id
NET2PHONE_API_BASE_URL=https://api.n2p.io
NET2PHONE_USERNAME=tu_usuario
NET2PHONE_PASSWORD=tu_contraseña
```

- `NET2PHONE_CLIENT_ID`: ID de cliente proporcionado por Net2Phone.
- `NET2PHONE_API_BASE_URL`: URL base de la API (por defecto `https://api.n2p.io`).
- `NET2PHONE_USERNAME`: Usuario autorizado para la API.
- `NET2PHONE_PASSWORD`: Contraseña del usuario.

## Ejemplo de uso en código
La función principal para realizar llamadas es:

```python
async def make_phone_call_handler(phone_number: str, contact_name: str = "Contacto de emergencia") -> Dict[str, Any]:
    # ...ver código en tools.py...
```

- **Parámetros**:
  - `phone_number`: Número de teléfono en formato internacional (ejemplo: +56XXXXXXXXX).
  - `contact_name`: Nombre del contacto (opcional).

## Proceso completo
1. El asistente recibe el número y nombre de contacto.
2. Obtiene el token OAuth2 (usando caché si está vigente).
3. Normaliza el número si es chileno (+56).
4. Realiza la llamada mediante una petición POST a `/v1/calls/`.
5. Devuelve el resultado (éxito o error) y detalles de la llamada.

## Notas de seguridad y operación
- El token se almacena temporalmente en `/tmp/n2p_token.json` para evitar múltiples autenticaciones.
- Asegúrate de proteger las credenciales y el archivo `.env`.
- El número "from" está fijo en el ejemplo (`234802005`), ajusta según tu configuración.

## Ejemplo de configuración en `.env`
```
NET2PHONE_CLIENT_ID=mi_cliente
NET2PHONE_API_BASE_URL=https://api.n2p.io
NET2PHONE_USERNAME=usuario_demo
NET2PHONE_PASSWORD=contraseña_segura
```

## Referencias
- [Net2Phone API Docs](https://www.net2phone.com/)
- Revisa el archivo `tools.py` para detalles técnicos y personalización.
