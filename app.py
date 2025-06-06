import os
import traceback
import asyncio
from openai import AsyncAzureOpenAI
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from uuid import uuid4
from chainlit.logger import logger
from realtime import RealtimeClient
from azure_tts import Client as AzureTTSClient
from tools import search_knowledge_base_handler, report_grounding_handler, tools
from msal import ConfidentialClientApplication
from typing import Optional, Dict

AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID")
AZURE_CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")
AUTHORITY = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}"
SCOPES = ["User.Read"]
REDIRECT_URI = os.environ.get("REDIRECT_URI", "https://itambotdevoz.eastus2.cloudapp.azure.com/auth/oauth/azure-ad/callback")

msal_app = ConfidentialClientApplication(
    AZURE_CLIENT_ID,
    authority=AUTHORITY,
    client_credential=AZURE_CLIENT_SECRET
)

voice = "es-AR-AlloyTurboMultilingualNeural"

VOICE_MAPPING = {
    "english": "en-IN-AnanyaNeural",
    "hindi": "hi-IN-AnanyaNeural",
    "tamil": "ta-IN-PallaviNeural",
    "odia": "or-IN-SubhasiniNeural",
    "bengali": "bn-IN-BashkarNeural",
    "gujarati": "gu-IN-DhwaniNeural",
    "kannada": "kn-IN-SapnaNeural",
    "malayalam": "ml-IN-MidhunNeural",
    "marathi": "mr-IN-AarohiNeural",
    "punjabi": "pa-IN-GurpreetNeural",
    "telugu": "te-IN-MohanNeural",
    "urdu": "ur-IN-AsadNeural"
}

tts_sentence_end = [".", "!", "?", ";", "。", "！", "？", "；", "\n", "।"]


async def setup_openai_realtime(system_prompt: str):
    openai_realtime = RealtimeClient(system_prompt=system_prompt)
    cl.user_session.set("track_id", str(uuid4()))
    voice = VOICE_MAPPING.get(cl.user_session.get("Language"))
    collected_messages = []

    async def handle_conversation_updated(event):
        item = event.get("item")
        delta = event.get("delta")
        if delta:
            if 'audio' in delta:
                audio = delta['audio']
                if not cl.user_session.get("useAzureVoice"):
                    await cl.context.emitter.send_audio_chunk(
                        cl.OutputAudioChunk(mimeType="pcm16", data=audio, track=cl.user_session.get("track_id"))
                    )
            if 'transcript' in delta:
                if cl.user_session.get("useAzureVoice"):
                    chunk_message = delta['transcript']
                    if item["status"] == "in_progress":
                        collected_messages.append(chunk_message)
                        if chunk_message in tts_sentence_end:
                            sent_transcript = ''.join(collected_messages).strip()
                            collected_messages.clear()
                            chunk = await AzureTTSClient.text_to_speech_realtime(text=sent_transcript, voice=voice)
                            await cl.context.emitter.send_audio_chunk(
                                cl.OutputAudioChunk(mimeType="audio/wav", data=chunk, track=cl.user_session.get("track_id"))
                            )

    async def handle_item_completed(item):
        try:
            transcript = item['item']['formatted']['transcript']
            if transcript.strip() != "":
                await cl.Message(content=transcript).send()
        except Exception as e:
            logger.error(f"Failed to generate transcript: {e}")
            logger.error(traceback.format_exc())

    async def handle_conversation_interrupt(event):
        cl.user_session.set("track_id", str(uuid4()))
        collected_messages.clear()
        await cl.context.emitter.send_audio_interrupt()

    async def handle_input_audio_transcription_completed(event):
        delta = event.get("delta")
        if 'transcript' in delta:
            transcript = delta['transcript']
            if transcript:
                await cl.Message(author="You", type="user_message", content=transcript).send()

    async def handle_error(event):
        logger.error(event)

    openai_realtime.on('conversation.updated', handle_conversation_updated)
    openai_realtime.on('conversation.item.completed', handle_item_completed)
    openai_realtime.on('conversation.interrupted', handle_conversation_interrupt)
    openai_realtime.on('conversation.item.input_audio_transcription.completed', handle_input_audio_transcription_completed)
    openai_realtime.on('error', handle_error)

    cl.user_session.set("openai_realtime", openai_realtime)
    await asyncio.gather(*[openai_realtime.add_tool(tool_def, tool_handler) for tool_def, tool_handler in tools])



@cl.oauth_callback
def oauth_callback(
    provider_id: str,  # ID of the OAuth provider (GitHub)
    token: str,  # OAuth access token
    raw_user_data: Dict[str, str],  # User data from GitHub
    default_user: cl.User,  # Default user object from Chainlit
) -> Optional[cl.User]:  # Return User object or None
    """
    Handle the OAuth callback from GitHub
    Return the user object if authentication is successful, None otherwise
    """

    print(f"Provider: {provider_id}")  # Print provider ID for debugging
    print(f"User data: {raw_user_data}")  # Print user data for debugging

    return default_user  # Return the default user object




@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    print("app_user", app_user)

    settings = await cl.ChatSettings([
        Select(
            id="Language",
            label="Choose Language",
            values=list(VOICE_MAPPING.keys()),
            initial_index=0,
        ),
        Switch(id="useAzureVoice", label="Use Azure Voice", initial=False),
        Slider(
            id="Temperature",
            label="Temperature",
            initial=1,
            min=0,
            max=2,
            step=0.1,
        )
    ]).send()
    await setup_agent(settings)


@cl.on_settings_update
async def setup_agent(settings):
    system_prompt = (
        "Eres un asistente de voz de la compañía ITAM. "
        "Antes de responder a cualquier pregunta, busca información relevante en la base de conocimientos interna. "
        "Si no encuentras información relevante, indica que no tienes una respuesta basada en la base de conocimientos. "
        "Responde siempre en el idioma español."
    )
    cl.user_session.set("useAzureVoice", settings["useAzureVoice"])
    cl.user_session.set("Temperature", settings["Temperature"])
    cl.user_session.set("Language", settings["Language"])
    await cl.Message(
        content="Hola Bienvenido al bot conversacional de ITAM. Puedo brindarte información sobre los contactos de emergencia de algún empleado o información general de la compañía. Presiona `P` para hablar! Prueba preguntarme cual es el contacto de emergencias o la dirección de la compañía."
    ).send()
    system_prompt = system_prompt.replace(
        "<customer_language>", settings["Language"])
    await setup_openai_realtime(system_prompt=system_prompt + "\n\n Customer ID: 12121")



@cl.on_message
async def on_message(message: cl.Message):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        search_results = await search_knowledge_base_handler(message.content)
        context = f"Información relevante:\n{search_results}\n\n"
        context += "Por favor, responde basándote en esta información y cita las fuentes usando report_grounding."
        await openai_realtime.send_message(content=message.content, context=context)
@cl.on_audio_start
async def on_audio_start():
    try:
        openai_realtime: RealtimeClient = cl.user_session.get(
            "openai_realtime")
        # TODO: might want to recreate items to restore context
        # openai_realtime.create_conversation_item(item)
        await openai_realtime.connect()
        logger.info("Connected to OpenAI realtime")
        return True
    except Exception as e:
        await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()
        return False


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime:
        if openai_realtime.is_connected():
            await openai_realtime.append_input_audio(chunk.data)
        else:
            logger.info("RealtimeClient is not connected")


@cl.on_logout
def on_logout(request: str, response: str):
    print("🔓 Cerrando sesión local y remota")
    # Limpiar token local si corresponde
    # auth_result.clear()  # Descomenta si usas auth_result para almacenar el token
    # Cerrar sesión en Azure AD
    tenant_id = AZURE_TENANT_ID
    redirect_uri = REDIRECT_URI
    logout_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/logout?post_logout_redirect_uri={redirect_uri}"
    import webbrowser
    webbrowser.open(logout_url)


@cl.on_audio_end
@cl.on_chat_end
@cl.on_stop
async def on_end():
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.disconnect()