"""
Agente con Google ADK. Tiene una sola herramienta, la búsqueda en la documentación, y la instrucción
de responder solo con lo que encuentre y citar la fuente. Si no está en los documentos, lo dice.
"""

from __future__ import annotations

import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from . import config
from .rag import Indice

APP = "agente_obra"
USUARIO = "web"

INSTRUCCION = """Sos un asistente técnico para arquitectos, constructoras e instaladores que eligen
iluminación LED para una obra. Respondés en español rioplatense, claro y concreto.

Reglas:
1. Antes de responder sobre productos, especificaciones, precios, garantía o envíos, usá SIEMPRE la
   herramienta buscar_documentacion. Podés llamarla más de una vez con consultas distintas.
2. Respondé solo con datos que aparezcan en los fragmentos. No inventes potencias, grados de
   protección, precios ni plazos. Si algo no está, decilo y sugerí consultar con un asesor.
3. Si la pregunta compara productos o pide recomendar uno para un uso (piscina, fachada, camino,
   jardín), explicá el criterio: grado IP, potencia, flujo luminoso y material.
4. Si usaste datos de los documentos, cerrá con una línea "Fuentes:" y los nombres de archivo exactos
   (por ejemplo piscinas-piscina-1400.md) de los que sacaste datos. Si no encontraste nada útil, no
   pongas fuentes.
5. Los precios cambian: aclaralo cuando menciones uno."""


def crear(indice: Indice) -> "Agente":
    async def buscar_documentacion(consulta: str) -> dict:
        """Busca en las fichas técnicas y políticas comerciales los fragmentos más relevantes.

        Args:
            consulta: lo que hay que encontrar, por ejemplo "embutido de piso IP67 potencia".

        Returns:
            Un diccionario con la lista de fragmentos encontrados, cada uno con su fuente y texto.
        """
        resultados = await indice.buscar(consulta)
        return {"fragmentos": [{"fuente": r["fuente"], "titulo": r["titulo"], "texto": r["texto"]}
                               for r in resultados]}

    agente = LlmAgent(
        name="asistente_de_obra",
        model=config.MODELO,
        description="Responde consultas técnicas de iluminación LED con la documentación del fabricante.",
        instruction=INSTRUCCION,
        tools=[buscar_documentacion],
        # respuestas de consulta técnica: sin "pensamiento" extendido baja la latencia de ~35 s a pocos segundos
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2, thinking_config=types.ThinkingConfig(thinking_budget=0)),
    )
    return Agente(agente)


class Agente:
    def __init__(self, agente: LlmAgent):
        self.sesiones = InMemorySessionService()
        self.runner = Runner(app_name=APP, agent=agente, session_service=self.sesiones)

    async def preguntar(self, pregunta: str, sesion: str | None = None) -> dict:
        sesion = sesion or uuid.uuid4().hex
        if not await self.sesiones.get_session(app_name=APP, user_id=USUARIO, session_id=sesion):
            await self.sesiones.create_session(app_name=APP, user_id=USUARIO, session_id=sesion)

        mensaje = types.Content(role="user", parts=[types.Part(text=pregunta)])
        respuesta, fuentes, busquedas = "", [], []
        async for evento in self.runner.run_async(user_id=USUARIO, session_id=sesion, new_message=mensaje):
            for llamada in evento.get_function_calls() or []:
                busquedas.append((llamada.args or {}).get("consulta", ""))
            for resp in evento.get_function_responses() or []:
                for f in (resp.response or {}).get("fragmentos", []):
                    if f["fuente"] not in fuentes:
                        fuentes.append(f["fuente"])
            if evento.is_final_response() and evento.content and evento.content.parts:
                respuesta = "".join(p.text or "" for p in evento.content.parts).strip()
        # solo cuentan como fuentes los documentos que la respuesta efectivamente cita
        citadas = [f for f in fuentes if f in respuesta]
        return {"respuesta": respuesta, "fuentes": citadas, "busquedas": busquedas, "sesion": sesion}
