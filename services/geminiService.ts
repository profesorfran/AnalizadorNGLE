
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import type { SentenceAnalysis } from '../types';

let ai: GoogleGenAI | null = null;

// Attempt to initialize AI client if API_KEY is present in the environment
if (process.env.API_KEY) {
  try {
    ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  } catch (e) {
    console.error("Failed to initialize Gemini Client. The API key might be malformed.", e);
    ai = null; // Ensure ai is null if initialization fails.
  }
}

/**
 * Checks if the Gemini API has been configured and initialized.
 * @returns {boolean} True if the API is ready, false otherwise.
 */
export const isApiKeyConfigured = (): boolean => {
  return !!ai;
};


const MODEL_NAME = "gemini-2.5-flash";

// Helper function for retries with exponential backoff
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

function parseGeminiResponse(responseText: string): SentenceAnalysis | null {
  let jsonStr = responseText.trim();
  // Regex to remove markdown fences (```json ... ``` or ``` ... ```)
  const fenceRegex = /^```(?:json)?\s*\n?(.*?)\n?\s*```$/si;
  const match = jsonStr.match(fenceRegex);
  if (match && match[1]) {
    jsonStr = match[1].trim();
  }

  try {
    const parsedData = JSON.parse(jsonStr);
    // Basic validation of structure format
    const isValidStructure = (elements: any[]): boolean => {
      return elements.every(el => 
        typeof el.text === 'string' &&
        typeof el.label === 'string' &&
        (!el.children || (Array.isArray(el.children) && isValidStructure(el.children)))
      );
    };

    if (parsedData && 
        typeof parsedData.fullSentence === 'string' && 
        typeof parsedData.classification === 'string' && 
        Array.isArray(parsedData.structure) &&
        isValidStructure(parsedData.structure)
    ) {
       return parsedData as SentenceAnalysis;
    }
    console.error("Parsed JSON does not match expected SentenceAnalysis structure or is invalid:", parsedData);
    return null;
  } catch (e) {
    console.error("Failed to parse JSON response:", e);
    console.error("Problematic JSON string that failed to parse:", jsonStr);
    return null;
  }
}

export const analyzeSentence = async (sentence: string): Promise<SentenceAnalysis | null> => {
  if (!ai) {
    throw new Error("La clave API de Gemini no ha sido configurada correctamente en el entorno de la aplicación.");
  }

  const prompt = `
Analiza sintácticamente la siguiente oración en español según los principios de la Nueva Gramática de la Lengua Española (NGLE) y proporciona la estructura en formato JSON. La oración es: '${sentence}'.

**OBJETIVO GENERAL:**
Producir un árbol sintáctico que refleje la estructura gramatical de la oración, identificando sintagmas, sus núcleos, funciones y las relaciones entre ellos, especialmente en oraciones compuestas y complejas.

**FORMATO JSON REQUERIDO:**
El objeto raíz debe tener:
- 'fullSentence': La oración original.
- 'classification': Clasificación detallada de la oración (ej: "Oración Simple, Enunciativa Afirmativa, Predicativa, Activa, Transitiva", "Oración Compuesta Coordinada Copulativa", "Oración Compleja Subordinada Sustantiva de CD").
- 'structure': Un array de elementos sintácticos. Generalmente, para una oración simple o la principal de una compleja/compuesta, este array contendrá dos elementos principales: 'SN Sujeto' y 'SV - Predicado verbal' (o 'SV - Predicado nominal'). Si hay un Sujeto Tácito, se incluirá un nodo 'ST'. Si hay varias oraciones coordinadas principales, cada una será un objeto 'Prop - Coordinada [Tipo]' en este array.

**ELEMENTO SINTÁCTICO ('SyntacticElement'):**
Cada elemento en 'structure' (y sus 'children') debe ser un objeto con:
- 'text': El fragmento de texto exacto de la oración que corresponde a este elemento. Para elementos como 'Sujeto Tácito', el texto puede ser "(ST)", "(Yo)", "(Él/Ella)", etc.
- 'label': La etiqueta gramatical del elemento (ver GUÍA DE ETIQUETAS más abajo).
- 'children': (Opcional) Un array de 'SyntacticElement' que son constituyentes de este elemento. Si no tiene hijos (es un elemento terminal o una palabra), omite esta propiedad o usa un array vacío.

**GUÍA DE ETIQUETAS (NGLE):**
Usa estas etiquetas de forma precisa. El primer nivel de 'structure' suele ser el Sujeto y Predicado de la oración principal.

1.  **Nivel Oracional Principal:**
    *   'SN Sujeto': Sintagma Nominal Sujeto.
    *   'SV - Predicado verbal': Sintagma Verbal Predicado Verbal.
    *   'SV - Predicado nominal': Sintagma Verbal Predicado Nominal (con verbos copulativos).
    *   'ST': Para indicar Sujeto Tácito. El 'text' puede ser "(ST)", "(Yo)", "(Nosotros)", "(Él/Ella)", etc., reflejando el sujeto omitido. Este nodo NO debe tener hijos y se coloca al mismo nivel que el predicado.

2.  **Sintagmas (Tipos y Funciones):**
    *   **Tipos de Sintagmas (como constituyentes):**
        *   'SN': Sintagma Nominal.
        *   'SAdj': Sintagma Adjetival.
        *   'SAdv': Sintagma Adverbial.
        *   'SPrep': Sintagma Preposicional.
    *   **Núcleos de Sintagmas (etiquetados con '(N)'):**
        *   'N (N)': Nombre (Núcleo de SN). Ejemplo: \`{"text": "libro", "label": "N (N)"}\`.
        *   'V (N)': Verbo (Núcleo de SV). Debe usarse esta etiqueta para el núcleo verbal. Ejemplo: \`{"text": "come", "label": "V (N)"}\`.
        *   'Adj (N)': Adjetivo (Núcleo de SAdj). Ejemplo: \`{"text": "grande", "label": "Adj (N)"}\`.
        *   'Adv (N)': Adverbio (Núcleo de SAdv). Ejemplo: \`{"text": "rápidamente", "label": "Adv (N)"}\`.
        *   'Prep (N)': Preposición (Núcleo de SPrep). Debe usarse esta etiqueta para la preposición. Ejemplo: \`{"text": "en", "label": "Prep (N)"}\`.
        *   'Pron (N)': Pronombre (Núcleo de SN, si se decide usar una etiqueta específica para pronombre núcleo, si no, 'N (N)' puede aplicar si el texto es un pronombre y actúa como tal). Para simplificar, priorizar 'N (N)' para núcleos nominales y 'Pron' para pronombres que no son núcleo de un SN mayor o tienen función propia.
    *   **Determinantes y Nexos (generalmente palabras solas):**
        *   'Det': Determinante.
        *   'nx': Nexo coordinante o subordinante que NO cumple otra función sintáctica.
        *   'PronRel': Pronombre Relativo (nexo + función dentro de la subordinada). Ejemplo: \`"text": "que", "label": "PronRel (Sujeto)"\` si 'que' es sujeto en la relativa.
        *   'AdvRel': Adverbio Relativo (nexo + función o CC). Ejemplo: \`"text": "donde", "label": "AdvRel (CCLugar)"\`.
        *   'DetRel': Determinante Relativo/Posesivo Relativo Cuyo (nexo + función Det). Ejemplo: \`"text": "cuyo", "label": "DetRel"\`.
    *   **Funciones Sintácticas (etiquetar el sintagma completo que cumple la función):**
        *   **Complemento Directo (CD):** Puede ser 'SN - CD' o 'SPrep - CD'.
            *   'SN - CD': Cuando el CD es un Sintagma Nominal.
            *   'SPrep - CD': Cuando el CD es un Sintagma Preposicional introducido por "a". En este caso, el SPrep tendrá un hijo \`{"text": "a", "label": "Prep (N)"}\` y otro hijo que será el Término (generalmente 'SN - Término').
        *   'SN - CI', 'SPrep - CI'.
        *   'SN - Atrib', 'SAdj - Atrib', 'SPrep - Atrib'.
        *   'SN - CPred', 'SAdj - CPred'.
        *   'SPrep - CRég' (Complemento de Régimen).
        *   'SPrep - CC de [Tipo]': Sintagma Preposicional C. Circunstancial (Lugar, Tiempo, Modo, Causa, Finalidad, Compañía, Instrumento, Cantidad, etc.). Ejemplo: \`"label": "SPrep - CC de Lugar"\`. El núcleo interno será \`Prep (N)\` y su término.
        *   'SAdv - CC de [Tipo]': Sintagma Adverbial C. Circunstancial. Ejemplo: \`"label": "SAdv - CC de Modo"\`. Si es una sola palabra (ej. "así"), su estructura interna será \`{"text": "así", "label": "Adv (N)"}\`.
        *   'SN - CC de [Tipo]': Sintagma Nominal C. Circunstancial (ej. "Esta mañana"). Su estructura interna contendrá \`Det\` y \`N (N)\`.
        *   'SPrep - CAg' (Complemento Agente).
        *   'SPrep - CN' (Complemento del Nombre, realizado por SPrep).
        *   'SAdj - CN' (Complemento del Nombre, cuando un SAdj modifica directamente a un N (N). No usar "Adyacente").
        *   'SPrep - CAdj' (Complemento del Adjetivo, realizado por SPrep).
        *   'SPrep - CAdv' (Complemento del Adverbio, realizado por SPrep).
        *   'SN - Voc' (Vocativo).
        *   **Término de SPrep:** El sintagma que sigue a la preposición dentro de un SPrep. El nodo de la preposición será 'Prep (N)'.
            *   'SN - Término' (con núcleo interno 'N (N)')
            *   'SAdj - Término' (con núcleo interno 'Adj (N)')
            *   'SAdv - Término' (con núcleo interno 'Adv (N)')
            *   'Oración - Subordinada Sustantiva de Término'

3.  **Oraciones Compuestas (Coordinadas):**
    *   'Prop - Coordinada [Tipo]': Cada proposición coordinada.
        *   'Tipo': Copulativa, Disyuntiva, Adversativa, Distributiva, Explicativa.
        *   El 'nx' (nexo) debe estar fuera de las proposiciones.
        *   Ejemplo de estructura para "Ana canta y Pedro baila":
          \`[ {"text": "Ana canta", "label": "Prop - Coordinada Copulativa", ...}, {"text": "y", "label": "nx"}, {"text": "Pedro baila", "label": "Prop - Coordinada Copulativa", ...} ]\`

4.  **Oraciones Complejas (Subordinadas):**
    *   **Subordinadas Sustantivas:**
        *   'Oración - Subordinada Sustantiva de Sujeto', '... de CD', '... de Término', '... de Atributo'.
        *   El 'nx' (que, si) es un hijo de la oración subordinada.
        *   Una 'Oración - Subordinada Sustantiva de CD' es un tipo de CD.
    *   **Subordinadas Relativas:**
        *   'Oración - Subordinada Relativa Especificativa', '... Explicativa'.
        *   Deben ser hijas del SN cuyo núcleo (\`N (N)\`) es el antecedente.
        *   El pronombre/adverbio/determinante relativo ('PronRel', 'AdvRel', 'DetRel') es parte de la subordinada relativa y cumple una función DENTRO de ella.
    *   **Subordinadas Construcciones (antes Adverbiales):**
        *   'Oración - Subordinada Construcción de [Tipo]': Para construcciones (antes adverbiales).
            *   'Tipo': Tiempo, Lugar, Modo, Causa, Finalidad, Concesión, Condición, Consecutiva, Comparativa, Ilativa, Superlativa.
            *   **IMPORTANTE:** Estas construcciones subordinadas deben ser hijas DIRECTAS del nodo 'SV - Predicado verbal' (o 'SV - Predicado nominal' si aplica) de la oración principal que modifican. No deben ser hijas de otros sintagmas dentro del predicado, a menos que la gramática lo exija explícitamente para un tipo muy específico de construcción anidada (lo cual es raro para estas).
            *   El 'nx' es un hijo de la oración subordinada construcción.

5.  **Otras Etiquetas:**
    *   'Interj': Interjección.
    *   'Perífrasis Verbal': Etiquetar el conjunto como 'V (N)' y en 'text' poner la perífrasis completa. Ej: \`{"text": "va a llover", "label": "V (N)"}\`.

**CONSIDERACIONES IMPORTANTES:**
- **Jerarquía:** Los hijos de un nodo deben ser sus constituyentes directos.
- **Núcleos (N):** Asegúrate de que los núcleos de los sintagmas (SN, SAdj, SAdv, SPrep, SV) se identifiquen consistentemente con la etiqueta \`(N)\` apropiada: \`N (N)\`, \`Adj (N)\`, \`Adv (N)\`, \`Prep (N)\`, \`V (N)\`.
- **Texto Completo:** La concatenación de los 'text' de los nodos terminales debe reconstruir la oración.

**Corrección al EJEMPLO DE SALIDA (para "El perro que ladra no muerde"):**
El N (N) no debe tener como hijo a la Oración Relativa. La Oración Relativa es un modificador del Nombre, por tanto, es hermana del N (N) dentro del SN.
\`\`\`json
{
  "fullSentence": "El perro que ladra no muerde",
  "classification": "Oración Compleja, Enunciativa Negativa, Predicativa, Activa, Intransitiva, con Subordinada Relativa Especificativa",
  "structure": [
    {
      "text": "El perro que ladra",
      "label": "SN Sujeto",
      "children": [
        { "text": "El", "label": "Det" },
        { "text": "perro", "label": "N (N)" },
        {
          "text": "que ladra",
          "label": "Oración - Subordinada Relativa Especificativa", 
          "children": [
            { "text": "que", "label": "PronRel (Sujeto)" },
            {
              "text": "ladra",
              "label": "SV - Predicado verbal",
              "children": [
                { "text": "ladra", "label": "V (N)" }
              ]
            }
          ]
        }
      ]
    },
    {
      "text": "no muerde",
      "label": "SV - Predicado verbal",
      "children": [
        {
          "text": "no",
          "label": "SAdv - CC de Negación",
          "children": [
            { "text": "no", "label": "Adv (N)" }
          ]
        },
        { "text": "muerde", "label": "V (N)" }
      ]
    }
  ]
}
\`\`\`

Proporciona SOLO el objeto JSON como respuesta. No incluyas explicaciones adicionales fuera del JSON.
`;

  console.log("Gemini API Prompt length:", prompt.length);

  const MAX_RETRIES = 3;
  let attempt = 0;
  let lastError: any = null;

  while (attempt < MAX_RETRIES) {
    try {
      const response: GenerateContentResponse = await ai.models.generateContent({
        model: MODEL_NAME,
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          // temperature: 0.3 (opcional, para mayor determinismo si es necesario)
        },
      });

      const responseText = response.text;
      if (!responseText) {
          console.error("Gemini response text is empty on attempt " + (attempt + 1));
          return null; 
      }
      return parseGeminiResponse(responseText);

    } catch (error: any) {
      lastError = error;
      const errorMessage = String(error.message || error).toLowerCase();
      console.error(`Error calling Gemini API (Attempt ${attempt + 1}/${MAX_RETRIES}):`, error);

      // Conditions for retry
      const isRetryableError = 
           errorMessage.includes("rpc failed") || 
           errorMessage.includes("xhr error") ||
           errorMessage.includes("fetch failed") || 
           errorMessage.includes("network error") || // Generic network error
           errorMessage.includes("timeout") || // Explicit timeout
           (error.status && error.status >= 500); // HTTP 5xx errors if available

      if (isRetryableError && attempt < MAX_RETRIES - 1) {
        const delayTime = 1000 * Math.pow(2, attempt); // Exponential backoff: 1s, 2s
        console.log(`Retrying in ${delayTime / 1000}s...`);
        await delay(delayTime);
        attempt++;
      } else {
        // Non-retryable error or max retries reached
        if (errorMessage.includes("api_key_invalid") || (error.message && error.message.includes("[400]"))) {
            throw new Error("La clave API de Gemini no es válida o ha expirado. Por favor, verifica la configuración.");
        }
        if (errorMessage.includes("fetch failed") || errorMessage.includes("inet") || errorMessage.includes("network")) {
             throw new Error("Error de red al contactar el servicio de Gemini. Verifica tu conexión a internet e inténtalo de nuevo.");
        }
        // Throw the last encountered error
        throw new Error(`Error al procesar la solicitud con Gemini después de ${MAX_RETRIES} intentos: ${lastError.message || lastError}`);
      }
    }
  }
  return null; // Should not be reached if error thrown, but as a fallback
};
