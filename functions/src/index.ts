import { gemini15Flash, googleAI } from "@genkit-ai/googleai";
import {genkit} from "genkit/beta"
import {z} from "genkit";
import {onCallGenkit, hasClaim} from "firebase-functions/https";
import {defineSecret} from "firebase-functions/params";
const googleAIapiKey = defineSecret("GEMINI_API_KEY");
const systemPrompt =
    "You are Kian, an assistant for an " +
  "AI documentation tool called DocIt.";

// configure a Genkit instance
const ai = genkit({
  plugins: [googleAI()],
  model: gemini15Flash,
});

/**
 * Generates a response using the Gemini model.
 *
 * @param {{prompt: string}} data - An object containing the prompt.
 * @return {Promise<string>} The generated response text.
 */
async function generateResponseFromGemini(data: {prompt: string}) {
    const { prompt } = data;
    
  const {text} = await ai.generate({
    system: systemPrompt,
    prompt: prompt,
    config: {
      maxOutputTokens: 400,
      stopSequences: ["<end>", "<fin>"],
      temperature: 1.2,
      topP: 0.4,
      topK: 50,
    },
  });
  return text;
}

const summarizeDataFlow = ai.defineFlow(
  {
    name: "summarizeData",
    inputSchema: z.object({prompt: z.string()}),
    outputSchema: z.string(),
  },
  async ({prompt}, {context}) => {
    // if (!context.auth?.uid) throw new Error("Must supply auth context.");
    const res = await generateResponseFromGemini({ prompt: prompt });
    if (context) {
      console.log(context.auth.uid)
    }
    console.log('hello from summarize Data Flow')
    console.log(res)
    return res;
  }
);


const summary = summarizeDataFlow({ prompt: "What is your name" }, {
  context: {
    auth: {
      uid: "a test uid"
}}});


export const summarizeData = onCallGenkit(
  {
    secrets: [googleAIapiKey],
    authPolicy: hasClaim("email_verified"),
  },
  summarizeDataFlow
);