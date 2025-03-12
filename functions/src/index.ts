import {gemini15Flash, googleAI} from "@genkit-ai/googleai";
import {genkit} from "genkit/beta";
import {z} from "genkit";
import {onCallGenkit, hasClaim} from "firebase-functions/https";
import {defineSecret} from "firebase-functions/params";
import {initializeApp} from "firebase-admin/app";
import {getFirestore} from "firebase-admin/firestore";


const app = initializeApp();
const db = getFirestore(app);

const googleAIapiKey = defineSecret("GEMINI_API_KEY");
const systemPrompt =
  "You are Kian, an assistant for an " +
  "AI documentation tool called DocIt. " +
  "DocIt is a tool where users can document things " +
  "that they did using counters and text description. " +
  "DocIt also provides File " +
  "organizations with Folders, Lists, and Items. " +
  "Talk to the user in a professional manner.";

// Configure a Genkit instance
const ai = genkit({
  plugins: [googleAI()],
  model: gemini15Flash,
});


// Define a tool that fetches user items
const getAllItems = ai.defineTool(
  {
    name: "getAllItems",
    description: "Gets all the items created by the current user user",
    inputSchema: z.object({
      userId: z.string().describe("The user ID to fetch data for"),
    }),
    outputSchema: z.array(
      z.object({
        id: z.string().describe("The ID of the item"),
        name: z.string().describe("The name of the item"),
        count: z.number().describe("The count of the item"),
        dateCreated: z.string().describe("The date the" +
           " item was created"),
        description: z.string().optional().describe("The description " +
          "of the item"),
        dateModified: z.string().optional().describe("The date the item " +
          "was modified"),
      })
    ),
  },
  async ({userId}) => {
    // Return an empty array if no userId is provided.
    if (!userId) return [];
    // Query the "items" collection group using the Admin SDK.
    const itemsQuery = db.collectionGroup("items").
      where("createdBy", "==", userId);
    const querySnapshot = await itemsQuery.get();
    const res = querySnapshot.docs.map((item) => {
      const data = item.data();
      return {
        id: item.id,
        name: data.name,
        count: data.count,
        dateCreated: data.dateCreated.toDate().toLocaleString(),
        description: data.description || "",
        dateModified: data.dateModified ?
          data.dateModified.toDate().toLocaleString() :
          "",
      };
    });
    return res;
  }
);


// Define a tool that fetches user lists
const getAllLists = ai.defineTool(
  {
    name: "getAllLists",
    description: "Gets all the lists created by the current user",
    inputSchema: z.object({
      userId: z.string().describe("The user ID to fetch data for"),
    }),
    outputSchema: z.array(
      z.object({
        id: z.string().describe("The ID of the list"),
        name: z.string().describe("The name of the list"),
        dateCreated: z.string().describe("The date the" +
           " list was created"),
        description: z.string().optional().describe("The description " +
          "of the list"),
        dateModified: z.string().optional().describe("The date the list " +
          "was modified"),
      })
    ),
  },
  async ({userId}) => {
    // Return an empty array if no userId is provided.
    if (!userId) return [];
    // Query the "items" collection group using the Admin SDK.
    const listQuery = db.collectionGroup("lists").
      where("createdBy", "==", userId);
    const querySnapshot = await listQuery.get();
    const res = querySnapshot.docs.map((list) => {
      const data = list.data();
      return {
        id: list.id,
        name: data.name,
        dateCreated: data.dateCreated.toDate().toLocaleString(),
        description: data.description || "",
        dateModified: data.dateModified ?
          data.dateModified.toDate().toLocaleString() :
          "",
      };
    });
    return res;
  }
);


// Define a tool that fetches user folders
const getAllFolders = ai.defineTool(
  {
    name: "getAllFolders",
    description: "Gets all the folders created by the current user",
    inputSchema: z.object({
      userId: z.string().describe("The user ID to fetch data for"),
    }),
    outputSchema: z.array(
      z.object({
        id: z.string().describe("The ID of the folder"),
        name: z.string().describe("The name of the folder"),
        dateCreated: z.string().describe("The date the" +
           " folder was created"),
        description: z.string().optional().describe("The description " +
          "of the folder"),
        dateModified: z.string().optional().describe("The date the folder " +
          "was modified"),
      })
    ),
  },
  async ({userId}) => {
    // Return an empty array if no userId is provided.
    if (!userId) return [];
    // Query the "items" collection group using the Admin SDK.
    const folderQuery = db.collectionGroup("folders").
      where("createdBy", "==", userId);
    const querySnapshot = await folderQuery.get();
    const res = querySnapshot.docs.map((folder) => {
      const data = folder.data();
      return {
        id: folder.id,
        name: data.name,
        dateCreated: data.dateCreated.toDate().toLocaleString(),
        description: data.description || "",
        dateModified: data.dateModified ?
          data.dateModified.toDate().toLocaleString() :
          "",
      };
    });
    return res;
  }
);



/**
 * Generates a response using the Gemini model.
 *
 * @param {{prompt: string}} data - An object containing the prompt.
 * @return {Promise<string>} The generated response text.
 */
async function generateResponseFromGemini(
  data: {prompt: string, userId: string}
): Promise<string> {
  const {prompt, userId} = data;
  const augmentedPrompt = `${prompt}\nUserId: ${userId}`;

  const {text} = await ai.generate({
    system: systemPrompt,
    prompt: augmentedPrompt,
    config: {
      maxOutputTokens: 400,
      stopSequences: ["<end>", "<fin>"],
      temperature: 1.2,
      topP: 0.4,
      topK: 50,
    },
    tools: [getAllItems, getAllLists, getAllFolders],
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
    if (!context?.auth?.uid) throw new Error("Must supply auth context.");
    const res = await generateResponseFromGemini(
      {prompt, userId: context.auth.uid}
    );
    return res;
  }
);


export const summarizeData = onCallGenkit(
  {
    secrets: [googleAIapiKey],
    authPolicy: hasClaim("email_verified"),
  },
  summarizeDataFlow
);

