import { NextRequest } from 'next/server';
import { Message as VercelChatMessage, StreamingTextResponse } from 'ai';
 
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { BytesOutputParser } from '@langchain/core/output_parsers';
import { PromptTemplate } from '@langchain/core/prompts';
import { Ollama } from "@langchain/community/llms/ollama";
import { OpenAIEmbeddings } from "@langchain/openai";


export const runtime = 'edge';
 
/**
 * Basic memory formatter that stringifies and passes
 * message history directly into the model.
 */
const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};
 
const ollama = new Ollama({
    baseUrl: "http://localhost:11434", // Default value
    model: "mistral", // Default value
  });

const TEMPLATE = `Answer the question: {query} based on the following documents and conversation.

Documents:
{documents}

Current conversation:
{chat_history}`;
 
/*
 * This handler initializes and calls a simple chain with a prompt,
 * chat model, and output parser. See the docs for more information:
 *
 * https://js.langchain.com/docs/guides/expression_language/cookbook#prompttemplate--llm--outputparser
 */
export async function POST(req: NextRequest) {
  const body = await req.json();
  const messages = body.messages ?? [];
  const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
  const currentMessageContent = messages[messages.length - 1].content;
 
  const vectorStore = await Chroma.fromExistingCollection(
    new OpenAIEmbeddings(),
    { collectionName: "makers",
      url: "http://localhost:8000"}
  );
  
  const response = await vectorStore.similaritySearch(currentMessageContent, 5);

  const prompt = PromptTemplate.fromTemplate(TEMPLATE);
 
  /**
   * Chat models stream message chunks rather than bytes, so this
   * output parser handles serialization and encoding.
   */
  const outputParser = new BytesOutputParser();

  /*
   * Can also initialize as:
   *
   * import { RunnableSequence } from "langchain/schema/runnable";
   * const chain = RunnableSequence.from([prompt, model, outputParser]);
   */
  const chain = prompt.pipe(ollama).pipe(outputParser);
 
  const stream = await chain.stream({
    query: currentMessageContent,
    documents: JSON.stringify(response),
    chat_history: formattedPreviousMessages.join('\n'),
  });
 
  return new StreamingTextResponse(stream);
}