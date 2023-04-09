// Instalando as bibliotecas necessárias
/* Devem ser executadas no terminal
npm install express // Express é utilizado para lidar com requisições HTTP
pip install hnswlib // HNSWLib é utilizado para construir um índice de vetores e fazer consultas de similaridade
npm install -g node-gyp // Node-gyp é necessário para instalar a biblioteca hnswlib-node
npm install -S hnswlib-node // hnswlib-node é uma biblioteca de índice de vetor rápida e fácil de usar para Node.js
***************/

// Importando as bibliotecas necessárias
const express = require('express'); // Para lidar com requisições HTTP
const fs = require('fs'); // Para lidar com o sistema de arquivos
const path = require('path'); // Para lidar com caminhos de arquivos
const { OpenAI } = require("langchain/llms"); // Para usar a API do OpenAI
const { RetrievalQAChain } = require("langchain/chains");
// Para criar uma cadeia de recuperação de perguntas e respostas
const { HNSWLib } = require("langchain/vectorstores"); // Para criar um vetor de armazenamento de dados baseado em HNSW
const { OpenAIEmbeddings } = require("langchain/embeddings"); // Para usar os recursos de incorporação do OpenAI
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter"); // Para dividir o texto em documentos menores

// Configurando o aplicativo Express
/*
Estamos usando o Express porque ele é um framework para Node.js que torna mais fácil a criação de aplicativos da Web e APIs. 
O Express fornece uma série de recursos úteis, como gerenciamento de rotas, middleware para manipulação de solicitações e respostas, e suporte para vários mecanismos de visualização. 
Ele também é muito popular e possui uma grande comunidade de desenvolvedores, o que torna mais fácil encontrar exemplos, soluções para problemas comuns e bibliotecas de terceiros que se integram bem com o Express.
*/
const app = express(); // Instanciando o objeto da aplicação com o Express
app.use(express.urlencoded({ extended: true })); // Configurando a aplicação para usar o encoding de urlencoded
app.use(express.json()); // Configurando a aplicação para usar o formato de dados JSON
app.use(express.static('public')); // Configurando a aplicação para servir arquivos estáticos na pasta 'public'

// Função para obter textos de um diretório
const getTextsFromDirectory = async (dirPath) => {
  const files = await fs.promises.readdir(dirPath);
  const textFiles = files.filter(file => path.extname(file) === '.txt');
  const texts = [];

  // Lendo os arquivos de texto e armazenando em um array
  for (const file of textFiles) {
    const text = await fs.promises.readFile(path.join(dirPath, file), 'utf8');
    texts.push(text);
  }

  return texts;
};


// Função responsável por rodar o modelo do OpenAI e retornar a resposta para a pergunta
async function run(question, openAIApiKey) {
  //console.log('function run:' + question, openAIApiKey, customUrl);
  // Setando a chave da API do OpenAI como variável de ambiente
  process.env.OPENAI_API_KEY = openAIApiKey;
  // Instanciando um novo objeto OpenAI
  const model = new OpenAI({});

  // Lendo os textos a partir do diretório "meusarquivos"
  const dirPath = 'meusarquivos'; // Substitua pelo caminho para o seu diretório de textos
  const texts = await getTextsFromDirectory(dirPath);

  console.log(texts);
  // Instanciando um novo objeto RecursiveCharacterTextSplitter para dividir o texto em chunks menores
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  // Criando documentos a partir dos chunks de texto
  const docs = await textSplitter.createDocuments(texts);
  // Passando a chave da API para inicializar o objeto OpenAIEmbeddings
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings(process.env.OPENAI_API_KEY));

  // Instanciando um novo objeto RetrievalQAChain, que irá usar o objeto model e o vectorStore para recuperar as respostas
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

  // Deletando a variável de ambiente OPENAI_API_KEY
  delete process.env.OPENAI_API_KEY;
  // Retornando a resposta
  return chain.call({ query: question }).then(res => res.text);
}


app.post('/ask', async (req, res) => {
  //console.log(req.body);
  const { question, openAIApiKey } = req.body;
  try {
    const answer = await run(question, openAIApiKey);
    res.json({ answer });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message });
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});