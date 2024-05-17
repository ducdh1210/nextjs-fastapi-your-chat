# Next.js + FastAPI: Chat with Your Website

This application facilitates a chatbot by leveraging Next.js as the frontend and FastAPI as the backend, utilizing the power of LangChain for dynamic web interactions.

![Chatbot Interface](images/chatbot.png)

## Features of the Hybrid App

- **Web Interaction via LangChain**: Utilizes the latest LangChain version for effective interaction and information extraction from websites.
- **Versatile Language Model Integration**: Offers compatibility with various models including GPT-4. Users can easily switch between models to suit their needs.
- **User-Friendly Next.js Frontend**: The interface is intuitive and accessible for users of all technical backgrounds.

## Operational Mechanics

The application integrates the Python/FastAPI server into the Next.js app under the `/api/` route. This is achieved through [`next.config.js` rewrites](https://github.com/digitros/nextjs-fastapi/blob/main/next.config.js), directing any `/api/:path*` requests to the FastAPI server located in the `/api` folder. Locally, FastAPI runs on `127.0.0.1:8000`, while in production, it operates as serverless functions on Vercel.

## Setting Up the Application

1. Install dependencies:
   ```bash
   npm install
   ```
2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=[your-openai-api-key]
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Access the application at [http://localhost:3000](http://localhost:3000). The FastAPI server runs on [http://127.0.0.1:8000](http://127.0.0.1:8000).

For backend-only testing:

```bash
poetry init 
make run-backend
```

