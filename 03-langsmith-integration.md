# INTEGRATING YOUR APP WITH LANGSMITH

The way to add tracing using Langsmith is pretty straight forward now.

Just Sign-Up here for a Langsmith account: https://smith.langchain.com/

Then, go to Tracing -> Create an API Key and add the below environment variables to the .env file:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your langsmith api key>
LANGSMITH_PROJECT=<your project name>
```

Then, just run your app using `uv run main.py` and you should be able to see the complete trace logs in https://smith.langchain.com/. 
