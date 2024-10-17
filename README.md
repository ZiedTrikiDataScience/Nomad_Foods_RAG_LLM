# RAG Fintech Chatbot 


## Deployment Instructions

1. **Obtain API Keys:**
   - Sign up for an account with MistralAI and obtain your API key.

2. **Create Kubernetes Secrets:**
   Run the following command to create a Kubernetes secret with your API key:
   ```bash
   kubectl create secret generic mistral-api-key --from-literal=MISTRAL_API_KEY=<your-api-key>
