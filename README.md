# Nomad Foods RAG LLM Chatbot System:

## Problem Description
In today’s fast-paced retail environment, food product companies face increasing challenges in providing accurate and immediate responses to a wide variety of questions from their customers (retailers and end consumers). Queries can range from product details, delivery options, and pricing, to sustainability practices and retailer support.

Managing this diverse set of inquiries can overwhelm customer service teams, leading to delayed responses and unsatisfied customers. Moreover, with a growing catalog of products and services, it becomes increasingly difficult for service representatives to have up-to-date information on every product or policy.

## Project Solution
To address these challenges, this project proposes building a Retrieval-Augmented Generation (RAG) system that uses a structured FAQ database to provide real-time responses to customer inquiries. The system will consist of a chatbot capable of retrieving and presenting relevant information from a well-curated database of FAQs that cover the company’s products, delivery options, pricing, sustainability practices, and retailer support.

### Key components of the solution include:

#### Instant, Accurate Responses: 
The chatbot will provide immediate answers to customer questions by retrieving relevant information from the FAQ database.
#### AI-Powered Search and Retrieval: 
The system will use advanced AI techniques to match user queries with the most appropriate answers, ensuring accurate responses even for complex questions.
#### Contextualized Answers: 
When necessary, the system will generate more personalized or detailed responses using a fine-tuned language model that builds on the retrieved information.
The system will automate common customer inquiries, helping the company provide faster response times and improving customer satisfaction. It will also reduce the burden on customer service teams, allowing them to focus on more complex or unique customer issues.

## Key Benefits:
Scalable and Fast Response System: The automated system will allow the company to scale customer service operations without requiring additional staff.
Customizable Chatbot Interaction: The chatbot can tailor its responses based on the user’s inquiry, ensuring a better customer experience.
Improved Customer Service Efficiency: By handling repetitive and common queries, the system allows customer service agents to focus on higher-value tasks.


## Pipeline Description :

### Prerequisites :

* Install Python 3.11.9

* Create a Virtual environment and activate it

* Install the requirements.txt :
```bash
pip install -r requirements.txt
```

### 1. Cloning the git repo to have all the files :
```bash
git clone https://github.com/ZiedTrikiDataScience/Nomad_Foods_RAG_LLM.git
```






## Deployment Instructions

1. **Obtain API Keys:**
   - Sign up for an account with MistralAI and obtain your API key.

2. **Create Kubernetes Secrets:**
   Run the following command to create a Kubernetes secret with your API key:
   ```bash
   kubectl create secret generic mistral-api-key --from-literal=MISTRAL_API_KEY=<your-api-key>
