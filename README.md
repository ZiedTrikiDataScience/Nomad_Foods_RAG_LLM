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


## Deployment Instructions

### 0. Cloning the git repo to have all the files :

```bash
git clone https://github.com/ZiedTrikiDataScience/Nomad_Foods_RAG_LLM.git
```

-  Navigate to the Cloned Repository Directory: 
```bash
   cd Nomad_Foods_RAG_LLM
```
### 1. **Obtain API Keys:**
   - Sign up for an account with MistralAI and obtain your API key.

### 2. **Create Kubernetes Secrets:**
   - Run the following command to create a Kubernetes secret with your API key:

```bash
   kubectl create secret generic mistral-api-key --from-literal=MISTRAL_API_KEY=<your-api-key>
 ```

### 3. **Set Up Docker Image:**  
 - Pull the Docker image from Docker Hub:

```bash
   docker pull ziedtrikimlops/rag-chatbot-nomad-food:v1
```

### 4. **Deploy the Application on Kubernetes:**
- Ensure Kubernetes is set up and running on your local machine or a cloud provider.
- Apply the deployment and service YAML files to start the application on Kubernetes:


#####  4.1: Apply the Kubernetes Deployment kubectl yaml file :
```bash
   kubectl apply -f rag_nomad_app_deployment.yaml
 ```

#####  4.2: Apply the Kubernetes Service kubectl yaml file :
```bash
   kubectl apply -f rag_nomad_app_service.yaml
```

##### 4.3: Verify the Deployment Status:
- To check the status of your pods and ensure they’re running correctly:

```bash
   kubectl get pods
   kubectl get services
 ```

- Confirm that the rag-nomad-streamlit-chatbot pod is running and the service is accessible.

### 5. **Accessing the Application:**
 - Open the app in your browser at :
```bash 
 http://localhost:30001/
```

### 6. **Testing the Application:**
 - Interact with the RAG Chatbot with Entering queries based on FAQs related to Nomad Foods and test the enhanced response given by the app

### 7. **Ingestion Pipeline:**
- Check the ***new_faq_data.json*** file that includes new faq data to be ingested and integrated to the dataset.
- Run the script ***prefect_new_faq_ingestion_pipeline.py*** that reads that new data and concatenates it to the original dataset to update it with the new faqs. 

### 8. **RAG Evaluation:**

### 8.0 **Generate the Ground truth Dataset:**

- Run the ***generate_ground_truth_dataset.ipynb*** to generate the ground truth dataset that we will use for the Retrieval and Generation Evaluation.

#### 8.1 : **Retrieval Evaluation :**

- Run the ***evaluate_retrieval_faiss_rank_bm25.ipynb*** to test retrieval performance and quality.
 
- This notebook allows you to assess the ***Hit-Rate*** and ***MMR*** of retrieved information compared to ground truth data using both ***FAISS*** and ***Rank_BM25***.

- The metrics' results are almost similar but I chose ***FAISS*** because in a production environment, and thinking about future scalability of the project, ***FAISS*** excels by being faster in terms of retrieval speed especially for large datasets and high-dimensional data.

#### 8.2 : **Generation Evaluation :**

- Run the ***evaluate_generator_rag.ipynb*** to test generation performance and quality.

- This notebook allows you to assess the generator quality through ***Cosine Similarity*** and ***LLM-As-A-Judge*** offline evaluation techniques.

- ***The Cosine Similarity*** is computed between the ***LLM_Answer*** and the ***Ground_Truth_Answer***.

- The ***LLM-As-A-judge*** is applied through calling ***ministral-3b-latest*** which is a different mistral model than the one used in the original app : ***mistral-large-latest*** so that we ensure an unbiased generation evaluation.

### 9 : Monitoring :
- To monitor the rag system , follow these steps :

  * ```bash
cd monitoring
```



### 10. Best Practises and Improvement Techniques Evaluation :

- Run the ***rag_nomad_foods_hybrid_re_rank_re_write.py*** to test the rag system and its comparision when working with 2 methods:
  
   * First Method: The straightforward ***FAISS vector search***.

   * Second Method: The addition of :
   **Hybrid Vector and Text Search**
   **Document re-ranking**
   **User query rewriting**


- The Second Method with the optimization practises of **Hybrid Vector and Text Search** , **Document re-ranking** , **User query rewriting** gave better results.

- Nevertheless, for this project , we will deploy the rag system working with the first method of the straightforward FAISS Vector Search as v1 with potentially working to integrate the second for the next v2.
