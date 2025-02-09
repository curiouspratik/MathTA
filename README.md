# 📚 MathTA

Local RAG application optimized for Math Q&A

#### Initial Setup -
![Initial App](/assets/Initial_app.png)

#### Changes made -


#### Final setup -



#### Demo video -


#### Setup instructions - 
Also look into the video above for better instructions
1. Clone the repo
2. Checkout to most recent branch (biggest number in at the start)
3. Make sure you have `docker` and `ollama` installed
4. Download some llm model on ollama, `llama3.2` base version is also fine
5. Open terminal, make sure Ollama is active
6. Go to the repo directory
7. Run `docker build -t math-ta-test .`
8. Then run `docker run -d -p 8501:8501 math-ta-test`
9. Now in any browser go to - `http://0.0.0.0:8501`
10. Upload a document you want to work with, and ask the questions
11. To close end the docker image and close the ollama in background

#### Citations -
This repository is fork of [local-rag](https://github.com/jonfairbanks/local-rag).

#### Warning -
Please only use this repository for educational purpose.