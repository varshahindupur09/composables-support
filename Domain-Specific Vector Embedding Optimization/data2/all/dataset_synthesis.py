# dataset_synthesis.py

import os
import chromadb
import logging
import argparse
import pandas as pd
import random
import json
import time
import re
import sys

from dotenv import load_dotenv
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# Import the separate modules
# from data2.all.
from url_knowledge_base import URLKnowledgeBase
# from data2.all.
from pdf_knowledge_base import PDFKnowledgeBase

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_synthesis.log", mode='w',encoding='utf-8'),
        # logging.StreamHandler()
        logging.StreamHandler(sys.stdout)  # ✅ Logs output to terminal
    ]
)
logger = logging.getLogger(__name__)


# Load API keys and models
load_dotenv()
API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2")
]

MODELS = [
    "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile",
    "qwen-2.5-32b"
]

class DatasetSynthesisModule:
    """Main module integrating URL and PDF knowledge bases into vector storage."""

    def __init__(self, use_url, use_pdf, knowledge_links, pdf_folder, embedding_model_name="all-MiniLM-L6-v2", api_key=None):
        self.use_url = use_url
        self.use_pdf = use_pdf
        self.embedding_model_name = embedding_model_name
        self.documents = []
        self.api_key_index = 0
        self.model_index = 0
        self.existing_questions = set()  # Persisted set across API switches

        # Load Paraphrasing Model
        self.paraphraser = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.paraphraser_tokenizer = T5Tokenizer.from_pretrained("t5-base")

        # Initialize knowledge sources
        if self.use_url:
            self.url_knowledge_base = URLKnowledgeBase(knowledge_links)
        if self.use_pdf:
            self.pdf_knowledge_base = PDFKnowledgeBase(pdf_folder)
        
        # Initialize LLM
        self.set_llm()

        self.setup()
    
    def set_llm(self):
        """Set the LLM with the current API key and model."""
        if self.api_key_index >= len(API_KEYS):
            logger.info("All API keys exhausted. Please try again later.")
            exit(1)

        # logger.info("which model and api index key: ", self.model_index, self.api_key_index)
        self.llm = Groq(model=MODELS[self.model_index], api_key=API_KEYS[self.api_key_index])
        logger.info(f"Using Model: {self.model_index}, {MODELS[self.model_index]} with API Key {self.api_key_index + 1}")

    def switch_api_key(self):
        """Switch to the next API key after cycling through all models."""
        self.api_key_index = (self.api_key_index + 1) % len(API_KEYS)
        # logger.info("has api key switched? ", self.api_key_index)

        if self.api_key_index >= len(API_KEYS):
            logger.error("⚠️ All API keys have been used. Stopping execution.")
            exit(1)

        self.set_llm()
        self.generate_synthetic_dataset(remaining_samples=True)  # Resume generation

    def switch_model(self):
        """Switch to the next model and cycle through all models before switching API key."""
        self.model_index += 1
        if self.model_index >= len(MODELS):
            logger.info(f"🔄 All models used for API Key {self.api_key_index + 1}. Switching API Key...")
            self.switch_api_key()
        
        logger.info(f"🔄 Switching to Model {self.model_index},{MODELS[self.model_index]},{self.api_key_index}")
        self.set_llm()
        # self.generate_synthetic_dataset()  # Resume generation
        self.generate_synthetic_dataset(remaining_samples=True)  # ✅ Resume generation

    def save_extracted_documents(self):
        """Save extracted document text to a file for verification."""
        with open('extracted_documents.txt', 'w', encoding='utf-8') as f:
            for doc in self.documents:
                # f.write(f"Source: {doc.metadata['source']}\n")
                f.write(doc.text + "\n\n" + "="*80 + "\n\n")  # Separate documents
            logger.info("Extracted documents saved to extracted_documents.txt.")

    def setup(self):
        """Initialize components, process documents, and generate synthetic cryptocurrency dataset."""
        if self.use_url:
            self.url_knowledge_base.fetch_documents()
            self.documents.extend(self.url_knowledge_base.documents)

        if self.use_pdf:
            self.pdf_knowledge_base.extract_documents()
            self.documents.extend(self.pdf_knowledge_base.documents)

        if not self.documents:
            logger.info("No documents available for indexing or synthetic data generation.")
            return 
        
        # filter out documents with non-meaningful text
        self.documents = [doc for doc in self.documents if len(doc.text.split()) > 20]
        
        # Save extracted documents to a file
        self.save_extracted_documents()
        
        # Initialize ChromaDB for vector storage
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Initialize LLM - llama-3.3-70b-versatile
        # if self.llm_name == "groq":
        #     self.llm = Groq(model="llama-3.3-70b-specdec", api_key=self.api_key)
        # else:
        #     raise ValueError("Invalid LLM name. Only 'groq' is supported for now.")

        # Set LlamaIndex settings
        Settings.llm = self.llm
        # Settings.embed_model = SentenceTransformer(self.embedding_model_name)
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
        # Settings.chunk_size = 200  # Daily limit is kept in mind
        # Settings.chunk_size = 500 # defi knowledge base
        # Settings.chunk_size = 350 # ai in fin knowledge base
        Settings.chunk_size = 200 # ai in fin knowledge base

        # Create vector store
        chroma_collection = chroma_client.get_or_create_collection("financial_data")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from documents
        if self.documents:
            self.index = VectorStoreIndex.from_documents(self.documents, storage_context=storage_context)
            logger.info("Vector index created successfully.")
        
        # Generate synthetic financial dataset
        self.generate_synthetic_dataset(num_samples=2500)

        # # Save dataset to CSV
        # if self.synthetic_dataset:
        #     df = pd.DataFrame({"synthetic_dataset": self.synthetic_dataset})
        #     df.to_csv("synthetic_dataset.csv", index=False)
        #     logger.info("Synthetic dataset saved to synthetic_dataset.csv.")

    def get_random_chunk(self, text, chunk_size=10000):
        """Select a random chunk of text from the document."""
        if len(text) <= chunk_size:
            return text  # Return the entire text if it's smaller than the chunk size
        start_index = random.randint(0, len(text) - chunk_size)
        return text[start_index:start_index + chunk_size]


    def generate_synthetic_dataset(self, num_samples=100, remaining_samples=False):
        """Generate synthetic financial dataset using the LLM."""
        output_file = "synthetic_crypto_qa_1.csv"

        if remaining_samples:
            logger.info("Resuming dataset generation after API key/model switch.")

        requests_made = 0
        start_time = time.time()
        rate_limit_reached_flag = False

        for i in range(num_samples):
            if requests_made >= 30:  # Enforce 30 requests per minute
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                start_time = time.time()
                requests_made = 0

            document_text = random.choice(self.documents).text[:5000]

            response = str()

            for attempt in range(3):  # Retry logic for failed API calls
                try:
                    response = self.llm.complete(
                        f"""Generate a unique finance-related question-answer pair:
                        {{"question": "...", "answer": "..."}}

                        Text:
                        {document_text}
                        """
                    )

                    response_text = response.text if hasattr(response, "text") else str(response)
                    qa_pair = json.loads(response_text.strip())

                    # Check for duplicates
                    if qa_pair["question"] in self.existing_questions:
                        # logger.info(f"Duplicate question detected. Skipping: {qa_pair['question']}")
                        continue

                    # Save to file immediately
                    self.existing_questions.add(qa_pair["question"])
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(f'"{qa_pair["question"]}","{qa_pair["answer"]}"\n')

                    requests_made += 1
                    break

                except json.JSONDecodeError:
                    logger.info(f"Invalid JSON format for entry {i}. Skipping.")
                    print("response: ", response)
                    break

                except Exception as e:
                    print("ERROR*** ", str(e).upper())
                    error_msg = str(e).lower()

                    # ✅ Handle 503 (Service Unavailable)
                    if "service unavailable" in error_msg or "503" in error_msg:
                        if attempt == 4:  # ✅ If all retries fail, switch model
                            logger.info("⚠️ Service Unavailable after multiple retries. Switching Model...")
                            self.switch_model()
                            return  # ✅ Exit loop and restart with a new model
                    # ✅ Handle 429 Rate Limit
                    elif "rate limit" in error_msg or "429" in error_msg:
                        if attempt == 4:  # ✅ If retries fail, switch API key
                            logger.info("⚠️ Rate Limit reached after multiple retries. Switching API Key...")
                            self.switch_api_key()
                            return  # ✅ Exit loop and restart with a new API key
                    else:
                        # ✅ Exponential Backoff for Temporary Issues
                        wait_time = 2 ** attempt  # 2, 4, 8, 16, 32 seconds
                        logger.warning(f"⚠️ Request failed (Attempt {attempt+1}/5). Retrying in {wait_time} sec... Error: {e}")
                        time.sleep(wait_time)

            if i % 100 == 0:
                logger.info(f"Generated {i}/{num_samples} synthetic dataset entries.")

            logger.info(f"✅ Finished generating with Model: {self.model_index}, {MODELS[self.model_index]} | API Key {self.api_key_index + 1}")

            # ✅ If all models are used, switch to the next API key
            if self.model_index == len(MODELS) - 1:
                self.switch_api_key()
            else:
                self.switch_model()
    

if __name__ == "__main__":
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Parse user input to choose sources
    parser = argparse.ArgumentParser(description="Choose knowledge sources.")
    parser.add_argument("--use-url", action="store_true", help="Use URL-based knowledge sources.")
    parser.add_argument("--use-pdf", action="store_true", help="Use PDF-based knowledge sources.")
    args = parser.parse_args()

    knowledge_base_links = [
        # "https://www.sciencedirect.com/science/article/pii/S2405918822000071?ssrnid=4053319&dgcid=SSRN_redirect_SD"
        # "https://binariks.com/blog/predictive-analytics-in-fintech/"
        # "https://www.finextra.com/blogposting/26980/al-driven-dashboards-how-predictive-al-analytics-is-shaping-the-future-financial-forecasting"
        # "https://www.finextra.com/blogposting/26980/al-driven-dashboards-how-predictive-al-analytics-is-shaping-the-future-financial-forecasting",
        # "https://slalom.com/us/en/insights/financial-services-outlook-2025?utm_source=google&utm_medium=paid_search&utm_campaign=2025-Q1-GBL-IND-CON-Paid-Ads-FS-Outlook&utm_term=Technology&utm_content=FS_Outlook&creative=730300919730&keyword=artificial%20intelligence%20in%20financial%20services&matchtype=p&network=g&device=c&gad_source=1&gclid=Cj0KCQiAz6q-BhCfARIsAOezPxlaPpmzH6jUKQWhrDcBo6v38syDCQGI_OTqCgiWlXC53RbDiPszH4MaAuacEALw_wcB",
        # "https://www.td.com/ca/en/investing/direct-investing/articles/what-is-stock-market"
        # "https://www.zeni.ai/blog/financial-predictive-analytics"
        # "https://aws.amazon.com/web3/what-is-defi/"
        # "https://disb.dc.gov/page/beware-decentralized-finance-defi"
        # "https://www.dcreport.org/2024/06/18/stock-market-simplified-a-students-guide-to-understanding-investments/?gad_source=1&gclid=CjwKCAiArKW-BhAzEiwAZhWsIO6X38U5_9sQYE-6OINTeNNXlal0gyOWFHx31up28hg8pW8YwngNFxoCcGEQAvD_BwE"
        # "https://business.fiu.edu/academics/graduate/insights/posts/artificial-intelligence-in-the-stock-market-how-did-it-happen.html"
        # "https://www.velvetech.com/blog/high-frequency-algorithmic-trading/"
        "https://www.investopedia.com/terms/a/algorithmictrading.asp"
    ]

    # 1-500
    # Define knowledge sources
    # knowledge_base_links = [
    #     # cryptocurrency
    #     # "https://usa.kaspersky.com/resource-center/definitions/what-is-cryptocurrency",
    #     "https://en.wikipedia.org/wiki/Cryptocurrency",
    #     "https://coinmarketcap.com/",
    #     "https://www.oswego.edu/cts/basics-about-cryptocurrency",
    #     "https://www.bankrate.com/investing/top-performing-crypto/",
    #     "https://coinledger.io/learn/best-long-term-crypto",
    #     "https://www.forbes.com/sites/digital-assets/article/top-cryptocurrencies-to-watch-2025/",
    #     "https://investinghaven.com/crypto-forecasts/15-cryptocurrency-forecasts-2025/",
    #     "https://www.investopedia.com/best-crypto-exchanges-5071855",
    #     "https://www.pwc.com/us/en/industries/financial-services/fintech/bitcoin-blockchain-cryptocurrency.html",
    #     "https://www.rba.gov.au/education/resources/explainers/cryptocurrencies.html",
    #     "https://www.google.com/finance/markets/cryptocurrencies?hl=en"
    #     "https://www.edwardjones.com/us-en/investment-services/investment-products/how-do-stocks-work?utm_source=google&utm_medium=paidsearch&utm_campaign=20833985210&utm_agid=157081167795&utm_term=investing%20in%20stocks&creative=683386381392&device=c&mjp=exp&mbu=per&mau=na&mob=stc&mbt=gim&&&&&gad_source=1&gclid=CjwKCAiArKW-BhAzEiwAZhWsII-iofxjgfP8SB3HPS6I4zj4AT9W9RfYwVFP8k7SNdlK39M5G9eBTxoChqMQAvD_BwE&gclsrc=aw.ds",
    #     "https://www.fidelity.com/learning-center/trading-investing/crypto/decentralized-finance-defined#:~:text=DeFi%20stands%20for%20decentralized%20finance,first%20cover%20traditional%2C%20centralized%20finance.",
    #     "https://smartvalor.com/ru/news/defi-basics",
    #     "https://www.blockpit.io/en-us/blog/what-is-defi-decentralized-finance",
    #     "https://www.argoblockchain.com/articles/defi-revolution-decentralized-finance",
    #     "https://disb.dc.gov/page/beware-decentralized-finance-defi",
    #     "https://chain.link/education/defi/",
    #     "https://www.coursera.org/articles/what-is-defi",
    #     "https://appinventiv.com/blog/decentralized-finance-defi-guide/",
    #     "https://academy.geniusyield.co/guides/defi",
    #     "https://ijaracdc.com/decentralized-finance/?utm_source=google&utm_campaign=22173774846&utm_content=&utm_term=&utm_medium=&gad_source=1&gclid=CjwKCAiArKW-BhAzEiwAZhWsIAo4izdu1B9Ai4wBSew2ehNLoLZIC8JjG0VfcNs8NrGl16xlarfStBoC9-oQAvD_BwE",
    #     "https://aws.amazon.com/web3/what-is-defi/",
    #     "https://n26.com/en-eu/blog/what-is-defi",
    #     "https://www.paystand.com/blog/decentralized-finance",

    #     # blockchain
    #     # "https://denver-south.com/blog/what-is-a-blockchain-an-easy-to-digest-guide-for-the-uninitiated/?gad_source=1&gclid=CjwKCAiArKW-BhAzEiwAZhWsIG8i1BFKr3w4Ya5q0xfpCkoVr3mWBIVUCfqsVmi84Gzk_zw-jzPhuhoC4sIQAvD_BwE",
    #     "https://www.cnn.com/markets",
    #     # "https://www.investors.com/market-trend/stock-market-today/dow-jones-futures-trump-blames-globalists-nvidia-tesla-palantir-broadcom-jobs-report/",
    #     # "https://finance.yahoo.com/news/live/stock-market-today-nasdaq-enters-correction-sp-500-sinks-to-lowest-since-november-as-stocks-get-clobbered-on-trump-tariff-whiplash-210544344.html",
    #     # "https://tradingeconomics.com/united-states/stock-market,"
    #     "https://www.google.com/finance/?hl=en",
    #     "https://www.nyse.com/index",
    #     # "https://www.nasdaq.com/",
    #     "https://apnews.com/article/investors-stock-market-wall-street-volatility-a8bb85c802be802929bda213253c8178",
    #     "https://en.wikipedia.org/wiki/Stock_market",
    #     "https://www.google.com/finance/markets/indexes?hl=en",
    #     # "https://www.wsj.com/market-data/stocks",
    #     # "https://www.nytimes.com/section/markets-overview",
    #     # "https://www.reuters.com/markets/us/",
    #     # "https://www.wsj.com/livecoverage/stock-market-today-dow-sp500-nasdaq-live-03-06-2025",
    #     "https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/stock-market/",
    #     # "https://www.bloomberg.com/markets",
    #     "https://www.schwab.com/learn/story/stock-market-update-open",
    #     "https://www.cnn.com/markets/after-hours",
    #     "https://www.foxbusiness.com/stocks",
    #     # "https://www.morningstar.com/markets",
    #     "https://en.wikipedia.org/wiki/Stock_market_index",
    #     # "https://www.nseindia.com/",
    #     # "https://www.bseindia.com/",
    #     # "https://www.dcreport.org/2024/06/18/stock-market-simplified-a-students-guide-to-understanding-investments/?gad_source=1&gclid=CjwKCAiArKW-BhAzEiwAZhWsIO6X38U5_9sQYE-6OINTeNNXlal0gyOWFHx31up28hg8pW8YwngNFxoCcGEQAvD_BwE",
    #     "https://www.td.com/ca/en/investing/direct-investing/articles/what-is-stock-market",
    #     "https://www.nyse.com/markets/nyse/trading-info",
    #     "https://edition.cnn.com/markets/premarkets",
    #     "https://business.fiu.edu/academics/graduate/insights/posts/artificial-intelligence-in-the-stock-market-how-did-it-happen.html",
    #     "https://www.bloomberg.com/news/articles/2025-03-05/stock-market-today-dow-s-p-live-updates",
    #     "https://jamapunji.pk/knowledge-center/how-trade-stock-market"

    #     # ai in finance
    #     "https://www.oecd.org/en/topics/sub-issues/digital-finance/artificial-intelligence-in-finance.html#:~:text=It%20is%20used%20in%20fraud,%2C%20trading%2C%20and%20risk%20analysis.",
    #     "https://cloud.google.com/discover/finance-ai",
    #     "https://cloud.google.com/discover/finance-ai#how-is-ai-used-in-finance",
    #     "https://cloud.google.com/discover/finance-ai#what-is-ml-in-finance",
    #     "https://cloud.google.com/discover/finance-ai#applications-how-ai-can-solve-real-challenges-in-financial-services",
    #     "https://cloud.google.com/discover/finance-ai#benefits-of-ai-in-finance",
    #     "https://cloud.google.com/discover/finance-ai#the-future-of-ai-in-financial-services",
    #     "https://cloud.google.com/discover/finance-ai#hear-from-our-customers",
    #     "https://cloud.google.com/discover/finance-ai#related-products-and-services",
    #     "https://www.deloitte.com/ng/en/services/risk-advisory/services/how-artificial-intelligence-is-transforming-the-financial-services-industry.html",
    #     "https://mitsloan.mit.edu/ideas-made-to-matter/financial-services-deliberate-approach-to-ai",
    #     # "https://online.mason.wm.edu/blog/the-future-of-finance-ai-machine-learning-predictive-analytics",
    #     "https://www.imf.org/en/News/Articles/2024/09/06/sp090624-artificial-intelligence-and-its-impact-on-financial-markets-and-financial-stability",
    #     "https://www.chicagobooth.edu/review/evolution-ai-finance",
    #     "https://builtin.com/artificial-intelligence/ai-finance-banking-applications-companies",
    #     "https://www.forbes.com/sites/kathleenwalch/2024/09/14/how-ai-is-transforming-the-finance-industry/",
    #     "https://www.ey.com/en_gr/insights/financial-services/how-artificial-intelligence-is-reshaping-the-financial-services-industry",
    #     "https://www.fsb.org/2024/11/the-financial-stability-implications-of-artificial-intelligence/",
    #     "https://slalom.com/us/en/insights/financial-services-outlook-2025?utm_source=google&utm_medium=paid_search&utm_campaign=2025-Q1-GBL-IND-CON-Paid-Ads-FS-Outlook&utm_term=Technology&utm_content=FS_Outlook&creative=730300919730&keyword=artificial%20intelligence%20in%20financial%20services&matchtype=p&network=g&device=c&gad_source=1&gclid=Cj0KCQiAz6q-BhCfARIsAOezPxlaPpmzH6jUKQWhrDcBo6v38syDCQGI_OTqCgiWlXC53RbDiPszH4MaAuacEALw_wcB",

    #     # algorithmic trading
    #     "https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp",
    #     "https://corporatefinanceinstitute.com/resources/equities/algorithmic-trading/",
    #     "https://en.wikipedia.org/wiki/Algorithmic_trading",
    #     # "https://www.stonex.com/en/financial-glossary/algorithmic-trading/",
    #     "https://www.fool.com/terms/a/algorithmic-trading/",
    #     "https://www.utradealgos.com/blog/how-to-get-started-with-algo-trading-a-step-by-step-guide",
    #     "https://www.velvetech.com/blog/high-frequency-algorithmic-trading/",
    #     "https://www.angelone.in/knowledge-center/online-share-trading/what-is-algo-trading",
    #     "https://www.rsj.com/en/securities/algorithmic-trading.html",
    #     # "chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://faculty.haas.berkeley.edu/hender/Algo.pdf",
    #     "https://www.investopedia.com/terms/a/algorithmictrading.asp",
    #     # "https://www.fe.training/free-resources/capital-markets/what-is-algorithmic-trading/"

    #     # risk in defi
    #     "https://medium.com/intotheblock/four-levels-of-risk-management-in-defi-2a1173465a46",
    #     "https://streamflow.finance/blog/risk-management-defi/", # pdf
    #     "https://www.mdpi.com/1911-8074/18/1/38",
    #     "https://foundershield.com/blog/guide-to-defi-risk-management/",
    #     # "https://trakx.io/resources/insights/risk-management-in-crypto-trading-effective-guide/",
    #     "https://blog.shimmer.network/beginners-guide-risk-management-1/",
    #     "https://www.nuant.com/blog/how-to-measure-and-manage-defi-risks",
    #     "https://tokenminds.co/blog/knowledge-base/defi-risk-management",
    #     "https://www.forbes.com/councils/forbesbusinesscouncil/2025/03/06/decentralized-finance-and-fraud-navigating-security-risks/",
    #     "https://www.linkedin.com/advice/3/what-risks-defi-how-can-you-avoid-them-skills-blockchain-n3avf",
    #     "https://www2.deloitte.com/us/en/pages/audit/articles/blockchain-digital-assets-risk-management.html",
    #     "https://www.nadcab.com/blog/smart-contract-risks",
    #     # "https://www.weforum.org/stories/2024/07/smart-contracts-technology-cybersecurity-legal-risks/",
    #     "https://synodus.com/blog/fintech/defi-risk/",
    #     # "https://pmc.ncbi.nlm.nih.gov/articles/PMC10088710/",
    #     "https://arxiv.org/html/2312.01018v1",
    #     # "https://www.researchgate.net/publication/384258147_Risk_Management_in_DeFi_copy",   
    #     "https://www.rapidinnovation.io/post/smart-contracts-and-defi-transforming-decentralized-finance",
    #     "https://alphadevelopment.com/insights/decentralised-finance-defi-risks-how-to-protect-your-investments-and-navigate-the-crypto-landscape-safely/",
    #     # "https://www.futurelearn.com/info/courses/defi-exploring-decentralised-finance-with-blockchain-technologies/0/steps/256218"


    #     # Predictive Analytics in Finance
    #     "https://www.ibm.com/think/topics/predictive-analytics?utm_content=SRCWW&p1=Search&p4=43700075153304567&p5=p&p9=58700008227853819&gad_source=1&gclid=Cj0KCQiA8q--BhDiARIsAP9tKI1lITZmGEGWv5aJcAtVcC_UH0JHgUmNUiHjG0y1VNaPuO3m_iaY22AaAjkREALw_wcB&gclsrc=aw.ds",
    #     "https://www.highradius.com/resources/Blog/predictive-analytics-in-finance-guide/#:~:text=What%20is%20Predictive%20Analytics%20in,about%20future%20events%20or%20behaviors.",
    #     "https://www.bluent.com/blog/predictive-analytics-in-finance/",
    #     "https://www.velvetech.com/blog/predictive-analytics-in-finance/",
    #     "https://blog.workday.com/en-us/how-ai-is-shaping-predictive-analytics-in-finance.html",
    #     "https://www.linkedin.com/pulse/predictive-analytics-transforming-risk-management-ekkarit-7mn1c/",
    #     "https://morningconsult.com/2022/11/29/predictive-analytics-in-finance/",
    #     # "https://www.datarails.com/predictive-analytics-in-finance/",
    #     "https://nowcfo.com/predictive-analytics-in-finance-the-hidden-gem-in-financial-planning/",
    #     # "https://www.sciencedirect.com/science/article/pii/S2405918822000071",
    #     "https://www.fm-magazine.com/news/2023/may/4-ways-cfos-maximise-benefits-predictive-analytics/",
    #     "https://www.pwc.ch/en/insights/finance-transformation/predictive-analytics.html",
    #     "https://www.explo.co/blog/financial-predictive-analytics",
    #     "https://ramp.com/blog/predictive-analytics-in-finance/",
    #     "https://www.linkedin.com/pulse/artificial-intelligence-finance-predictive-smart-josyula-mba/",
    #     "https://sumatosoft.com/blog/predictive-analytics-in-finance-use-cases",
    #     "https://panintelligence.com/blog/predicitive-analytics-in-finance/",
    #     "https://yellow.systems/blog/predictive-analytics-in-finance",
    #     "https://svitla.com/blog/predictive-analytics-finance/",
    #     "https://www.jedox.com/en/blog/predictive-analytics/",
    #     "https://biztechmagazine.com/article/2022/02/what-predictive-analytics-and-how-can-it-help-financial-institutions-manage-risk",
    #     "https://online.hbs.edu/blog/post/predictive-analytics",
    #     "https://www.futureviewsystems.com/blog/predictive-vs-prescriptive-analytics-in-finance",
    #     "https://www.engineersmind.com/Services/AI-and-Data?gad_source=1&gclid=Cj0KCQiA8q--BhDiARIsAP9tKI2afm8bZAVIKsWLcVkrm1UC1MpX6u9I7WGBSaiCjTiocjYUes96EJwaArbmEALw_wcB",
    #     "https://innovyne.com/predictive-analytics-in-finance/",
    #     "https://www.dexmiq.com/blog-posts/predictive-analytics-in-finance-e083f",
    #     "https://blog.fabrichq.ai/predictive-analytics-in-finance-here-is-everything-you-should-know-70fba95b1cb5",
    #     "https://www.swapsupport.com/articles/predictive-analytics-finance",
    #     "https://www.qulix.com/about/blog/predictive-analytics-in-finance/",
    #     "https://www.softkraft.co/predictive-analytics-in-finance/",
    #     "https://saxon.ai/blogs/predictive-analytics-in-finance-use-cases-and-benefits/",
    #     "https://febi.ai/blog/how-to-use-predictive-analytics-for-financial-planning/",
    #     "https://www.finextra.com/blogposting/26980/al-driven-dashboards-how-predictive-al-analytics-is-shaping-the-future-financial-forecasting",
    #     "https://www.prioxis.com/blog/predictive-analytics-in-finance",
    #     "https://www.qlik.com/us/data-analytics/financial-analytics",
    #     "https://www.rapidinnovation.io/post/predictive-analytics-in-finance-anticipating-market-trends",
    #     "https://innovatureinc.com/the-advantage-of-predictive-analytics-in-finance/",
    #     'https://imarticus.org/blog/role-of-predictive-analytics-in-cash-flow-forecasting/'
    #     "https://www.zeni.ai/blog/financial-predictive-analytics",
    #     "https://neontri.com/blog/predictive-analytics-banking/",
    #     "https://www.pi.exchange/blog/predictive-analytics-in-banking",
    #     "https://binariks.com/blog/predictive-analytics-in-fintech/",
    #     # "https://www.sciencedirect.com/science/article/pii/S2405918822000071?ssrnid=4053319&dgcid=SSRN_redirect_SD"
    # ]
  
    pdf_folder = "pdfs_data"

    # Initialize the synthesis module based on the user's choice
    synthesis_module = DatasetSynthesisModule(
        use_url=args.use_url,
        use_pdf=args.use_pdf,
        knowledge_links=knowledge_base_links,
        pdf_folder=pdf_folder,
        embedding_model_name="all-MiniLM-L6-v2",
        api_key=groq_api_key
    )