import requests
from bs4 import BeautifulSoup
# from googleapiclient.discovery import build
import re
from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.schema import Document
from duckduckgo_search import DDGS

def clean_text(text):
    """Remove extra whitespace and special characters"""
    return re.sub(r'\s+', ' ', text).strip()

def crawler(url):
    try:
        # Fetch page with headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # Extract content
        content = {
            'title': clean_text(soup.title.string) if soup.title else '',
            'main_content': [],
            'headings': [],
        }
        
        # Get headings
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            if heading.text:
                content['headings'].append(clean_text(heading.text))
        
        # Get main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            paragraphs = main_content.find_all('p')
        else:
            paragraphs = soup.find_all('p')
            
        for p in paragraphs:
            text = clean_text(p.text)
            if len(text) > 50:  # Filter out short snippets
                content['main_content'].append(text)
        
        return content
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def web_search(query, num_results=5):
    """
    Perform web search using DuckDuckGo
    """
    try:
        results = DDGS().text(query, max_results=5)
        print (results)
        return results
    except Exception as e:
        print(f"Search Error: {e}")
        return None

# def test():
#     # You need to get these from Google Cloud Console
#     API_KEY = 'YOUR_API_KEY'
#     CSE_ID = ''
    
#     try:
#         results = google_search('jet fuel', API_KEY, CSE_ID, num=10)
#         for result in results:
#             print(f"Title: {result['title']}")
#             print(f"URL: {result['link']}")
#             print(f"Snippet: {result['snippet']}\n")
    
    # except Exception as e:
    #     print(f"Error: {e}")


def process_content_for_llm(content):
    """Format crawler content for LLM"""
    return f"""
    Title: {content['title']}
    
    Main Content:
    {' '.join(content['main_content'][:3])}
    
    Key Sections:
    {' '.join(content['headings'])}
    """

def query_with_langchain(content, question, model_name="llama3.2:3b"):
    """Query Ollama through LangChain"""
    # Initialize Ollama
    llm = Ollama(model=model_name)
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Context: {context}
        
        Question: {question}
        
        Answer: Let me analyze this information and answer your question.
        """
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Process content
    context = process_content_for_llm(content)
    
    # Execute chain
    response = chain.run(context=context, question=question)
    return response

def analyze_webpage(url, question):
    """Combine crawler with LangChain query"""
    content = crawler(url)
    if content:
        return query_with_langchain(content, question)
    return None



def run():
    while (True):
        query = input("Enter your query: ")
        results = web_search(query)
        
        url = []
        #display results
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['href']}")
            url.append(result['href'])
            # print(f"Snippet: {result['snippet']}\n")
        
        print("\n\n\n",url)
        for i in url:    
            response = analyze_webpage(i, query)
            print(f"Answer: {response}\n")
        
        cont = input("Do you want to continue? (y/n): ")
        if cont.lower() != 'y':
            break



if (__name__ == "__main__"):
    run()
