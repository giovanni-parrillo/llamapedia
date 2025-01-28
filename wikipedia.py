import requests
from bs4 import BeautifulSoup, Comment, Tag
import logging
from typing import Optional
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def clean_html(html_content: str) -> str:
    """
    Cleans the HTML content by removing unnecessary elements and formatting the text.

    Args:
        html_content (str): The raw HTML content of the Wikipedia page.

    Returns:
        str: The cleaned HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove the header container
    header_container = soup.find('div', {'class': 'vector-header-container'})
    if header_container and isinstance(header_container, Tag):
        header_container.decompose()

    # Remove the footer container
    footer_container = soup.find('div', {'class': 'mw-footer'})
    if footer_container:
        footer_container.decompose()

    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove edit links
    for edit_link in soup.find_all('span', {'class': 'mw-editsection'}):
        edit_link.decompose()

    # Remove navigation links
    for nav_link in soup.find_all('div', {'class': 'navbox'}):
        nav_link.decompose()

    # Remove the body header
    body_header = soup.find('div', {'class': 'mw-body-header', 'class': 'vector-page-titlebar'})
    if body_header:
        body_header.decompose()

    
    return str(soup)


def get_wikipedia_html(search_query: str, language: str = 'en') -> Optional[str]:
    """
    Fetches the HTML content of the first Wikipedia search result for the given query in the specified language, saving it to output.html.

    Args:
        search_query (str): The search query to look up on Wikipedia.
        language (str): The language code for the Wikipedia site (default is 'en').

    Returns:
        Optional[str]: The HTML content of the first search result page, or an error message if the request fails.
    """
    search_url = f"https://{language}.wikipedia.org/w/index.php"
    params = {
        'search': search_query,
        'title': 'Special:Search',
        'fulltext': '1'
    }
    
    response = requests.get(search_url, params=params)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        logging.info("Fetched search results.")
        first_result = soup.find('div', {'class': 'mw-search-result-heading'})
        
        if first_result:
            link_tag = first_result.find('a')
            if link_tag and 'href' in link_tag.attrs:
                page_url = f"https://{language}.wikipedia.org" + link_tag['href']
                page_response = requests.get(page_url)
                logging.info(f"Fetching page content from: {page_url}")
                
                if page_response.status_code == 200:
                    logging.info("Fetched page content.")

                    clean_html_content = clean_html(page_response.text)

                    soup = BeautifulSoup(clean_html_content, 'html.parser')
                    page_title = soup.title.string if soup.title else "No title found"
                    logging.info(f"Page title: {page_title}")

                    # Replace internal links with actual external pages
                    substituted_urls = 0
                    for a_tag in soup.find_all('a', href=True):
                        # print(a_tag['href'])
                        href = a_tag['href']
                        if href.startswith('/wiki/'):
                            a_tag['href'] = f"https://{language}.wikipedia.org{href}"
                            substituted_urls += 1
                    
                    logging.info(f"Substituted {substituted_urls} internal links with external URLs.")

                    with open('output.html', 'w', encoding='utf-8') as file:
                        file.write(clean_html_content)
                    logging.info("Saved page content to output.html.")

                    return page_response.text
                else:
                    logging.error(f"Failed to fetch the page. Status code: {page_response.status_code}")
                    return None
            else:
                logging.info("No valid link found in the search result.")
                return None
        else:
            logging.info("No search results found.")
            return None
    else:
        logging.error(f"Failed to perform the search. Status code: {response.status_code}")

def find_sources(html_content: str) -> list[str]:
    """
    Extracts the URLs of the sources cited in the Wikipedia page.

    Args:
        html_content (str): The HTML content of the Wikipedia page.

    Returns:
        list[str]: A list of URLs of the sources cited in the page.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    sources = soup.find_all('a', {'class': 'external text'})
    logging.info(f"Found {len(sources)} sources.")
    return [source['href'] for source in sources]

def convert_to_simple_text(html_content: str) -> str:
    """
    Converts the HTML content to a simple text format by removing all HTML tags.

    Args:
        html_content (str): The HTML content to convert.

    Returns:
        str: The plain text content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()
    text_content = text_content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(text_content)
    
    return text_content

def answer(text_content: str, query:str) -> str:
    """
   Retrieves specific information from the text content and summarizes it.

    Args:
        text_content (str): The text content to summarize.

    Returns:
        str: The summarized text.
    """

    response = ollama.chat(
            model="llama3.1:8B",
            messages=[
                {"role": "system", "content": f"You are an information retriever from a specified wikipedia article. have access to the latest Wikipedia article, which I will now provide. Read the wikipedia article and answer to a precise question from the user. Try to be brief precise and reliable. Possibly, directly cite the article and cite the relevant sources. FOCUS ON WHAT IS WRITTEN IN THE WIKIPEDIA ARTICLE. Here is the article: "},
                {"role":"user", "content": text_content},
                {"role": "user", "content": query}
            ],
            stream = True
        )
    
    # print(text_content)

    answer = ""
    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)
        answer += chunk['message']['content']

    


    return answer  # Clean up any formatting


def main():

    search_query = input("Cerca su wikipedia: ")
    language = "it"  # Change this to the desired language code
    html_content = get_wikipedia_html(search_query, language)
    # html_content = clean_html(html_content)

    find_sources(html_content)
    html_content = clean_html(html_content)
    html_content = convert_to_simple_text(html_content)

    while True:
        test_query = input("\n >  ")
        answer(html_content, test_query)
    
    # for source in sources:
    #     print(source)

if __name__ == "__main__":
    main()

