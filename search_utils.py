import requests
import xml.etree.ElementTree as ET

def construct_query_baseline(keywords):
    """
    Constructs a basic PubMed query string using AND operators for a given list of keywords.
    """
    return ' AND '.join(keywords)

def ncbi_querybuilder(keywords):
    """
    Constructs a PubMed query string using AND operators and keyword formatting to replace spaces with "+".
    """
    # Return an empty string if no keywords are provided
    if not keywords or len(keywords) == 0:
        return ""
    return ' AND '.join(keyword.replace(' ', '+') for keyword in keywords)

def ncbi_query(ncbi_retmax, query_term, min_date, max_date):
    """
    Queries the NCBI e-utils API to retrieve article IDs (PMIDs) based on the query term.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    db_param = "db=pubmed"
    retmax_param = f"retmax={ncbi_retmax}"
    term_param = f"term={query_term}"
    date_range_param = f'mindate={min_date}&maxdate={max_date}'
    full_url = f"{base_url}?{db_param}&{term_param}&{retmax_param}&{date_range_param}"

    # Call NCBI's e-utils API based on the constructed full_url
    response = requests.get(full_url)

    if response.status_code == 200:
        content = ET.fromstring(response.content)
        # Extract PMIDs from the API response
        pmid_list = [id_elem.text for id_elem in content.findall('.//IdList/Id')]
        return pmid_list
    else:
        print(f"Unsuccessful for {query_term}. Status code: {response.status_code}")
        return []

def ncbi_title_abstract_query(pmid_list):
    """
    Fetches article details (PMID, title, abstract) from PubMed based on a list of PMIDs.
    """
    efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(pmid_list)}&retmode=xml"

    response = requests.get(efetch_url)

    result = []
    if response.status_code == 200:
        root = ET.fromstring(response.content)

        for article in root.findall('.//PubmedArticle'):
            item = {}

            # Get the PMID
            pmid_elem = article.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ''
            item['pmid'] = pmid

            # Get the article title
            article_title_elem = article.find('.//ArticleTitle')
            article_title = article_title_elem.text if article_title_elem is not None else ''
            item['title'] = article_title

            # Get the article abstract
            abstract_elem = article.find('.//Abstract')
            abstract_full_text = ''
            if abstract_elem:
                for abs_nested_ele in abstract_elem:
                    if abs_nested_ele.tag == 'AbstractText':
                        if abs_nested_ele.attrib and ('Label' in abs_nested_ele.attrib):
                            abstract_full_text += abs_nested_ele.attrib['Label'] + ': '
                        if abs_nested_ele.text:
                            abstract_full_text += abs_nested_ele.text
                        else:
                            for ele_next in abs_nested_ele.itertext():
                                abstract_full_text += ele_next

            item['abstract'] = abstract_full_text
            result.append(item)
    else:
        print(f"Unsuccessful for {pmid_list}. Status code: {response.status_code}")
    return result

if __name__ == '__main__':
    pass
