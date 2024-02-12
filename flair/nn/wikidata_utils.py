import time

import mkwikidata  # https://pypi.org/project/mkwikidata/
import requests
import json
import urllib


# try out Wikidata Query Service:
# https://query.wikidata.org/#SELECT%20%3Fitem%20%3FitemLabel%0AWHERE%0A%7B%0A%20%20wd%3AQ8880%20%28wdt%3AP31%2a%7Cwdt%3AP279%2a%29%20%3Fitem%20.%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22en%22.%20%7D%0A%7D%0A%0ALIMIT%20100
# Doc: https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial


def format_to_query(wikidata_id,
                    method="strict",
                    add_occupation=True,
                    add_field_of_work=False,
                    add_country = False):
    add_occupation_string = """
            UNION
               {{ ?id wdt:P106 ?item .
            }}
          """
    add_field_of_work_string = """
            UNION
             {{ ?id wdt:P101 ?item .
            }}
          """

    add_country_string = """
            UNION
              {{
               ?id wdt:P17 ?item .
              }}
          """

    ## only allow ONE or no instance_of and ONE subclass_of jump
    if method == "strict":
        query = f"""
            SELECT ?item ?itemLabel ?sitelinks ?outcoming # ?midLabel # (COUNT(?mid) AS ?distance)
           WHERE
           {{BIND(wd:{wikidata_id} AS ?id)
                {{
                  ?id wdt:P31 ?item .
                }}
                UNION
                {{
                  ?id wdt:P31 ?mid .
                  ?mid wdt:P279 ?item .
                }}

                {f'{add_occupation_string}' if add_occupation else ''}

                {f'{add_field_of_work_string}' if add_field_of_work else ''}
                
                {f'{add_country_string}' if add_country else ''}

                ?item wikibase:statements ?outcoming .
                ?item wikibase:sitelinks ?sitelinks .

                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
           }}

        GROUP BY ?item ?itemLabel ?sitelinks ?outcoming #?midLabel
        ORDER BY DESC(?sitelinks)
        #ORDER BY ASC(?itemLabel)
        #ORDER BY DESC(?outcoming)



        LIMIT 50

        """

    ## only one jump for instance_of and subclass_of, adding country and part_of relations
    if method == "only_one_level_up":
        query = f"""
            SELECT ?item ?itemLabel ?sitelinks ?outcoming # ?midLabel # (COUNT(?mid) AS ?distance)
           WHERE
           {{BIND(wd:{wikidata_id} AS ?id)
                {{
                  ?id wdt:P31 ?item .
                }}
                UNION
                {{
                  ?id wdt:P279 ?item .
                }}

                UNION
                {{
                  ?id wdt:P361 ?item .
                }}

                {f'{add_occupation_string}' if add_occupation else ''}

                {f'{add_field_of_work_string}' if add_field_of_work else ''}
                
                {f'{add_country_string}' if add_country else ''}

                ?item wikibase:statements ?outcoming .
                ?item wikibase:sitelinks ?sitelinks .

                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
           }}

        GROUP BY ?item ?itemLabel ?sitelinks ?outcoming #?midLabel
        #ORDER BY DESC(?sitelinks)
        #ORDER BY ASC(?itemLabel)


        LIMIT 50

        """

    ## initial version: only instance_of OR subclass_of in series allowed
    if method == "only_separate_paths":
        line = f"wd:{wikidata_id} (wdt:P31*|wdt:P279*) ?item ."  # either several of P31 or several of P279, no mix

        query = f"""
       SELECT ?item ?itemLabel ?sitelinks
       WHERE
       {{
         {line}
         ?item wikibase:sitelinks ?sitelinks .
         SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
       }}
       ORDER BY DESC (?sitelinks)
       LIMIT 100
       """

    ## too many classes: allowing any combination of instance_of and subclass_of
    if method == "allow_combination":
        query = f"""

       SELECT ?item ?itemLabel ?sitelinks ?outcoming (COUNT(?mid) AS ?distance)
       WHERE
       {{
         wd:{wikidata_id} wdt:P31* ?mid .
         ?mid (wdt:P31|wdt:P279)+ ?item . 
         # ?mid (wdt:P279)+ ?item . 

         ?item wikibase:statements ?outcoming .
         ?item wikibase:sitelinks ?sitelinks .
         SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }}

      GROUP BY ?item ?itemLabel ?sitelinks ?outcoming
      ORDER BY ?distance DESC(?sitelinks)

      LIMIT 50

      """

    return query


def extract_list_from_result(result):
    links = []
    names = []
    already_added = []
    for r in result["results"]["bindings"]:
        item_link = r["item"]["value"]
        item_name = r["itemLabel"]["value"]
        if item_name not in already_added:
            links.append(item_link)
            names.append(item_name)
            already_added.append(item_name)

    return links, names

def get_pageviews_of_entity_precomputed(entity, source_file_path):
    pageviews = None

    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'titles': entity,
        'prop': 'pageprops',
        'redirects': True,

    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        page = next(iter(data['query']['pages'].values()))
        wikipedia_id = page["pageid"]
    except:
        wikipedia_id = None
        return None

    with open(source_file_path) as file:
        for line in file:
            if line.startswith(f"{wikipedia_id} "):
                pageviews = int(line.split(" ")[1].strip())
                return pageviews

    return pageviews


def get_url_from_pageID(pageID):
    entity_label = ""
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'prop': 'info',
        'pageids': pageID,
        'inprop': 'url',
        'format': 'json',
        # 'exintro': True,
        # 'explaintext': True,
    }

    try:
        response = requests.get(url, params=params).json()
        page = next(iter(response['query']['pages'].values()))
        wikipedia_url = urllib.parse.unquote(page["fullurl"])
        entity_label = wikipedia_url[len("https://en.wikipedia.org/wiki/"):]
    except:
        try:
            time.sleep(10)
            response = requests.get(url, params=params).json()
            page = next(iter(response['query']['pages'].values()))
            wikipedia_url = urllib.parse.unquote(page["fullurl"])
            entity_label = wikipedia_url[len("https://en.wikipedia.org/wiki/"):]
        except:
            entity_label = ""

    return entity_label

def get_wikidata_categories(entity, method, add_occupation, add_field_of_work, add_country):
    wikidata_id = None
    wikibase_shortdesc = ""

    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'titles': entity,
        'prop': 'pageprops',
        'redirects': True,

    }

    response = requests.get(url, params=params)
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    wikidata_title = page["title"]
    pageprops = page.get("pageprops", None)
    if pageprops:
        wikidata_id = pageprops.get("wikibase_item", None)
        wikibase_shortdesc = pageprops.get("wikibase-shortdesc", "")

    if wikidata_id:

        wikidata_url = f"https://www.wikidata.org/wiki/{wikidata_id}"

        query = format_to_query(wikidata_id=wikidata_id,
                                method=method,
                                add_occupation=add_occupation,
                                add_field_of_work=add_field_of_work,
                                add_country=add_country,
                                )
        try:
            query_result = mkwikidata.run_query(query, params={})
        except:
            time.sleep(10)
            query_result = mkwikidata.run_query(query, params={})

        class_links, class_names = extract_list_from_result(result=query_result)

    else:
        wikidata_url = None
        class_names = []
        wikidata_title = ""

    return {"wikidata_id": wikidata_id,
            "wikidata_url": wikidata_url,
            "class_names": class_names,
            "wikidata_title": wikidata_title,
            "wikibase_description": wikibase_shortdesc,
            }

def get_sitelinks_of_entity(entity):
    wikidata_id = None

    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'titles': entity,
        'prop': 'pageprops',
        'redirects': True,

    }

    response = requests.get(url, params=params)
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    wikidata_title = page["title"]
    pageprops = page.get("pageprops", None)
    if pageprops:
        wikidata_id = pageprops.get("wikibase_item", None)
        wikibase_shortdesc = pageprops.get("wikibase-shortdesc", "")

    if wikidata_id:
        wikidata_url = f"https://www.wikidata.org/wiki/{wikidata_id}"

    query = f"""
             SELECT ?item
                    ?itemLabel
                    ( COUNT( ?sitelink ) AS ?sitelink_count )
       
             WHERE {{
                BIND(wd:{wikidata_id} AS ?item).
                ?sitelink schema:about ?item. # sitelink about the item
                                              # label in English
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}

                }}
             GROUP BY ?item ?itemLabel
             """

    try:
        query_result = mkwikidata.run_query(query, params={})
    except:
        time.sleep(10)
        query_result = mkwikidata.run_query(query, params={})

    try:
        return query_result["results"]["bindings"][0]["sitelink_count"]["value"]
    except:
        return None # Fallback

#### MAPPING THE WIKIDATA CLASSES TO OUR NER LABELS ####

org_classes = ["organization", "political party", "political organization", "confederation", "sports club",
               "political party", "business", "public company", "type of organisation",
               "national sports team", "association football club", "sports team", "government organization"
               ]
loc_classes = ["state", "country", "city", "classification of human settlements", "human settlement",
               "geographic entity",
               "physical location", "U.S. state", "historical country", "island", "geographic region"]
per_classes = ["human", "person",
               "Wikimedia human name disambiguation page",
               ]

# if after that and nothing found, you could look for substrings:
# org_strings = ["organization", "business",
#                "team",
#                "football club", "sports club", "political party",
#                "Broadcasting Company"]
# loc_strings = ["city", "town", "continent", "state of", "province of", "comune of"]
# per_strings = ["human name disambiguation page"]
# else: MISC



def map_wikidata_list_to_ner(wikidata_classes):

    tag_dict = {"ORG": False,
                "LOC": False,
                "PER": False,
                "MISC": False}

    for c in wikidata_classes:
        if c in per_classes:
            tag_dict["PER"] = True
        if c in loc_classes:
            tag_dict["LOC"] = True
        if c in org_classes:
            tag_dict["ORG"] = True

    # if no rule applied? --> MISC
    if not True in tag_dict.values():
        tag_dict["MISC"] = True

    # if more than one rule applied? --> MISC
    if sum(tag_dict.values()) > 1:
        tag_dict["MISC"] = True
        tag_dict["PER"] = False
        tag_dict["LOC"] = False
        tag_dict["ORG"] = False

    ner_label = [k for k,v in tag_dict.items() if v == True][0]

    return ner_label, tag_dict

if __name__ == "__main__":

    for e in ["Berlin", "Chris_Harris_(cricketer)", "Council_of_the_European_Union"]:
        print("--")
        print(e)

        # print("Nr of Sitelinks:", get_sitelinks_of_entity(e))
        #
        # print("The list that we used:")
        # item_info = get_wikidata_categories(e, method="strict", add_occupation=True, add_field_of_work=False)
        # print(json.dumps(item_info, indent = 4))
        #
        # print("NER:", map_wikidata_list_to_ner(item_info["class_names"])[0])

        #print("\ndifferent combinations of the relations, one lead to too many, one to too little:")
        #print(get_wikidata_categories(e, method= "only_one_level_up", add_occupation=True, add_field_of_work=False))
        #print(get_wikidata_categories(e, method= "allow_combination", add_occupation=True, add_field_of_work=False))

        print(get_pageviews_of_entity_precomputed(e,
                                                   "/vol/tmp/ruckersu/data/wikipedia_pageviews/en_wikipedia_ranking.txt"
                                                   ))



