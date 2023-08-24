import time

import mkwikidata  # https://pypi.org/project/mkwikidata/
import requests

# try out Wikidata Query Service:
# https://query.wikidata.org/#SELECT%20%3Fitem%20%3FitemLabel%0AWHERE%0A%7B%0A%20%20wd%3AQ8880%20%28wdt%3AP31%2a%7Cwdt%3AP279%2a%29%20%3Fitem%20.%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22en%22.%20%7D%0A%7D%0A%0ALIMIT%20100
# Doc: https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial


def format_to_query(wikidata_id, method="strict"):
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

                ?item wikibase:statements ?outcoming .
                ?item wikibase:sitelinks ?sitelinks .

                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
           }}

        GROUP BY ?item ?itemLabel ?sitelinks ?outcoming # ?midLabel
        ORDER BY DESC(?sitelinks)

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

    ## allowing any combination of instance_of and subclass_of
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


def get_wikidata_categories(entity, method):
    wikidata_id = None
    wikibase_shortdesc = ""

    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'titles': entity,
        'prop': 'pageprops',
        'redirects': True,
        # 'exintro': True,
        # 'explaintext': True,
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
                                method=method
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
