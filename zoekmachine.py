#%matplotlib inline
from IPython.display import display, clear_output
from IPython.core.display import HTML
from ipywidgets import widgets, interact
from os import listdir
from os.path import isfile
from matplotlib.ticker import MaxNLocator
from stop_words import get_stop_words
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter
from PIL import Image
from elasticsearch import Elasticsearch, helpers

import math
import matplotlib.pyplot as plt
import numpy as np
import snowballstemmer
import sys
import json

def search(query, advanced=False):
    """
    Given a query it returns the results from an ElasticSearch query
    """
    if advanced:

        must = []

        if query[0]:
            must = [{"type": {"value": i}} for i in query[0]]
        # Advanced search query
        q = {"query":
                {"filtered":
                    {"query": {
                        "multi_match":
                            {"query" : query[1],
                             "type" : "cross_fields",  # with 'and' operator this is strict
                             "fields" : query[2],
                             "operator" : 'and'
                            }
                        },
                     "filter":
                        {'and':
                            [{"range":
                                {"year":
                                    {"gte":query[3][0],
                                     "lte":query[3][1]
                                    }
                                }
                              },
                             {"bool":
                                 {"should":must
                                 }
                             }
                              ]
                         }
                    }
                }
            }
    else:
        # Simple search
        q = {'query':
                {'multi_match':
                    {'query' : query,
                     'type' : 'cross_fields',  # with 'and' operator
                     'fields' : ['title', 'text'],
                     'operator' : 'and'
                     }
                 }
             }
    res = es.search(index='telegraaf', size=50, body=q)
    return res


def position_sentences(positions, text, m):
    """
    Return a sentence in which multiple m words
    from the text occur, based on a list of positions.
    """
    mini = positions[0]
    maxi = mini
    for i in positions[1:]:
        if i > mini and i <= mini + m:
            maxi = i
    diff = int(math.floor((m - (maxi - mini)) / 2))
    return '...'+' '.join(text[mini-diff:maxi+diff])+'...'


def extract_description(query, text, m):
    """
    Given a query, select m words from the text that
    contain words from the query.
    """
    query = query.split()
    stext = text.split(' ')
    positions = []
    # get the word position
    for word in query:
        for i,term in enumerate(stext):
            if word in term:
                positions.append(i)

    for i, word in enumerate(stext):
        if i in positions:
            stext[i] = '<b>' + word + '</b>'
        else:
            stext[i] = word

    positions = [i for i in sorted(positions) if i > 7]

    # If word(s) appeared in text, return these sentences
    if positions:
        description = position_sentences(positions, stext, m)
    # If the word only occured in title, return first sentence/part of first sentence
    else:
        description = ' '.join(stext[:15]) + '...'
    return description


def result_page(query, total_hits, hits):
    """
    Given a query and its hits, return what information
    to output to the user in a SERP.
    """
    total = widgets.HTML('Total hits: '+str(total_hits)+" Shown: 10")
    results = []
    descriptions = []
    for elem in hits:
        if elem['_source']['title'] == '':
            results.append(widgets.HTML('<h3><a href="http://kranten.kb.nl/view/article/id/'+
                                        str(elem['_id'])+'" target="_blank">No Title Available</a></h3>'))
        else:
            results.append(widgets.HTML(value = '<h3><a href="http://kranten.kb.nl/view/article/id/'+
                                        str(elem['_id'])+'" target="_blank">'+elem['_source']['title']+'</a></h3>'))
        results.append(widgets.HTML(extract_description(query, elem['_source']['text'],15)))
        results.append(widgets.HTML("Score: " + str(elem['_score'])))
    return results

def create_timeline(years, dates):
    """
    Display the years or dates from the hits
    on a timeline.
    """
    ax = plt.figure(figsize=(15,4)).gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if len(set(years)) == 1:
        xlabel = "Months of the year " + str(years[0])
        bins = [ int(date[:2]) for date in dates ]
        #counted = Counter(months)
        names = ['           Januari','             Februari','         Maart',
                 '        April', '        Mei','        Juni','        Juli',
                 '               Augustus','                 September',
                 '               Oktober','                November',
                 '                December']
        xlim = [1,13]
        plt.xticks(range(1,13),names)
        bin_range = range(1,13)
    else:
        bins = years
        bin_range = range(min(years),max(years)+1)
        xlabel = "Years"
        xlim = [min(years),max(years)]

    ax.set_xlim(xlim)

    # Mooi roze is niet leelijk
    plt.hist(bins, bins=bin_range, color='Crimson')
    plt.xlabel(xlabel)
    plt.ylabel("Number of documents")
    plt.show()

def create_wordcloud(text, n):
    """
    Display a wordcloud with at most n words, generated
    from the given text.
    """
    # Filter words to use for the wordcloud, by stemming and stop words removal
    stop_words = get_stop_words("dutch")
    stemmer = snowballstemmer.stemmer("dutch")
    text = [word for word in text if word.lower() not in stop_words]
    text = stemmer.stemWords(text)

    # Plot wordcloud
    wordcloud = WordCloud(background_color="white", min_font_size = 10,
                          max_words = n).generate(" ".join(text))
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def get_count(ty, query, fields,time):
    """Get the count of a type, query, fields, and time."""

    return es.count(index='telegraaf',
                    body={'query':{"filtered":{"query":
                                                  {"multi_match":
                                                      {"query" : query,
                                                          "type" : "cross_fields",
                                                          "fields" : fields,
                                                          "operator" : 'and'
                                                       }
                                                  },
                                                  "filter":
                                                      {"bool":
                                                           {"must":
                                                               [{"term": {"_type": ty}},
                                                                {"range":{"year":{
                                                                   "gte":time[0],
                                                                   "lte":time[1]}}}]
                                                            }
                                                      }
                                              }
                                    }
                         })['count']

#==============================================================
#        DOCUMENT TYPES: CHECKBOXES - TEXT VALUES
#==============================================================

# this should probably be rewritten to not include a class. It's overkill
# Just setting them as a ??_text should be suficient.
class doc_widget:

    def __init__(self,name,value):
        self.value = value
        self.name = name
        self.widget = widgets.Text(str(self.name)+' ('+str(self.value)+')',
                                   width='220px', disabled=True)
        # Disable=True to make sure the user can't change the value

    def set_value(self, value):
        self.value = value
        self.widget = widgets.Text(str(self.name)+' ('+str(self.value)+')',
                                   width='220px', disabled=True)
        # Disable=True to make sure the user can't change the value

#=====================================================================

def advanced_options(sender):
    if sender['new'] == True:
        container_adv.visible = True
        # Show the advanced settings

    elif sender['new'] == False:
        container_adv.visible = False
        # Hide the advanced settings

def show_timeline(sender):
    #res = RES
    if sender['new']==True:
        if timeline_check.value:
            timeline_years = [hit['_source']['year'] for hit in res['hits']['hits']]
            dates = [hit['_source']['date'] for hit in res['hits']['hits']]
            if timeline_years != []:
                create_timeline(timeline_years, dates)
    elif sender['new']==False:
        clear_output()

#=====================================================================

def handle_submit(sender):
    """
    This function handles the search after the button has been
    pressed or the search field has been submitted
    """
    # Define SERP as a global variable, such that for the first
    # loop it doesn't try to close the non-existing SERP
    global SERP
    if SERP != '':
        SERP.close()
    clear_output()


    # SIMPLE SEARCH
    if container_adv.visible == False:
        display('Zoeken..')
        res = search(text.value)
        clear_output()
        results = result_page(text.value, res['hits']['total'], res['hits']['hits'])
        RES = res
        if results == []:
            SERP = widgets.VBox(children=(widgets.HTML('', height='20px'),
                                          widgets.HTML("<center>Geen zoekresultaten \
                                          zijn gevonden. Probeer het opnieuw</center>")))
        else:
            pages = []
            page = []
            for i, result in enumerate(results):
                if (i+1) is len(results):
                    page.append(result)
                    pages.append(widgets.VBox(children=tuple(page)))
                elif (i)%30 is 0 and not i is 0:
                    pages.append(widgets.VBox(children=tuple(page)))
                    page = []
                page.append(result)

#             SERP = widgets.VBox(children=(tuple([i for i in results])))
#             children = [widgets.Text(description=name) for name in list]
            SERP = widgets.Tab(children=pages)
        display(SERP)

#----------------------------------------------------------------------------
    # ADVANCED SEARCH
    else:
        display('Zoeken..')

        # SELECTED DOCUMENT TYPES
        types = []
        if ad_check.value:
            types.append('advertentie')
        if ar_check.value:
            types.append('artikel')
        if io_check.value:
            types.append('illustratie met onderschrift')
        if fb_check.value:
            types.append('familiebericht')
        fields = []

        # SELECTED TEXT FIELDS
        if title_check.value:
            fields.append('title')
        if text_check.value:
            fields.append('text')

        # Search. The values for the slider and text are immediately inputted
        res = search([types,text.value,fields,years.value], advanced=True)
        RES = res
        clear_output()
        results = result_page(text.value, res['hits']['total'], res['hits']['hits'])

        # Print the Search results
        if results == []:
            SERP = widgets.VBox(children=(widgets.HTML('', height='20px'),
                                          widgets.HTML("<center>Geen zoekresultaten \
                                          zijn gevonden. Probeer het opnieuw</center>")))
        else:
            pages = []
            page = []
            for i, result in enumerate(results):
                if (i+1) is len(results):
                    page.append(result)
                    pages.append(widgets.VBox(children=tuple(page)))
                elif (i)%30 is 0 and not i is 0:
                    pages.append(widgets.VBox(children=tuple(page)))
                    page = []
                page.append(result)
            SERP = widgets.Tab(children=pages)
        display(SERP)

        # update the type doc numbers
        for i in doc_types:
            doc_types[i].widget.value= doc_types[i].name+' ('+str(get_count(i,text.value,fields,years.value))+')'

        # DISPLAY TIMELINE
        if timeline_check.value:
            timeline_years = [int(hit['_source']['year']) for hit in res['hits']['hits']]
            dates = [hit['_source']['date'] for hit in res['hits']['hits']]
            if timeline_years != []:
                create_timeline(timeline_years, dates)

        #DISPLAY WORDCLOUD
        if wordcloud_check.value:
            total_text = []
            for hit in res['hits']['hits']:
                total_text.extend(hit['_source']['text'].split())
            create_wordcloud(total_text, 0)

################### VERNA

def show_search_engine():

  global es
  global SERP
  global text
  global text_check
  global title_check
  global ad_check
  global ar_check
  global io_check
  global fb_check
  global doc_types
  global years
  global timeline_check
  global wordcloud_check
  global container_adv

  SERP = ''

  HOST = 'http://localhost:9200/'
  es = Elasticsearch(hosts=[HOST],retry_on_timeout=True)

  documents = ['./Telegraaf/'+i for i in listdir('./Telegraaf') if not isfile(i)]

  # Determine values for the year facets
  agg={"aggs" : {
          "_source" : {
              "terms" : { "field" : "year", "size" : len(documents) }}}}

  agg2={"aggs" : {
          "_type" : {
              "terms" : { "field" : "_type" }}}}

  # Get field values for the year
  res = es.search(index='telegraaf', body=agg)
  unique_years_string = sorted([ "%s (%d documents)" % (item['key'], item['doc_count'])
                        for item in res['aggregations']['_source']['buckets']])
  unique_years = sorted([item['key'] for item in res['aggregations']['_source']['buckets']])

  # Get field values for document type
  res = es.search(index='telegraaf', body=agg2)
  unique_doc_types_string = [ "%s (%d documents)" % (item['key'], item['doc_count'])
                              for item in res['aggregations']['_type']['buckets']]
  unique_doc_types = [item['key'] for item in res['aggregations']['_type']['buckets']]

  # Initialise SERP such that on the first the system doesn't return an error

  # SEARCH FIELD AND BUTTON
  text = widgets.Text(placeholder='Vul een zoekterm in')
  search_button = widgets.Button(description="Search")

  # ADVANCED SEARCH TOGGLE BUTTON
  advanced = widgets.ToggleButton(description="Toggle Advanced Search",
                                  width='220px', button_style='success', value=True)


  # INITIAL SEARCHBAR CONTAINER
  container = widgets.HBox((widgets.HTML('',width='130px'),
                            widgets.HTML('Zoektermen:'),
                            text,
                            widgets.HTML(''),
                            search_button,
                            widgets.HTML('',width='50px'),
                            advanced))
  container.layout.align_items='center'

  #===================================================================
  #                   TEXT FIELDS: CHECKBOXES
  #===================================================================

  # TEXT FIELD CHECKBOX

  text_check = widgets.Checkbox(value=True, width='50px')
  textfield = widgets.HBox(children=(widgets.HTML('',width='20px'),
                                     widgets.HTML('Text'),
                                     text_check))
  textfield.layout.align_items='center'

  # TITLE FIELD CHECKBOX
  title_check = widgets.Checkbox(value=True, width='50px')
  titlefield = widgets.HBox(children=(widgets.HTML('',width='20px'),
                                      widgets.HTML('Title'),
                                      title_check))
  titlefield.layout.align_items='center'

  # FINAL CONTAINER
  c_textfields = widgets.VBox(children=(widgets.HTML('Welke zoekvelden?'),
                                        textfield,
                                        titlefield))
  titlefield.layout.align_items='center'

  # THE TEXT VALUES
  doc_types = {}
  for i in unique_doc_types:
      doc_types[i]  = doc_widget(i,0)

  # CHECKBOXES
  ad_check = widgets.Checkbox(value=True, width='20px')
  ar_check = widgets.Checkbox(value=True, width='20px')
  io_check = widgets.Checkbox(value=True, width='20px')
  fb_check = widgets.Checkbox(value=True, width='20px')

  # FIRST ROW
  row1 = widgets.HBox(children=(ad_check,
                                doc_types['advertentie'].widget,
                                widgets.HTML('',width='20px'),
                                ar_check,
                                doc_types['artikel'].widget))
  row1.layout.align_items='center'

  # SECOND ROW
  row2 = widgets.HBox(children=(io_check,
                                doc_types['illustratie met onderschrift'].widget,
                                widgets.HTML('',width='20px'),
                                fb_check,
                                doc_types['familiebericht'].widget))
  row2.layout.align_items='center'

  # FINAL CONTAINER
  c_types = widgets.VBox(children=(widgets.HTML('Welke type documenten?'),
                                   row1,
                                   row2))

  #==================================================================================
  #                  TIME PERIOD SLIDER - TIMELINE - WORDCLOUD
  #==================================================================================

  # TIME PERIOD SLIDER
  years = widgets.IntRangeSlider(value=[int(unique_years[0]), int(unique_years[-1])],
                                 min=int(unique_years[0]),
                                 max= int(unique_years[-1]),
                                 step=1)

  # TIMELINE  - WORDCLOUD
  timeline_check = widgets.Checkbox(value=False, width='50px')
  wordcloud_check = widgets.Checkbox(value=False, width='50px')
  c_extra_options = widgets.HBox(children=(widgets.HTML("Tijdlijn"),
                                           timeline_check,
                                           widgets.HTML("Wordcloud"),
                                           wordcloud_check))
  c_extra_options.layout.align_items = 'center'
  c_extra_options.layout.justify_content = 'center'

  # FINAL CONTAINER
  c_slide = widgets.VBox(children=(widgets.HTML("Kies tijdsperiode:"),
                                   years,
                                   c_extra_options
                                  ))
  #=====================================================================
  # Advanced settings container
  container_adv = widgets.HBox((c_textfields,
                                c_types,
                                c_slide), )

  container_adv.layout.justify_content = 'space-around'
  container_adv.visible=True


  # Display the main items and set their submits/change response
  display(container)
  display(container_adv)

  # Set the widgets their response
  text.on_submit(handle_submit)
  search_button.on_click(handle_submit)
  advanced.observe(advanced_options,names='value')
