import unittest
import torch

from flair.data import Sentence
from unittest.mock import patch, MagicMock

from flair.embeddings import (
    GazetteerEmbeddings,
)


class GazetteerEmbeddingsTest(unittest.TestCase):
    full_match_dict = {'!Xunkhwesa Combined School': ['ORG'],
                       "'t Vliegend Peert Centre for Old Arts": ['ORG'],
                       ' National Archives and Library Agency': ['ORG'],
                       ', Berhampur': ['ORG'], '+EU': ['ORG'],
                       'America de Manta': ['ORG'],
                       'หนังสือพิมพ์ผู้จัดการ': ['ORG'],
                       'Sandys': ['ORG'],
                       'I': ['ORG'],
                       'Land Tenure Reform Association': ['ORG'],
                       'Sandys Fort Spring': ['ORG', 'LOC'],
                       '!Bang!': ['ORG'],
                       '""Onderzoek - en Documentatiecentrum Beweging van Mensen met Laag Inkomen en Kinderen"':
                           ['ORG'],
                       '"Mons. Aurelio Sorrentino" Diocesan Museum': ['ORG'],
                       '"Octavian Goga" Cluj County Library': ['ORG'],
                       '"Pedro Arrupe" Political Training Institute': ['ORG'],
                       '"Red Terror" Martyrs\' Memorial Museum': ['ORG'],
                       '#Minas para Todos': ['ORG'],
                       '#Nome?': ['ORG'],
                       "' Cogozzo - Don Eugenio Mazzi '": ['ORG'],
                       "' Villa Saviola '": ['ORG'],
                       "''A. Ilvento''-Grassano": ['ORG'],
                       "'48 Smallholders Party": ['ORG'],
                       "'A. Rosmini' - Robecchetto C/I": ['ORG'],
                       '1 - Antonio Salvetti Colle V.E.': ['ORG'],
                       '1 - I.C. Nocera Inferiore': ['ORG'],
                       '1 2 3 Stella': ['ORG'],
                       '1 C.D. Bovio Ruvo': ['ORG'],
                       '1 C.D.Gramsci - S.M. Pende': ['ORG'],
                       'Raymond and Beverly Sackler Institute for Biological,'
                       ' Physical and Engineering Sciences, Yale University': ['ORG'],
                       'Ąžuolas': ['ORG'],
                       'ΑΕΝ Dromolaxia': ['ORG'],
                       'Дом-музей Островского': ['ORG'],
                       'Jekabpils Agrobusiness College': ['ORG'],
                       'Ειδικό Γυμνάσιο - Ειδικό ΓΕΛ Αθηνών': ['ORG'],
                       'Партия Развития Украины': ['ORG'],
                       'Русскій BЂстникъ': ['ORG'],
                       '알로이시오중학교': ['ORG'],
                       'Художньо-меморіальний музей Олекси Новаківського': ['ORG'],
                       'โรงเรียนหาดใหญ่รัฐประชาสรรค์': ['ORG'],
                       '"Congarees" Site': ['LOC'],
                       '"El Salto" Tonatico': ['LOC'],
                       '"La Cetățui" archaeological site in Tismana': ['LOC'],
                       '"La Grande Cité"': ['LOC'],
                       "' Azeïlât Aḥmoûd": ['LOC'],
                       "'Achar Teïdoûm": ['LOC'],
                       "'Adade Yus Mountain": ['LOC'],
                       "'Adam Dihé": ['LOC'],
                       "'Aklé 'Âouâna": ['LOC'],
                       "'Aklé Aoukâr": ['LOC'],
                       "'Aklé el Liyene": ['LOC'],
                       "'Aleb Bel 'Aïd": ['LOC'],
                       'Sandøydrætten': ['LOC'],
                       'Sandøyfjorden': ['LOC'],
                       'Sandøygrunnen': ['LOC'],
                       'Ķūķu cliffs': ['LOC'],
                       'آبشار شوی': ['LOC'],
                       'تاهات أتاكور': ['LOC'],
                       'Голоух': ['LOC'],
                       'дерево спарапетов в Иджеване': ['LOC'],
                       'Опорски рид': ['LOC'],
                       'კუნძული ორპირი': ['LOC']}

    full_match_dict2 = {' National Archives and Library Agency': ['1'],
                        '!Bang!': ['3'],
                        '!Xunkhwesa Combined School': ['1', '3'],
                        '""Onderzoek - en Documentatiecentrum Beweging van Mensen met Laag Inkomen en Kinderen"': [
                            '3'],
                        '"Congarees" Site': ['2'],
                        '"El Salto" Tonatico': ['2'],
                        '"La Cetățui" archaeological site in Tismana': ['2'],
                        '"La Grande Cité"': ['2'],
                        '"Mons. Aurelio Sorrentino" Diocesan Museum': ['3'],
                        '"Octavian Goga" Cluj County Library': ['3'],
                        '"Pedro Arrupe" Political Training Institute': ['3'],
                        '"Red Terror" Martyrs\' Memorial Museum': ['3'],
                        '#Minas para Todos': ['3'],
                        '#Nome?': ['3'],
                        "' Azeïlât Aḥmoûd": ['2'],
                        "' Cogozzo - Don Eugenio Mazzi '": ['3'],
                        "' Villa Saviola '": ['3'],
                        "''A. Ilvento''-Grassano": ['3'],
                        "'48 Smallholders Party": ['3'],
                        "'A. Rosmini' - Robecchetto C/I": ['3'],
                        "'Achar Teïdoûm": ['2'],
                        "'Adade Yus Mountain": ['2'],
                        "'Adam Dihé": ['2'],
                        "'Aklé 'Âouâna": ['2'],
                        "'Aklé Aoukâr": ['2'],
                        "'Aklé el Liyene": ['2'],
                        "'Aleb Bel 'Aïd": ['2'],
                        "'t Gaverhopke Kerstbier": ['0'],
                        "'t Gaverhopke Kriek": ['0'],
                        "'t Vliegend Peert Centre for Old Arts": ['1'],
                        '+EU': ['1'],
                        ', Berhampur': ['1'],
                        '1 - Antonio Salvetti Colle V.E.': ['3'],
                        '1 - I.C. Nocera Inferiore': ['3'],
                        '1 2 3 Stella': ['3'],
                        '1 C.D. Bovio Ruvo': ['3'],
                        '1 C.D.Gramsci - S.M. Pende': ['3'],
                        '7α-hydroxydehydroepiandrosterone': ['0'],
                        '7α-methylestr-4-ene-3,17-dione': ['0'],
                        '7β-hydroxydehydroepiandrosterone': ['0'],
                        'America de Manta': ['1'],
                        'I': ['1'],
                        'Jekabpils Agrobusiness College': ['3'],
                        'Land Tenure Reform Association': ['3'],
                        'Raymond and Beverly Sackler Institute for Biological, Physical and Engineering Sciences, Yale University': [
                            '3'],
                        'Sandys': ['1'],
                        'Sandys Fort Spring': ['2', '3'],
                        'Sandøydrætten': ['2'],
                        'Sandøyfjorden': ['2'],
                        'Sandøygrunnen': ['2'],
                        'Ąžuolas': ['3'],
                        'Ķūķu cliffs': ['2'],
                        'ΑΕΝ Dromolaxia': ['3'],
                        'Ειδικό Γυμνάσιο - Ειδικό ΓΕΛ Αθηνών': ['3'],
                        'Голоух': ['2'],
                        'Дом-музей Островского': ['3'],
                        'Опорски рид': ['2'],
                        'Партия Развития Украины': ['3'],
                        'Русскій BЂстникъ': ['3'],
                        'Художньо-меморіальний музей Олекси Новаківського': ['3'],
                        'дерево спарапетов в Иджеване': ['2'],
                        'آبشار شوی': ['2'],
                        'تاهات أتاكور': ['2'],
                        'หนังสือพิมพ์ผู้จัดการ': ['1'],
                        'โรงเรียนหาดใหญ่รัฐประชาสรรค์': ['3'],
                        'კუნძული ორპირი': ['2'],
                        '알로이시오중학교': ['3']}

    full_match_dict3 = {'! Bang !': ['ORG'],
                        '! Xunkhwesa Combined School': ['ORG'],
                        '" Congarees " Site': ['LOC'],
                        '" El Salto " Tonatico': ['LOC'],
                        '" La Cetățui " archaeological site in Tismana': ['LOC'],
                        '" La Grande Cité "': ['LOC'],
                        '" Mons . Aurelio Sorrentino " Diocesan Museum': ['ORG'],
                        '" Octavian Goga " Cluj County Library': ['ORG'],
                        '" Pedro Arrupe " Political Training Institute': ['ORG'],
                        '" Red Terror " Martyrs\' Memorial Museum': ['ORG'],
                        '"" Onderzoek - en Documentatiecentrum Beweging van Mensen met Laag Inkomen en Kinderen "': [
                            'ORG'],
                        '# Minas para Todos': ['ORG'],
                        '# Nome ?': ['ORG'],
                        "' 48 Smallholders Party": ['ORG'],
                        "' A . Rosmini ' - Robecchetto C / I": ['ORG'],
                        "' Achar Teïdoûm": ['LOC'],
                        "' Adade Yus Mountain": ['LOC'],
                        "' Adam Dihé": ['LOC'],
                        "' Aklé ' Âouâna": ['LOC'],
                        "' Aklé Aoukâr": ['LOC'],
                        "' Aklé el Liyene": ['LOC'],
                        "' Aleb Bel ' Aïd": ['LOC'],
                        "' Azeïlât Aḥmoûd": ['LOC'],
                        "' Cogozzo - Don Eugenio Mazzi '": ['ORG'],
                        "' Villa Saviola '": ['ORG'],
                        "' t Vliegend Peert Centre for Old Arts": ['ORG'],
                        "'' A . Ilvento ''- Grassano": ['ORG'],
                        '+ EU': ['ORG'],
                        ', Berhampur': ['ORG'],
                        '1 - Antonio Salvetti Colle V.E .': ['ORG'],
                        '1 - I.C . Nocera Inferiore': ['ORG'],
                        '1 2 3 Stella': ['ORG'],
                        '1 C.D . Bovio Ruvo': ['ORG'],
                        '1 C.D.Gramsci - S.M . Pende': ['ORG'],
                        'America de Manta': ['ORG'],
                        'I': ['ORG'],
                        'Jekabpils Agrobusiness College': ['ORG'],
                        'Land Tenure Reform Association': ['ORG'],
                        'National Archives and Library Agency': ['ORG'],
                        'Raymond and Beverly Sackler Institute for Biological , Physical and Engineering Sciences , Yale University': [
                            'ORG'],
                        'Sandys': ['ORG'],
                        'Sandys Fort Spring': ['ORG', 'LOC'],
                        'Sandøydrætten': ['LOC'],
                        'Sandøyfjorden': ['LOC'],
                        'Sandøygrunnen': ['LOC'],
                        'Ąžuolas': ['ORG'],
                        'Ķūķu cliffs': ['LOC'],
                        'ΑΕΝ Dromolaxia': ['ORG'],
                        'Ειδικό Γυμνάσιο - Ειδικό ΓΕΛ Αθηνών': ['ORG'],
                        'Голоух': ['LOC'],
                        'Дом-музей Островского': ['ORG'],
                        'Опорски рид': ['LOC'],
                        'Партия Развития Украины': ['ORG'],
                        'Русскій BЂстникъ': ['ORG'],
                        'Художньо-меморіальний музей Олекси Новаківського': ['ORG'],
                        'дерево спарапетов в Иджеване': ['LOC'],
                        'آبشار شوی': ['LOC'],
                        'تاهات أتاكور': ['LOC'],
                        'หนังสือพิมพ์ผู้จัดการ': ['ORG'],
                        'โรงเรียนหาดใหญ่รัฐประชาสรรค์': ['ORG'],
                        'კუნძული ორპირი': ['LOC'],
                        '알로이시오중학교': ['ORG']}

    full_match_dict4 = {'! Bang !': ['3'],
                        '! Xunkhwesa Combined School': ['1', '3'],
                        '" Congarees " Site': ['2'],
                        '" El Salto " Tonatico': ['2'],
                        '" La Cetățui " archaeological site in Tismana': ['2'],
                        '" La Grande Cité "': ['2'],
                        '" Mons . Aurelio Sorrentino " Diocesan Museum': ['3'],
                        '" Octavian Goga " Cluj County Library': ['3'],
                        '" Pedro Arrupe " Political Training Institute': ['3'],
                        '" Red Terror " Martyrs\' Memorial Museum': ['3'],
                        '"" Onderzoek - en Documentatiecentrum Beweging van Mensen met Laag Inkomen en Kinderen "': [
                            '3'],
                        '# Minas para Todos': ['3'],
                        '# Nome ?': ['3'],
                        "' 48 Smallholders Party": ['3'],
                        "' A . Rosmini ' - Robecchetto C / I": ['3'],
                        "' Achar Teïdoûm": ['2'],
                        "' Adade Yus Mountain": ['2'],
                        "' Adam Dihé": ['2'],
                        "' Aklé ' Âouâna": ['2'],
                        "' Aklé Aoukâr": ['2'],
                        "' Aklé el Liyene": ['2'],
                        "' Aleb Bel ' Aïd": ['2'],
                        "' Azeïlât Aḥmoûd": ['2'],
                        "' Cogozzo - Don Eugenio Mazzi '": ['3'],
                        "' Villa Saviola '": ['3'],
                        "' t Gaverhopke Kerstbier": ['0'],
                        "' t Gaverhopke Kriek": ['0'],
                        "' t Vliegend Peert Centre for Old Arts": ['1'],
                        "'' A . Ilvento ''- Grassano": ['3'],
                        '+ EU': ['1'],
                        ', Berhampur': ['1'],
                        '1 - Antonio Salvetti Colle V.E .': ['3'],
                        '1 - I.C . Nocera Inferiore': ['3'],
                        '1 2 3 Stella': ['3'],
                        '1 C.D . Bovio Ruvo': ['3'],
                        '1 C.D.Gramsci - S.M . Pende': ['3'],
                        '7α-hydroxydehydroepiandrosterone': ['0'],
                        '7α-methylestr-4-ene-3,17-dione': ['0'],
                        '7β-hydroxydehydroepiandrosterone': ['0'],
                        'America de Manta': ['1'],
                        'I': ['1'],
                        'Jekabpils Agrobusiness College': ['3'],
                        'Land Tenure Reform Association': ['3'],
                        'National Archives and Library Agency': ['1'],
                        'Raymond and Beverly Sackler Institute for Biological , Physical and Engineering Sciences , Yale University': [
                            '3'],
                        'Sandys': ['1'],
                        'Sandys Fort Spring': ['2', '3'],
                        'Sandøydrætten': ['2'],
                        'Sandøyfjorden': ['2'],
                        'Sandøygrunnen': ['2'],
                        'Ąžuolas': ['3'],
                        'Ķūķu cliffs': ['2'],
                        'ΑΕΝ Dromolaxia': ['3'],
                        'Ειδικό Γυμνάσιο - Ειδικό ΓΕΛ Αθηνών': ['3'],
                        'Голоух': ['2'],
                        'Дом-музей Островского': ['3'],
                        'Опорски рид': ['2'],
                        'Партия Развития Украины': ['3'],
                        'Русскій BЂстникъ': ['3'],
                        'Художньо-меморіальний музей Олекси Новаківського': ['3'],
                        'дерево спарапетов в Иджеване': ['2'],
                        'آبشار شوی': ['2'],
                        'تاهات أتاكور': ['2'],
                        'หนังสือพิมพ์ผู้จัดการ': ['1'],
                        'โรงเรียนหาดใหญ่รัฐประชาสรรค์': ['3'],
                        'კუნძული ორპირი': ['2'],
                        '알로이시오중학교': ['3']}

    partial_match_dict = {'!Bang!': ['S-ORG'],
                          '!Xunkhwesa': ['B-ORG'],
                          '""Onderzoek': ['B-ORG'],
                          '"Congarees"': ['B-LOC'],
                          '"El': ['B-LOC'],
                          '"La': ['B-LOC'],
                          '"Mons.': ['B-ORG'],
                          '"Octavian': ['B-ORG'],
                          '"Pedro': ['B-ORG'],
                          '"Red': ['B-ORG'],
                          '#Minas': ['B-ORG'],
                          '#Nome?': ['S-ORG'],
                          "''A.": ['B-ORG'],
                          "'48": ['B-ORG'],
                          "'A.": ['B-ORG'],
                          "'Achar": ['B-LOC'],
                          "'Adade": ['B-LOC'],
                          "'Adam": ['B-LOC'],
                          "'Aklé": ['B-LOC'],
                          "'Aleb": ['B-LOC'],
                          "'Aïd": ['E-LOC'],
                          "'t": ['B-ORG'],
                          "'Âouâna": ['E-LOC'],
                          '+EU': ['S-ORG'],
                          'Agency': ['E-ORG'],
                          'Agrobusiness': ['I-ORG'],
                          'America': ['B-ORG'],
                          'Antonio': ['B-ORG'],
                          'Aoukâr': ['E-LOC'],
                          'Archives': ['I-ORG'],
                          'Arrupe"': ['I-ORG'],
                          'Arts': ['E-ORG'],
                          'Association': ['E-ORG'],
                          'Aurelio': ['I-ORG'],
                          'Azeïlât': ['B-LOC'],
                          'Aḥmoûd': ['E-LOC'],
                          'Bel': ['I-LOC'],
                          'Berhampur': ['S-ORG'],
                          'Beverly': ['I-ORG'],
                          'Beweging': ['I-ORG'],
                          'Biological,': ['I-ORG'],
                          'Bovio': ['I-ORG'],
                          'BЂстникъ': ['E-ORG'],
                          'C.D.': ['B-ORG'],
                          'C.D.Gramsci': ['B-ORG'],
                          'C/I': ['E-ORG'],
                          'Centre': ['I-ORG'],
                          'Cetățui"': ['I-LOC'],
                          'Cité"': ['E-LOC'],
                          'Cluj': ['I-ORG'],
                          'Cogozzo': ['B-ORG'],
                          'Colle': ['I-ORG'],
                          'College': ['E-ORG'],
                          'Combined': ['I-ORG'],
                          'County': ['I-ORG'],
                          'Dihé': ['E-LOC'],
                          'Diocesan': ['I-ORG'],
                          'Documentatiecentrum': ['I-ORG'],
                          'Don': ['I-ORG'],
                          'Dromolaxia': ['E-ORG'],
                          'Engineering': ['I-ORG'],
                          'Eugenio': ['I-ORG'],
                          'Fort': ['I-ORG', 'I-LOC'],
                          'Goga"': ['I-ORG'],
                          'Grande': ['I-LOC'],
                          'I.C.': ['B-ORG'],
                          "Ilvento''-Grassano": ['E-ORG'],
                          'Inferiore': ['E-ORG'],
                          'Inkomen': ['I-ORG'],
                          'Institute': ['E-ORG', 'I-ORG'],
                          'Jekabpils': ['B-ORG'],
                          'Kinderen"': ['E-ORG'],
                          'Laag': ['I-ORG'],
                          'Land': ['B-ORG'],
                          'Library': ['I-ORG', 'E-ORG'],
                          'Liyene': ['E-LOC'],
                          'Manta': ['E-ORG'],
                          "Martyrs'": ['I-ORG'],
                          'Mazzi': ['E-ORG'],
                          'Memorial': ['I-ORG'],
                          'Mensen': ['I-ORG'],
                          'Mountain': ['E-LOC'],
                          'Museum': ['E-ORG'],
                          'National': ['B-ORG'],
                          'Nocera': ['I-ORG'],
                          'Old': ['I-ORG'],
                          'Party': ['E-ORG'],
                          'Peert': ['I-ORG'],
                          'Pende': ['E-ORG'],
                          'Physical': ['I-ORG'],
                          'Political': ['I-ORG'],
                          'Raymond': ['B-ORG'],
                          'Reform': ['I-ORG'],
                          'Robecchetto': ['I-ORG'],
                          "Rosmini'": ['I-ORG'],
                          'Ruvo': ['E-ORG'],
                          'S.M.': ['I-ORG'],
                          'Sackler': ['I-ORG'],
                          'Salto"': ['I-LOC'],
                          'Salvetti': ['I-ORG'],
                          'Sandys': ['S-ORG', 'B-ORG', 'B-LOC'],
                          'I': ['S-ORG'],
                          'Sandøydrætten': ['S-LOC'],
                          'Sandøyfjorden': ['S-LOC'],
                          'Sandøygrunnen': ['S-LOC'],
                          'Saviola': ['E-ORG'],
                          'School': ['E-ORG'],
                          'Sciences,': ['I-ORG'],
                          'Site': ['E-LOC'],
                          'Smallholders': ['I-ORG'],
                          'Sorrentino"': ['I-ORG'],
                          'Spring': ['E-ORG', 'E-LOC'],
                          'Stella': ['S-ORG'],
                          'Tenure': ['I-ORG'],
                          'Terror"': ['I-ORG'],
                          'Teïdoûm': ['E-LOC'],
                          'Tismana': ['E-LOC'],
                          'Todos': ['E-ORG'],
                          'Tonatico': ['E-LOC'],
                          'Training': ['I-ORG'],
                          'University': ['E-ORG'],
                          'V.E.': ['E-ORG'],
                          'Villa': ['B-ORG'],
                          'Vliegend': ['I-ORG'],
                          'Yale': ['I-ORG'],
                          'Yus': ['I-LOC'],
                          'and': ['I-ORG'],
                          'archaeological': ['I-LOC'],
                          'cliffs': ['E-LOC'],
                          'de': ['I-ORG'],
                          'el': ['I-LOC'],
                          'en': ['I-ORG'],
                          'for': ['I-ORG'],
                          'in': ['I-LOC'],
                          'met': ['I-ORG'],
                          'para': ['I-ORG'],
                          'site': ['I-LOC'],
                          'van': ['I-ORG'],
                          'Ąžuolas': ['S-ORG'],
                          'Ķūķu': ['B-LOC'],
                          'ΑΕΝ': ['B-ORG'],
                          'Αθηνών': ['E-ORG'],
                          'ΓΕΛ': ['I-ORG'],
                          'Γυμνάσιο': ['I-ORG'],
                          'Ειδικό': ['B-ORG'],
                          'Голоух': ['S-LOC'],
                          'Дом-музей': ['B-ORG'],
                          'Иджеване': ['E-LOC'],
                          'Новаківського': ['E-ORG'],
                          'Олекси': ['I-ORG'],
                          'Опорски': ['B-LOC'],
                          'Островского': ['E-ORG'],
                          'Партия': ['B-ORG'],
                          'Развития': ['I-ORG'],
                          'Русскій': ['B-ORG'],
                          'Украины': ['E-ORG'],
                          'Художньо-меморіальний': ['B-ORG'],
                          'дерево': ['B-LOC'],
                          'музей': ['I-ORG'],
                          'рид': ['E-LOC'],
                          'спарапетов': ['I-LOC'],
                          'آبشار': ['B-LOC'],
                          'أتاكور': ['E-LOC'],
                          'تاهات': ['B-LOC'],
                          'شوی': ['E-LOC'],
                          'หนังสือพิมพ์ผู้จัดการ': ['S-ORG'],
                          'โรงเรียนหาดใหญ่รัฐประชาสรรค์': ['S-ORG'],
                          'კუნძული': ['B-LOC'],
                          'ორპირი': ['E-LOC'],
                          '알로이시오중학교': ['S-ORG']}

    partial_match_dict2 = {'!Bang!': ['S-3'],
                           '!Xunkhwesa': ['B-1', 'B-3'],
                           '""Onderzoek': ['B-3'],
                           '"Congarees"': ['B-2'],
                           '"El': ['B-2'],
                           '"La': ['B-2'],
                           '"Mons.': ['B-3'],
                           '"Octavian': ['B-3'],
                           '"Pedro': ['B-3'],
                           '"Red': ['B-3'],
                           '#Minas': ['B-3'],
                           '#Nome?': ['S-3'],
                           "''A.": ['B-3'],
                           "'48": ['B-3'],
                           "'A.": ['B-3'],
                           "'Achar": ['B-2'],
                           "'Adade": ['B-2'],
                           "'Adam": ['B-2'],
                           "'Aklé": ['B-2'],
                           "'Aleb": ['B-2'],
                           "'Aïd": ['E-2'],
                           "'t": ['B-0', 'B-1'],
                           "'Âouâna": ['E-2'],
                           '+EU': ['S-1'],
                           '7α-hydroxydehydroepiandrosterone': ['S-0'],
                           '7α-methylestr-4-ene-3,17-dione': ['S-0'],
                           '7β-hydroxydehydroepiandrosterone': ['S-0'],
                           'Agency': ['E-1'],
                           'Agrobusiness': ['I-3'],
                           'America': ['B-1'],
                           'Antonio': ['B-3'],
                           'Aoukâr': ['E-2'],
                           'Archives': ['I-1'],
                           'Arrupe"': ['I-3'],
                           'Arts': ['E-1'],
                           'Association': ['E-3'],
                           'Aurelio': ['I-3'],
                           'Azeïlât': ['B-2'],
                           'Aḥmoûd': ['E-2'],
                           'Bel': ['I-2'],
                           'Berhampur': ['S-1'],
                           'Beverly': ['I-3'],
                           'Beweging': ['I-3'],
                           'Biological,': ['I-3'],
                           'Bovio': ['I-3'],
                           'BЂстникъ': ['E-3'],
                           'C.D.': ['B-3'],
                           'C.D.Gramsci': ['B-3'],
                           'C/I': ['E-3'],
                           'Centre': ['I-1'],
                           'Cetățui"': ['I-2'],
                           'Cité"': ['E-2'],
                           'Cluj': ['I-3'],
                           'Cogozzo': ['B-3'],
                           'Colle': ['I-3'],
                           'College': ['E-3'],
                           'Combined': ['I-1', 'I-3'],
                           'County': ['I-3'],
                           'Dihé': ['E-2'],
                           'Diocesan': ['I-3'],
                           'Documentatiecentrum': ['I-3'],
                           'Don': ['I-3'],
                           'Dromolaxia': ['E-3'],
                           'Engineering': ['I-3'],
                           'Eugenio': ['I-3'],
                           'Fort': ['I-2', 'I-3'],
                           'Gaverhopke': ['I-0'],
                           'Goga"': ['I-3'],
                           'Grande': ['I-2'],
                           'I': ['S-1'],
                           'I.C.': ['B-3'],
                           "Ilvento''-Grassano": ['E-3'],
                           'Inferiore': ['E-3'],
                           'Inkomen': ['I-3'],
                           'Institute': ['E-3', 'I-3'],
                           'Jekabpils': ['B-3'],
                           'Kerstbier': ['E-0'],
                           'Kinderen"': ['E-3'],
                           'Kriek': ['E-0'],
                           'Laag': ['I-3'],
                           'Land': ['B-3'],
                           'Library': ['I-1', 'E-3'],
                           'Liyene': ['E-2'],
                           'Manta': ['E-1'],
                           "Martyrs'": ['I-3'],
                           'Mazzi': ['E-3'],
                           'Memorial': ['I-3'],
                           'Mensen': ['I-3'],
                           'Mountain': ['E-2'],
                           'Museum': ['E-3'],
                           'National': ['B-1'],
                           'Nocera': ['I-3'],
                           'Old': ['I-1'],
                           'Party': ['E-3'],
                           'Peert': ['I-1'],
                           'Pende': ['E-3'],
                           'Physical': ['I-3'],
                           'Political': ['I-3'],
                           'Raymond': ['B-3'],
                           'Reform': ['I-3'],
                           'Robecchetto': ['I-3'],
                           "Rosmini'": ['I-3'],
                           'Ruvo': ['E-3'],
                           'S.M.': ['I-3'],
                           'Sackler': ['I-3'],
                           'Salto"': ['I-2'],
                           'Salvetti': ['I-3'],
                           'Sandys': ['S-1', 'B-2', 'B-3'],
                           'Sandøydrætten': ['S-2'],
                           'Sandøyfjorden': ['S-2'],
                           'Sandøygrunnen': ['S-2'],
                           'Saviola': ['E-3'],
                           'School': ['E-1', 'E-3'],
                           'Sciences,': ['I-3'],
                           'Site': ['E-2'],
                           'Smallholders': ['I-3'],
                           'Sorrentino"': ['I-3'],
                           'Spring': ['E-2', 'E-3'],
                           'Stella': ['S-3'],
                           'Tenure': ['I-3'],
                           'Terror"': ['I-3'],
                           'Teïdoûm': ['E-2'],
                           'Tismana': ['E-2'],
                           'Todos': ['E-3'],
                           'Tonatico': ['E-2'],
                           'Training': ['I-3'],
                           'University': ['E-3'],
                           'V.E.': ['E-3'],
                           'Villa': ['B-3'],
                           'Vliegend': ['I-1'],
                           'Yale': ['I-3'],
                           'Yus': ['I-2'],
                           'and': ['I-1', 'I-3'],
                           'archaeological': ['I-2'],
                           'cliffs': ['E-2'],
                           'de': ['I-1'],
                           'el': ['I-2'],
                           'en': ['I-3'],
                           'for': ['I-1', 'I-3'],
                           'in': ['I-2'],
                           'met': ['I-3'],
                           'para': ['I-3'],
                           'site': ['I-2'],
                           'van': ['I-3'],
                           'Ąžuolas': ['S-3'],
                           'Ķūķu': ['B-2'],
                           'ΑΕΝ': ['B-3'],
                           'Αθηνών': ['E-3'],
                           'ΓΕΛ': ['I-3'],
                           'Γυμνάσιο': ['I-3'],
                           'Ειδικό': ['B-3'],
                           'Голоух': ['S-2'],
                           'Дом-музей': ['B-3'],
                           'Иджеване': ['E-2'],
                           'Новаківського': ['E-3'],
                           'Олекси': ['I-3'],
                           'Опорски': ['B-2'],
                           'Островского': ['E-3'],
                           'Партия': ['B-3'],
                           'Развития': ['I-3'],
                           'Русскій': ['B-3'],
                           'Украины': ['E-3'],
                           'Художньо-меморіальний': ['B-3'],
                           'дерево': ['B-2'],
                           'музей': ['I-3'],
                           'рид': ['E-2'],
                           'спарапетов': ['I-2'],
                           'آبشار': ['B-2'],
                           'أتاكور': ['E-2'],
                           'تاهات': ['B-2'],
                           'شوی': ['E-2'],
                           'หนังสือพิมพ์ผู้จัดการ': ['S-1'],
                           'โรงเรียนหาดใหญ่รัฐประชาสรรค์': ['S-3'],
                           'კუნძული': ['B-2'],
                           'ორპირი': ['E-2'],
                           '알로이시오중학교': ['S-3']}

    partial_match_dict3 = {'48': ['B-ORG'],
                           'Achar': ['B-LOC'],
                           'Adade': ['B-LOC'],
                           'Adam': ['B-LOC'],
                           'Agency': ['E-ORG'],
                           'Agrobusiness': ['I-ORG'],
                           'Aklé': ['B-LOC'],
                           'Aleb': ['B-LOC'],
                           'America': ['B-ORG'],
                           'Antonio': ['B-ORG'],
                           'Aoukâr': ['E-LOC'],
                           'Archives': ['I-ORG'],
                           'Arrupe': ['I-ORG'],
                           'Arts': ['E-ORG'],
                           'Association': ['E-ORG'],
                           'Aurelio': ['I-ORG'],
                           'Azeïlât': ['B-LOC'],
                           'Aïd': ['E-LOC'],
                           'Aḥmoûd': ['E-LOC'],
                           'Bang': ['S-ORG'],
                           'Bel': ['I-LOC'],
                           'Berhampur': ['S-ORG'],
                           'Beverly': ['I-ORG'],
                           'Beweging': ['I-ORG'],
                           'Biological': ['I-ORG'],
                           'Bovio': ['I-ORG'],
                           'BЂстникъ': ['E-ORG'],
                           'C.D': ['B-ORG'],
                           'C.D.Gramsci': ['B-ORG'],
                           'Centre': ['I-ORG'],
                           'Cetățui': ['I-LOC'],
                           'Cité': ['E-LOC'],
                           'Cluj': ['I-ORG'],
                           'Cogozzo': ['B-ORG'],
                           'Colle': ['I-ORG'],
                           'College': ['E-ORG'],
                           'Combined': ['I-ORG'],
                           'Congarees': ['B-LOC'],
                           'County': ['I-ORG'],
                           'Dihé': ['E-LOC'],
                           'Diocesan': ['I-ORG'],
                           'Documentatiecentrum': ['I-ORG'],
                           'Don': ['I-ORG'],
                           'Dromolaxia': ['E-ORG'],
                           'EU': ['S-ORG'],
                           'El': ['B-LOC'],
                           'Engineering': ['I-ORG'],
                           'Eugenio': ['I-ORG'],
                           'Fort': ['I-ORG', 'I-LOC'],
                           'Goga': ['I-ORG'],
                           'Grande': ['I-LOC'],
                           'Grassano': ['E-ORG'],
                           'I': ['S-ORG'],
                           'I.C': ['B-ORG'],
                           'Ilvento': ['B-ORG'],
                           'Inferiore': ['E-ORG'],
                           'Inkomen': ['I-ORG'],
                           'Institute': ['E-ORG', 'I-ORG'],
                           'Jekabpils': ['B-ORG'],
                           'Kinderen': ['E-ORG'],
                           'La': ['B-LOC'],
                           'Laag': ['I-ORG'],
                           'Land': ['B-ORG'],
                           'Library': ['I-ORG', 'E-ORG'],
                           'Liyene': ['E-LOC'],
                           'Manta': ['E-ORG'],
                           "Martyrs'": ['I-ORG'],
                           'Mazzi': ['E-ORG'],
                           'Memorial': ['I-ORG'],
                           'Mensen': ['I-ORG'],
                           'Minas': ['B-ORG'],
                           'Mons': ['B-ORG'],
                           'Mountain': ['E-LOC'],
                           'Museum': ['E-ORG'],
                           'National': ['B-ORG'],
                           'Nocera': ['I-ORG'],
                           'Nome': ['S-ORG'],
                           'Octavian': ['B-ORG'],
                           'Old': ['I-ORG'],
                           'Onderzoek': ['B-ORG'],
                           'Party': ['E-ORG'],
                           'Pedro': ['B-ORG'],
                           'Peert': ['I-ORG'],
                           'Pende': ['E-ORG'],
                           'Physical': ['I-ORG'],
                           'Political': ['I-ORG'],
                           'Raymond': ['B-ORG'],
                           'Red': ['B-ORG'],
                           'Reform': ['I-ORG'],
                           'Robecchetto': ['E-ORG'],
                           'Rosmini': ['B-ORG'],
                           'Ruvo': ['E-ORG'],
                           'S.M': ['I-ORG'],
                           'Sackler': ['I-ORG'],
                           'Salto': ['I-LOC'],
                           'Salvetti': ['I-ORG'],
                           'Sandys': ['S-ORG', 'B-ORG', 'B-LOC'],
                           'Sandøydrætten': ['S-LOC'],
                           'Sandøyfjorden': ['S-LOC'],
                           'Sandøygrunnen': ['S-LOC'],
                           'Saviola': ['E-ORG'],
                           'School': ['E-ORG'],
                           'Sciences': ['I-ORG'],
                           'Site': ['E-LOC'],
                           'Smallholders': ['I-ORG'],
                           'Sorrentino': ['I-ORG'],
                           'Spring': ['E-ORG', 'E-LOC'],
                           'Stella': ['S-ORG'],
                           'Tenure': ['I-ORG'],
                           'Terror': ['I-ORG'],
                           'Teïdoûm': ['E-LOC'],
                           'Tismana': ['E-LOC'],
                           'Todos': ['E-ORG'],
                           'Tonatico': ['E-LOC'],
                           'Training': ['I-ORG'],
                           'University': ['E-ORG'],
                           'V.E': ['E-ORG'],
                           'Villa': ['B-ORG'],
                           'Vliegend': ['B-ORG'],
                           'Xunkhwesa': ['B-ORG'],
                           'Yale': ['I-ORG'],
                           'Yus': ['I-LOC'],
                           'and': ['I-ORG'],
                           'archaeological': ['I-LOC'],
                           'cliffs': ['E-LOC'],
                           'de': ['I-ORG'],
                           'el': ['I-LOC'],
                           'en': ['I-ORG'],
                           'for': ['I-ORG'],
                           'in': ['I-LOC'],
                           'met': ['I-ORG'],
                           'para': ['I-ORG'],
                           'site': ['I-LOC'],
                           'van': ['I-ORG'],
                           'Âouâna': ['E-LOC'],
                           'Ąžuolas': ['S-ORG'],
                           'Ķūķu': ['B-LOC'],
                           'ΑΕΝ': ['B-ORG'],
                           'Αθηνών': ['E-ORG'],
                           'ΓΕΛ': ['I-ORG'],
                           'Γυμνάσιο': ['I-ORG'],
                           'Ειδικό': ['B-ORG'],
                           'Голоух': ['S-LOC'],
                           'Дом-музей': ['B-ORG'],
                           'Иджеване': ['E-LOC'],
                           'Новаківського': ['E-ORG'],
                           'Олекси': ['I-ORG'],
                           'Опорски': ['B-LOC'],
                           'Островского': ['E-ORG'],
                           'Партия': ['B-ORG'],
                           'Развития': ['I-ORG'],
                           'Русскій': ['B-ORG'],
                           'Украины': ['E-ORG'],
                           'Художньо-меморіальний': ['B-ORG'],
                           'дерево': ['B-LOC'],
                           'музей': ['I-ORG'],
                           'рид': ['E-LOC'],
                           'спарапетов': ['I-LOC'],
                           'آبشار': ['B-LOC'],
                           'أتاكور': ['E-LOC'],
                           'تاهات': ['B-LOC'],
                           'شوی': ['E-LOC'],
                           'หนังสือพิมพ์ผู้จัดการ': ['S-ORG'],
                           'โรงเรียนหาดใหญ่รัฐประชาสรรค์': ['S-ORG'],
                           'კუნძული': ['B-LOC'],
                           'ორპირი': ['E-LOC'],
                           '알로이시오중학교': ['S-ORG']}

    partial_match_dict4 = {'48': ['B-3'],
                           '7α-hydroxydehydroepiandrosterone': ['S-0'],
                           '7α-methylestr-4-ene-3,17-dione': ['S-0'],
                           '7β-hydroxydehydroepiandrosterone': ['S-0'],
                           'Achar': ['B-2'],
                           'Adade': ['B-2'],
                           'Adam': ['B-2'],
                           'Agency': ['E-1'],
                           'Agrobusiness': ['I-3'],
                           'Aklé': ['B-2'],
                           'Aleb': ['B-2'],
                           'America': ['B-1'],
                           'Antonio': ['B-3'],
                           'Aoukâr': ['E-2'],
                           'Archives': ['I-1'],
                           'Arrupe': ['I-3'],
                           'Arts': ['E-1'],
                           'Association': ['E-3'],
                           'Aurelio': ['I-3'],
                           'Azeïlât': ['B-2'],
                           'Aïd': ['E-2'],
                           'Aḥmoûd': ['E-2'],
                           'Bang': ['S-3'],
                           'Bel': ['I-2'],
                           'Berhampur': ['S-1'],
                           'Beverly': ['I-3'],
                           'Beweging': ['I-3'],
                           'Biological': ['I-3'],
                           'Bovio': ['I-3'],
                           'BЂстникъ': ['E-3'],
                           'C.D': ['B-3'],
                           'C.D.Gramsci': ['B-3'],
                           'Centre': ['I-1'],
                           'Cetățui': ['I-2'],
                           'Cité': ['E-2'],
                           'Cluj': ['I-3'],
                           'Cogozzo': ['B-3'],
                           'Colle': ['I-3'],
                           'College': ['E-3'],
                           'Combined': ['I-1', 'I-3'],
                           'Congarees': ['B-2'],
                           'County': ['I-3'],
                           'Dihé': ['E-2'],
                           'Diocesan': ['I-3'],
                           'Documentatiecentrum': ['I-3'],
                           'Don': ['I-3'],
                           'Dromolaxia': ['E-3'],
                           'EU': ['S-1'],
                           'El': ['B-2'],
                           'Engineering': ['I-3'],
                           'Eugenio': ['I-3'],
                           'Fort': ['I-2', 'I-3'],
                           'Gaverhopke': ['B-0'],
                           'Goga': ['I-3'],
                           'Grande': ['I-2'],
                           'Grassano': ['E-3'],
                           'I': ['S-1'],
                           'I.C': ['B-3'],
                           'Ilvento': ['B-3'],
                           'Inferiore': ['E-3'],
                           'Inkomen': ['I-3'],
                           'Institute': ['E-3', 'I-3'],
                           'Jekabpils': ['B-3'],
                           'Kerstbier': ['E-0'],
                           'Kinderen': ['E-3'],
                           'Kriek': ['E-0'],
                           'La': ['B-2'],
                           'Laag': ['I-3'],
                           'Land': ['B-3'],
                           'Library': ['I-1', 'E-3'],
                           'Liyene': ['E-2'],
                           'Manta': ['E-1'],
                           "Martyrs'": ['I-3'],
                           'Mazzi': ['E-3'],
                           'Memorial': ['I-3'],
                           'Mensen': ['I-3'],
                           'Minas': ['B-3'],
                           'Mons': ['B-3'],
                           'Mountain': ['E-2'],
                           'Museum': ['E-3'],
                           'National': ['B-1'],
                           'Nocera': ['I-3'],
                           'Nome': ['S-3'],
                           'Octavian': ['B-3'],
                           'Old': ['I-1'],
                           'Onderzoek': ['B-3'],
                           'Party': ['E-3'],
                           'Pedro': ['B-3'],
                           'Peert': ['I-1'],
                           'Pende': ['E-3'],
                           'Physical': ['I-3'],
                           'Political': ['I-3'],
                           'Raymond': ['B-3'],
                           'Red': ['B-3'],
                           'Reform': ['I-3'],
                           'Robecchetto': ['E-3'],
                           'Rosmini': ['B-3'],
                           'Ruvo': ['E-3'],
                           'S.M': ['I-3'],
                           'Sackler': ['I-3'],
                           'Salto': ['I-2'],
                           'Salvetti': ['I-3'],
                           'Sandys': ['S-1', 'B-2', 'B-3'],
                           'Sandøydrætten': ['S-2'],
                           'Sandøyfjorden': ['S-2'],
                           'Sandøygrunnen': ['S-2'],
                           'Saviola': ['E-3'],
                           'School': ['E-1', 'E-3'],
                           'Sciences': ['I-3'],
                           'Site': ['E-2'],
                           'Smallholders': ['I-3'],
                           'Sorrentino': ['I-3'],
                           'Spring': ['E-2', 'E-3'],
                           'Stella': ['S-3'],
                           'Tenure': ['I-3'],
                           'Terror': ['I-3'],
                           'Teïdoûm': ['E-2'],
                           'Tismana': ['E-2'],
                           'Todos': ['E-3'],
                           'Tonatico': ['E-2'],
                           'Training': ['I-3'],
                           'University': ['E-3'],
                           'V.E': ['E-3'],
                           'Villa': ['B-3'],
                           'Vliegend': ['B-1'],
                           'Xunkhwesa': ['B-1', 'B-3'],
                           'Yale': ['I-3'],
                           'Yus': ['I-2'],
                           'and': ['I-1', 'I-3'],
                           'archaeological': ['I-2'],
                           'cliffs': ['E-2'],
                           'de': ['I-1'],
                           'el': ['I-2'],
                           'en': ['I-3'],
                           'for': ['I-1', 'I-3'],
                           'in': ['I-2'],
                           'met': ['I-3'],
                           'para': ['I-3'],
                           'site': ['I-2'],
                           'van': ['I-3'],
                           'Âouâna': ['E-2'],
                           'Ąžuolas': ['S-3'],
                           'Ķūķu': ['B-2'],
                           'ΑΕΝ': ['B-3'],
                           'Αθηνών': ['E-3'],
                           'ΓΕΛ': ['I-3'],
                           'Γυμνάσιο': ['I-3'],
                           'Ειδικό': ['B-3'],
                           'Голоух': ['S-2'],
                           'Дом-музей': ['B-3'],
                           'Иджеване': ['E-2'],
                           'Новаківського': ['E-3'],
                           'Олекси': ['I-3'],
                           'Опорски': ['B-2'],
                           'Островского': ['E-3'],
                           'Партия': ['B-3'],
                           'Развития': ['I-3'],
                           'Русскій': ['B-3'],
                           'Украины': ['E-3'],
                           'Художньо-меморіальний': ['B-3'],
                           'дерево': ['B-2'],
                           'музей': ['I-3'],
                           'рид': ['E-2'],
                           'спарапетов': ['I-2'],
                           'آبشار': ['B-2'],
                           'أتاكور': ['E-2'],
                           'تاهات': ['B-2'],
                           'شوی': ['E-2'],
                           'หนังสือพิมพ์ผู้จัดการ': ['S-1'],
                           'โรงเรียนหาดใหญ่รัฐประชาสรรค์': ['S-3'],
                           'კუნძული': ['B-2'],
                           'ორპირი': ['E-2'],
                           '알로이시오중학교': ['S-3']}

    def setUp(self) -> None:
        self.maxDiff = None
        pass

    def test_matching_methods_good1(self):
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_matching=True,
                                                                           partial_matching=False)
            self.assertEqual(gazetteer_embedding.matching_methods, ['full_match'])

    def test_matching_methods_good2(self):
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_matching=False,
                                                                           partial_matching=True)
            self.assertEqual(gazetteer_embedding.matching_methods, ['partial_match'])

    def test_set_feature_list_good1(self):
        label_dict = MagicMock()
        label_dict.get_items.return_value = ['PER', 'ORG', 'LOC', 'MISC']
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict,
                                                                           full_matching=True,
                                                                           partial_matching=True)
            self.assertEqual(gazetteer_embedding.feature_list, ['O', 'S-PER', 'B-PER', 'E-PER', 'I-PER', 'S-ORG',
                                                                'B-ORG', 'E-ORG', 'I-ORG', 'S-LOC', 'B-LOC', 'E-LOC',
                                                                'I-LOC', 'S-MISC', 'B-MISC', 'E-MISC', 'I-MISC',
                                                                'PER', 'ORG', 'LOC', 'MISC'])
            self.assertEqual(gazetteer_embedding.embedding_length, 21)

    def test_set_feature_list_good2(self):
        label_dict = MagicMock()
        label_dict.get_items.return_value = ['PER', 'ORG', 'LOC', 'MISC']
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict,
                                                                           full_matching=False,
                                                                           partial_matching=True)
            self.assertEqual(gazetteer_embedding.feature_list, ['O', 'S-PER', 'B-PER', 'E-PER', 'I-PER', 'S-ORG',
                                                                'B-ORG', 'E-ORG', 'I-ORG', 'S-LOC', 'B-LOC', 'E-LOC',
                                                                'I-LOC', 'S-MISC', 'B-MISC', 'E-MISC', 'I-MISC'])
            self.assertEqual(gazetteer_embedding.embedding_length, 17)

    def test_set_feature_list_good3(self):
        label_dict = MagicMock()
        label_dict.get_items.return_value = ['PER', 'ORG', 'LOC', 'MISC']
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict,
                                                                           full_matching=True,
                                                                           partial_matching=False)
            self.assertEqual(gazetteer_embedding.feature_list, ['O', 'PER', 'ORG', 'LOC', 'MISC'])
            self.assertEqual(gazetteer_embedding.embedding_length, 5)

    def test_set_feature_list_good4(self):
        label_dict = MagicMock()
        label_dict.get_items.return_value = ['PER', 'ORG', 'LOC', 'MISC']
        gazetteers = [{'0': ['eng-CHEM-name-test.txt']},
                      {'1': ['eng-ORG-alias-test.txt']},
                      {'2': ['eng-LOC-name-test.txt']},
                      {'3': ['eng-ORG-name-test.txt']}]
        with patch.object(GazetteerEmbeddings, '_get_gazetteers', return_value=gazetteers), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict,
                                                                           full_matching=True,
                                                                           partial_matching=True,
                                                                           use_all_gazetteers=True)
            self.assertEqual(gazetteer_embedding.feature_list, ['O', 'S-0', 'B-0', 'E-0', 'I-0', 'S-1', 'B-1', 'E-1',
                                                                'I-1', 'S-2', 'B-2', 'E-2', 'I-2', 'S-3', 'B-3', 'E-3',
                                                                'I-3', '0', '1', '2', '3'])
            self.assertEqual(gazetteer_embedding.embedding_length, 21)

    def test_set_feature_list_good5(self):
        label_dict = MagicMock()
        label_dict.get_items.return_value = ['PER', 'ORG', 'LOC', 'MISC']
        gazetteers = [{'0': ['eng-CHEM-name-test.txt']},
                      {'1': ['eng-ORG-alias-test.txt']},
                      {'2': ['eng-LOC-name-test.txt']},
                      {'3': ['eng-ORG-name-test.txt']}]
        with patch.object(GazetteerEmbeddings, '_get_gazetteers', return_value=gazetteers), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict,
                                                                           full_matching=False,
                                                                           partial_matching=True,
                                                                           use_all_gazetteers=True)
            self.assertEqual(gazetteer_embedding.feature_list, ['O', 'S-0', 'B-0', 'E-0', 'I-0', 'S-1', 'B-1', 'E-1',
                                                                'I-1', 'S-2', 'B-2', 'E-2', 'I-2', 'S-3', 'B-3', 'E-3',
                                                                'I-3'])
            self.assertEqual(gazetteer_embedding.embedding_length, 17)

    def test_set_feature_list_good6(self):
        label_dict = MagicMock()
        label_dict.get_items.return_value = ['PER', 'ORG', 'LOC', 'MISC']
        gazetteers = [{'0': ['eng-CHEM-name-test.txt']},
                      {'1': ['eng-ORG-alias-test.txt']},
                      {'2': ['eng-LOC-name-test.txt']},
                      {'3': ['eng-ORG-name-test.txt']}]
        with patch.object(GazetteerEmbeddings, '_get_gazetteers', return_value=gazetteers), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict,
                                                                           full_matching=True,
                                                                           partial_matching=False,
                                                                           use_all_gazetteers=True)
            self.assertEqual(gazetteer_embedding.feature_list, ['O', '0', '1', '2', '3'])
            self.assertEqual(gazetteer_embedding.embedding_length, 5)

    def test_set_feature_list_no_label_dict1(self):
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           full_matching=True,
                                                                           partial_matching=True)
            self.assertEqual(gazetteer_embedding.feature_list, ['O'])
            self.assertEqual(gazetteer_embedding.embedding_length, 1)

    def test_set_feature_list_no_label_dict2(self):
        gazetteers = [{'0': ['eng-CHEM-name-test.txt']},
                      {'1': ['eng-ORG-alias-test.txt']},
                      {'2': ['eng-LOC-name-test.txt']},
                      {'3': ['eng-ORG-name-test.txt']}]
        with patch.object(GazetteerEmbeddings, '_get_gazetteers', return_value=gazetteers), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           full_matching=True,
                                                                           partial_matching=True,
                                                                           use_all_gazetteers=True)
            self.assertEqual(gazetteer_embedding.feature_list, ['O', 'S-0', 'B-0', 'E-0', 'I-0', 'S-1', 'B-1', 'E-1',
                                                                'I-1', 'S-2', 'B-2', 'E-2', 'I-2', 'S-3', 'B-3', 'E-3',
                                                                'I-3', '0', '1', '2', '3'])
            self.assertEqual(gazetteer_embedding.embedding_length, 21)

    def test_get_gazetteers_good1(self):
        label_dict = MagicMock()
        label_dict.get_items.return_value = ['PER', 'ORG', 'LOC', 'MISC']
        with patch.object(GazetteerEmbeddings, '_process_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                           label_dict=label_dict)
            self.assertEqual(gazetteer_embedding.gazetteer_file_dict_list, [{'PER': []},
                                                                            {'ORG': ['eng-ORG-alias-test.txt',
                                                                                     'eng-ORG-name-test.txt']},
                                                                            {'LOC': ['eng-LOC-name-test.txt']},
                                                                            {'MISC': []}])

    def test_get_gazetteers_good2(self):
        label_dict = MagicMock()
        label_dict.get_items.return_value = ['PER', 'ORG', 'LOC', 'MISC']
        with patch.object(GazetteerEmbeddings, '_process_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                           label_dict=label_dict,
                                                                           use_all_gazetteers=True)
            self.assertEqual(gazetteer_embedding.gazetteer_file_dict_list, [{'0': ['eng-CHEM-name-test.txt']},
                                                                            {'1': ['eng-ORG-alias-test.txt']},
                                                                            {'2': ['eng-LOC-name-test.txt']},
                                                                            {'3': ['eng-ORG-name-test.txt']}])

    def test_get_gazetteers_no_label_dict1(self):
        with patch.object(GazetteerEmbeddings, '_process_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources")
            self.assertEqual(gazetteer_embedding.gazetteer_file_dict_list, [])

    def test_get_gazetteers_no_label_dict2(self):
        with patch.object(GazetteerEmbeddings, '_process_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                           use_all_gazetteers=True)
            self.assertEqual(gazetteer_embedding.gazetteer_file_dict_list, [{'0': ['eng-CHEM-name-test.txt']},
                                                                            {'1': ['eng-ORG-alias-test.txt']},
                                                                            {'2': ['eng-LOC-name-test.txt']},
                                                                            {'3': ['eng-ORG-name-test.txt']}])

    def test_process_gazetteers_good1(self):
        gazetteer_files = [{'PER': []},
                           {'ORG': ['eng-ORG-alias-test.txt', 'eng-ORG-name-test.txt']},
                           {'LOC': ['eng-LOC-name-test.txt']},
                           {'MISC': []}
                           ]
        with patch.object(GazetteerEmbeddings, '_get_gazetteers', return_value=gazetteer_files), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                           label_dict=MagicMock())
            self.assertEqual(gazetteer_embedding.gazetteers_dicts['full_match'], self.full_match_dict)
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['full_match']), 62)

            self.assertEqual(gazetteer_embedding.gazetteers_dicts['partial_match'], self.partial_match_dict)
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['partial_match']), 170)

    def test_process_gazetteers_good2(self):
        gazetteer_files = [{'0': ['eng-CHEM-name-test.txt']},
                           {'1': ['eng-ORG-alias-test.txt']},
                           {'2': ['eng-LOC-name-test.txt']},
                           {'3': ['eng-ORG-name-test.txt']}]
        with patch.object(GazetteerEmbeddings, '_get_gazetteers', return_value=gazetteer_files), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                           use_all_gazetteers=True,
                                                                           label_dict=MagicMock())
            self.assertEqual(gazetteer_embedding.gazetteers_dicts['full_match'], self.full_match_dict2)
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['full_match']), 67)

            self.assertEqual(gazetteer_embedding.gazetteers_dicts['partial_match'], self.partial_match_dict2)
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['partial_match']), 176)

    def test_process_gazetteers_good3(self):
        gazetteer_files = [{'PER': []},
                           {'ORG': ['eng-ORG-alias-test.txt', 'eng-ORG-name-test.txt']},
                           {'LOC': ['eng-LOC-name-test.txt']},
                           {'MISC': []}
                           ]
        with patch.object(GazetteerEmbeddings, '_get_gazetteers', return_value=gazetteer_files), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                           tokenize_gazetteer_entries=True,
                                                                           label_dict=MagicMock())
            self.assertEqual(gazetteer_embedding.gazetteers_dicts['full_match'], self.full_match_dict3)
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['full_match']), 62)

            self.assertEqual(gazetteer_embedding.gazetteers_dicts['partial_match'], self.partial_match_dict3)
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['partial_match']), 167)

    def test_process_gazetteers_good4(self):
        gazetteer_files = [{'0': ['eng-CHEM-name-test.txt']},
                           {'1': ['eng-ORG-alias-test.txt']},
                           {'2': ['eng-LOC-name-test.txt']},
                           {'3': ['eng-ORG-name-test.txt']}]
        with patch.object(GazetteerEmbeddings, '_get_gazetteers', return_value=gazetteer_files), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                           use_all_gazetteers=True,
                                                                           tokenize_gazetteer_entries=True,
                                                                           label_dict=MagicMock())
            self.assertEqual(gazetteer_embedding.gazetteers_dicts['full_match'], self.full_match_dict4)
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['full_match']), 67)

            self.assertEqual(gazetteer_embedding.gazetteers_dicts['partial_match'], self.partial_match_dict4)
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['partial_match']), 173)

    def test_process_gazetteers_no_label_dict1(self):
        with patch.object(GazetteerEmbeddings, '_get_gazetteers', return_value=[]), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                           label_dict=MagicMock())
            self.assertEqual(gazetteer_embedding.gazetteers_dicts['full_match'], {})
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['full_match']), 0)

            self.assertEqual(gazetteer_embedding.gazetteers_dicts['partial_match'], {})
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['partial_match']), 0)

    def test_add_embeddings_internal_good1(self):
        sentences_1 = Sentence('I love Sandys Fort Spring!')
        sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
        sentence_list = [sentences_1, sentences_2]
        feature_list = ['O',
                        'B-PER', 'I-PER', 'E-PER', 'S-PER', 'B-ORG', 'I-ORG',
                        'E-ORG', 'S-ORG', 'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC',
                        'B-MISC', 'I-MISC', 'E-MISC', 'S-MISC']
        gazetteers = {'partial_match': self.partial_match_dict, 'full_match': self.full_match_dict}

        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list', return_value=feature_list), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers', return_value=gazetteers):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_matching=False,
                                                                           partial_matching=True)
            gazetteer_embedding.embed(sentence_list)

            for sentence in sentence_list:
                for token in sentence:
                    assert len(token.get_embedding()) == len(feature_list)

            # I
            self.assertEqual(torch.sum(sentence_list[0][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][0].embedding[8], torch.tensor(1))

            # love
            self.assertEqual(torch.sum(sentence_list[0][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][1].embedding[0], torch.tensor(1))

            # Sandys
            self.assertEqual(torch.sum(sentence_list[0][2].embedding), torch.tensor(3))
            self.assertEqual(sentence_list[0][2].embedding[5], torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[9], torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[8], torch.tensor(1))

            # Fort
            self.assertEqual(torch.sum(sentence_list[0][3].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][3].embedding[10], torch.tensor(1))
            self.assertEqual(sentence_list[0][3].embedding[6], torch.tensor(1))

            # Spring
            self.assertEqual(torch.sum(sentence_list[0][4].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][4].embedding[11], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[7], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[0], torch.tensor(1))
            ############################################################################
            # The
            self.assertEqual(torch.sum(sentence_list[1][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][0].embedding[0], torch.tensor(1))

            # Land
            self.assertEqual(torch.sum(sentence_list[1][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][1].embedding[5], torch.tensor(1))

            # Tenure
            self.assertEqual(torch.sum(sentence_list[1][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[6], torch.tensor(1))

            # Reform
            self.assertEqual(torch.sum(sentence_list[1][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][3].embedding[6], torch.tensor(1))

            # Association
            self.assertEqual(torch.sum(sentence_list[1][4].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][4].embedding[7], torch.tensor(1))

            # (
            self.assertEqual(torch.sum(sentence_list[1][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][5].embedding[0], torch.tensor(1))

            # LTRA
            self.assertEqual(torch.sum(sentence_list[1][6].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][6].embedding[0], torch.tensor(1))

            # )
            self.assertEqual(torch.sum(sentence_list[1][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][7].embedding[0], torch.tensor(1))

            # .
            self.assertEqual(torch.sum(sentence_list[1][8].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][8].embedding[0], torch.tensor(1))

    def test_add_embeddings_internal_good2(self):
        sentences_1 = Sentence('I !love! Sandys Fort Spring!')
        sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
        sentence_list = [sentences_1, sentences_2]
        feature_list = ['O', 'PER', 'ORG', 'LOC', 'MISC']
        gazetteers = {'partial_match': self.partial_match_dict, 'full_match': self.full_match_dict}

        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list', return_value=feature_list), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers', return_value=gazetteers):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_matching=True,
                                                                           partial_matching=False)
            gazetteer_embedding.embed(sentence_list)

            for sentence in sentence_list:
                for token in sentence:
                    assert len(token.get_embedding()) == len(feature_list)

            # I
            self.assertEqual(torch.sum(sentence_list[0][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][0].embedding[2], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][1].embedding[0], torch.tensor(1))

            # love
            self.assertEqual(torch.sum(sentence_list[0][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[0], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][3].embedding[0], torch.tensor(1))

            # Sandys
            self.assertEqual(torch.sum(sentence_list[0][4].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][4].embedding[2], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[3], torch.tensor(1))

            # Fort
            self.assertEqual(torch.sum(sentence_list[0][5].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][5].embedding[2], torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[3], torch.tensor(1))

            # Spring
            self.assertEqual(torch.sum(sentence_list[0][6].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][6].embedding[2], torch.tensor(1))
            self.assertEqual(sentence_list[0][6].embedding[3], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][7].embedding[0], torch.tensor(1))
            ############################################################################
            # The
            self.assertEqual(torch.sum(sentence_list[1][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][0].embedding[0], torch.tensor(1))

            # Land
            self.assertEqual(torch.sum(sentence_list[1][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][1].embedding[2], torch.tensor(1))

            # Tenure
            self.assertEqual(torch.sum(sentence_list[1][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[2], torch.tensor(1))

            # Reform
            self.assertEqual(torch.sum(sentence_list[1][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][3].embedding[2], torch.tensor(1))

            # Association
            self.assertEqual(torch.sum(sentence_list[1][4].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][4].embedding[2], torch.tensor(1))

            # (
            self.assertEqual(torch.sum(sentence_list[1][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][5].embedding[0], torch.tensor(1))

            # LTRA
            self.assertEqual(torch.sum(sentence_list[1][6].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][6].embedding[0], torch.tensor(1))

            # )
            self.assertEqual(torch.sum(sentence_list[1][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][7].embedding[0], torch.tensor(1))

            # .
            self.assertEqual(torch.sum(sentence_list[1][8].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][8].embedding[0], torch.tensor(1))

    def test_add_embeddings_internal_good3(self):
        sentences_1 = Sentence('I !love! Sandys Fort Spring!')
        sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
        sentences_3 = Sentence('Голоух')
        sentence_list = [sentences_1, sentences_2, sentences_3]
        feature_list = ['O',
                        'B-PER', 'I-PER', 'E-PER', 'S-PER', 'B-ORG', 'I-ORG',
                        'E-ORG', 'S-ORG', 'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC',
                        'B-MISC', 'I-MISC', 'E-MISC', 'S-MISC', 'PER', 'ORG',
                        'LOC', 'MISC']
        gazetteers = {'partial_match': self.partial_match_dict, 'full_match': self.full_match_dict}
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list', return_value=feature_list), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers', return_value=gazetteers):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_matching=True,
                                                                           partial_matching=True)
            gazetteer_embedding.embed(sentence_list)

            for sentence in sentence_list:
                for token in sentence:
                    assert len(token.get_embedding()) == len(feature_list)

            # I
            self.assertEqual(torch.sum(sentence_list[0][0].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][0].embedding[8], torch.tensor(1))
            self.assertEqual(sentence_list[0][0].embedding[18], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][1].embedding[0], torch.tensor(1))

            # love
            self.assertEqual(torch.sum(sentence_list[0][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[0], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][3].embedding[0], torch.tensor(1))

            # Sandys
            self.assertEqual(torch.sum(sentence_list[0][4].embedding), torch.tensor(5))
            self.assertEqual(sentence_list[0][4].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[19], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[9], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[8], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[5], torch.tensor(1))

            # Fort
            self.assertEqual(torch.sum(sentence_list[0][5].embedding), torch.tensor(4))
            self.assertEqual(sentence_list[0][5].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[19], torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[10], torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[6], torch.tensor(1))

            # Spring
            self.assertEqual(torch.sum(sentence_list[0][6].embedding), torch.tensor(4))
            self.assertEqual(sentence_list[0][6].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[0][6].embedding[19], torch.tensor(1))
            self.assertEqual(sentence_list[0][6].embedding[11], torch.tensor(1))
            self.assertEqual(sentence_list[0][6].embedding[7], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][7].embedding[0], torch.tensor(1))
            ############################################################################
            # The
            self.assertEqual(torch.sum(sentence_list[1][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][0].embedding[0], torch.tensor(1))

            # Land
            self.assertEqual(torch.sum(sentence_list[1][1].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][1].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[1][1].embedding[5], torch.tensor(1))

            # Tenure
            self.assertEqual(torch.sum(sentence_list[1][2].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][2].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[6], torch.tensor(1))

            # Reform
            self.assertEqual(torch.sum(sentence_list[1][3].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][3].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[1][3].embedding[6], torch.tensor(1))

            # Association
            self.assertEqual(torch.sum(sentence_list[1][4].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][4].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[1][4].embedding[7], torch.tensor(1))

            # (
            self.assertEqual(torch.sum(sentence_list[1][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][5].embedding[0], torch.tensor(1))

            # LTRA
            self.assertEqual(torch.sum(sentence_list[1][6].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][6].embedding[0], torch.tensor(1))

            # )
            self.assertEqual(torch.sum(sentence_list[1][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][7].embedding[0], torch.tensor(1))

            # .
            self.assertEqual(torch.sum(sentence_list[1][8].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][8].embedding[0], torch.tensor(1))
            ############################################################################
            # Голоух
            self.assertEqual(torch.sum(sentence_list[2][0].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[2][0].embedding[12], torch.tensor(1))
            self.assertEqual(sentence_list[2][0].embedding[19], torch.tensor(1))

    def test_add_embeddings_internal_good4(self):
        sentences_1 = Sentence('7α-methylestr-4-ene-3,17-dione is a chemical')
        sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
        sentence_list = [sentences_1, sentences_2]
        feature_list = ['O', 'S-0', 'B-0', 'E-0', 'I-0', 'S-1', 'B-1', 'E-1',
                        'I-1', 'S-2', 'B-2', 'E-2', 'I-2', 'S-3', 'B-3', 'E-3',
                        'I-3', '0', '1', '2', '3']
        gazetteers = {'partial_match': self.partial_match_dict2, 'full_match': self.full_match_dict2}
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list', return_value=feature_list), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers', return_value=gazetteers):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_matching=True,
                                                                           partial_matching=True,
                                                                           use_all_gazetteers=True)
            gazetteer_embedding.embed(sentence_list)

            for sentence in sentence_list:
                for token in sentence:
                    assert len(token.get_embedding()) == len(feature_list)

            # 7α-methylestr-4-ene-3,17-dione
            self.assertEqual(torch.sum(sentence_list[0][0].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][0].embedding[1], torch.tensor(1))
            self.assertEqual(sentence_list[0][0].embedding[17], torch.tensor(1))

            # is
            self.assertEqual(torch.sum(sentence_list[0][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][1].embedding[0], torch.tensor(1))

            # a
            self.assertEqual(torch.sum(sentence_list[0][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[0], torch.tensor(1))

            # chemical
            self.assertEqual(torch.sum(sentence_list[0][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][3].embedding[0], torch.tensor(1))
            ############################################################################
            # The
            self.assertEqual(torch.sum(sentence_list[1][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][0].embedding[0], torch.tensor(1))

            # Land
            self.assertEqual(torch.sum(sentence_list[1][1].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][1].embedding[14], torch.tensor(1))
            self.assertEqual(sentence_list[1][1].embedding[20], torch.tensor(1))

            # Tenure
            self.assertEqual(torch.sum(sentence_list[1][2].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][2].embedding[16], torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[20], torch.tensor(1))

            # Reform
            self.assertEqual(torch.sum(sentence_list[1][3].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][3].embedding[16], torch.tensor(1))
            self.assertEqual(sentence_list[1][3].embedding[20], torch.tensor(1))

            # Association
            self.assertEqual(torch.sum(sentence_list[1][4].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][4].embedding[15], torch.tensor(1))
            self.assertEqual(sentence_list[1][4].embedding[20], torch.tensor(1))

            # (
            self.assertEqual(torch.sum(sentence_list[1][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][5].embedding[0], torch.tensor(1))

            # LTRA
            self.assertEqual(torch.sum(sentence_list[1][6].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][6].embedding[0], torch.tensor(1))

            # )
            self.assertEqual(torch.sum(sentence_list[1][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][7].embedding[0], torch.tensor(1))

            # .
            self.assertEqual(torch.sum(sentence_list[1][8].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][8].embedding[0], torch.tensor(1))

    def test_add_embeddings_internal_good5(self):
        sentences_1 = Sentence('7α-methylestr-4-ene-3,17-dione is a chemical')
        sentences_2 = Sentence('I love Sandys Fort Spring!')
        sentence_list = [sentences_1, sentences_2]
        feature_list = ['O', 'S-0', 'B-0', 'E-0', 'I-0', 'S-1', 'B-1', 'E-1',
                        'I-1', 'S-2', 'B-2', 'E-2', 'I-2', 'S-3', 'B-3', 'E-3',
                        'I-3']
        gazetteers = {'partial_match': self.partial_match_dict2, 'full_match': self.full_match_dict2}
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list', return_value=feature_list), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers', return_value=gazetteers):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_matching=False,
                                                                           partial_matching=True,
                                                                           use_all_gazetteers=True)
            gazetteer_embedding.embed(sentence_list)

            for sentence in sentence_list:
                for token in sentence:
                    assert len(token.get_embedding()) == len(feature_list)

            # 7α-methylestr-4-ene-3,17-dione
            self.assertEqual(torch.sum(sentence_list[0][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][0].embedding[1], torch.tensor(1))

            # is
            self.assertEqual(torch.sum(sentence_list[0][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][1].embedding[0], torch.tensor(1))

            # a
            self.assertEqual(torch.sum(sentence_list[0][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[0], torch.tensor(1))

            # chemical
            self.assertEqual(torch.sum(sentence_list[0][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][3].embedding[0], torch.tensor(1))
            ############################################################################
            # I
            self.assertEqual(torch.sum(sentence_list[1][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][0].embedding[5], torch.tensor(1))

            # love
            self.assertEqual(torch.sum(sentence_list[1][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][1].embedding[0], torch.tensor(1))

            # Sandys
            self.assertEqual(torch.sum(sentence_list[1][2].embedding), torch.tensor(3))
            self.assertEqual(sentence_list[1][2].embedding[10], torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[14], torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[5], torch.tensor(1))

            # Fort
            self.assertEqual(torch.sum(sentence_list[1][3].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][3].embedding[12], torch.tensor(1))
            self.assertEqual(sentence_list[1][3].embedding[16], torch.tensor(1))

            # Spring
            self.assertEqual(torch.sum(sentence_list[1][4].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][4].embedding[11], torch.tensor(1))
            self.assertEqual(sentence_list[1][4].embedding[15], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[1][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][5].embedding[0], torch.tensor(1))

    def test_add_embeddings_internal_good6(self):
        sentences_1 = Sentence('7α-methylestr-4-ene-3,17-dione is a chemical')
        sentences_2 = Sentence('America de Manta')
        sentence_list = [sentences_1, sentences_2]
        feature_list = ['O', '0', '1', '2', '3']
        gazetteers = {'partial_match': self.partial_match_dict2, 'full_match': self.full_match_dict2}
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list', return_value=feature_list), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers', return_value=gazetteers):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_matching=True,
                                                                           partial_matching=False,
                                                                           use_all_gazetteers=True)
            gazetteer_embedding.embed(sentence_list)

            for sentence in sentence_list:
                for token in sentence:
                    assert len(token.get_embedding()) == len(feature_list)

            # 7α-methylestr-4-ene-3,17-dione
            self.assertEqual(torch.sum(sentence_list[0][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][0].embedding[1], torch.tensor(1))

            # is
            self.assertEqual(torch.sum(sentence_list[0][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][1].embedding[0], torch.tensor(1))

            # a
            self.assertEqual(torch.sum(sentence_list[0][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[0], torch.tensor(1))

            # chemical
            self.assertEqual(torch.sum(sentence_list[0][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][3].embedding[0], torch.tensor(1))
            ############################################################################
            # America
            self.assertEqual(torch.sum(sentence_list[1][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][0].embedding[2], torch.tensor(1))

            # de
            self.assertEqual(torch.sum(sentence_list[1][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][1].embedding[2], torch.tensor(1))

            # Manta
            self.assertEqual(torch.sum(sentence_list[1][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[2], torch.tensor(1))

    def test_add_embeddings_internal_no_label_dict(self):
        sentences_1 = Sentence('I !love! Sandys Fort Spring!')
        sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
        sentence_list = [sentences_1, sentences_2]
        feature_list = ['O']
        gazetteers = {'partial_match': {}, 'full_match': {}}
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list', return_value=feature_list), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers', return_value=gazetteers):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_matching=True,
                                                                           partial_matching=True)
            gazetteer_embedding.embed(sentence_list)

            for sentence in sentence_list:
                for token in sentence:
                    assert len(token.get_embedding()) == len(feature_list)

            # I
            self.assertEqual(torch.sum(sentence_list[0][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][0].embedding[0], torch.tensor(1))

            # love
            self.assertEqual(torch.sum(sentence_list[0][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][1].embedding[0], torch.tensor(1))

            # Sandys
            self.assertEqual(torch.sum(sentence_list[0][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[0], torch.tensor(1))

            # Fort
            self.assertEqual(torch.sum(sentence_list[0][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][3].embedding[0], torch.tensor(1))

            # Spring
            self.assertEqual(torch.sum(sentence_list[0][4].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[0], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[0], torch.tensor(1))
            ############################################################################
            # The
            self.assertEqual(torch.sum(sentence_list[1][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][0].embedding[0], torch.tensor(1))

            # Land
            self.assertEqual(torch.sum(sentence_list[1][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][1].embedding[0], torch.tensor(1))

            # Tenure
            self.assertEqual(torch.sum(sentence_list[1][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[0], torch.tensor(1))

            # Reform
            self.assertEqual(torch.sum(sentence_list[1][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][3].embedding[0], torch.tensor(1))

            # Association
            self.assertEqual(torch.sum(sentence_list[1][4].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][4].embedding[0], torch.tensor(1))

            # (
            self.assertEqual(torch.sum(sentence_list[1][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][5].embedding[0], torch.tensor(1))

            # LTRA
            self.assertEqual(torch.sum(sentence_list[1][6].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][6].embedding[0], torch.tensor(1))

            # )
            self.assertEqual(torch.sum(sentence_list[1][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][7].embedding[0], torch.tensor(1))

            # .
            self.assertEqual(torch.sum(sentence_list[1][8].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][8].embedding[0], torch.tensor(1))
