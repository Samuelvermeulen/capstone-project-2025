
import pandas as pd
import numpy as np
import re

def clean_height(height_str):
    """Convertir 179cm en 179"""
    if pd.isna(height_str):
        return np.nan
    if isinstance(height_str, str) and 'cm' in height_str:
        return int(height_str.replace('cm', '').strip())
    try:
        return int(height_str)
    except:
        return np.nan

def clean_weight(weight_str):
    """Convertir 69kg en 69"""
    if pd.isna(weight_str):
        return np.nan
    if isinstance(weight_str, str) and 'kg' in weight_str:
        return int(weight_str.replace('kg', '').strip())
    try:
        return int(weight_str)
    except:
        return np.nan

def convert_value(value_str):
    """Convertir €107.5M en 107500000"""
    if pd.isna(value_str) or value_str == '€0' or value_str == '0':
        return 0
    
    value_str = str(value_str).replace('€', '').strip()
    
    # Gérer les valeurs en K (milliers)
    if 'K' in value_str:
        number = float(value_str.replace('K', '').strip())
        return int(number * 1000)
    
    # Gérer les valeurs en M (millions)
    elif 'M' in value_str:
        number = float(value_str.replace('M', '').strip())
        return int(number * 1000000)
    
    # Gérer les valeurs sans suffixe
    else:
        try:
            return int(float(value_str))
        except:
            return 0

def clean_player_names(player_name):
    """Nettoyer et corriger les noms des joueurs"""
    if pd.isna(player_name):
        return player_name
    
    name = str(player_name)
    
    # 1. Supprimer les numéros au début (avec espace normal ou insécable)
    name = re.sub(r'^\d+\s*', '', name)  # Pour espace normal
    name = re.sub(r'^\d+ ', '', name)    # Pour espace insécable
    
    # 2. Corriger les noms asiatiques
    asian_names = {
        '武藤 嘉紀': 'Yoshinori Muto',
        '손흥민 孙兴慜': 'Heung-min Son', 
        '기성용 寄诚庸': 'Sung-yueng Ki',
        '岡崎 慎司': 'Shinji Okazaki',
        '吉田 麻也': 'Maya Yoshida',
        '南野 拓実': 'Takumi Minamino',
        'H. Son': 'Heung-min Son'
    }
    
    for asian_name, english_name in asian_names.items():
        if asian_name in name:
            name = english_name
            break
    
    # 3. Expansion des noms abrégés (liste partielle basée sur les noms connus)
    name_expansions = {
        'J. Henderson': 'Jordan Henderson',
        'I. Gündoğan': 'İlkay Gündoğan',
        'K. Walker': 'Kyle Walker',
        'S. Mané': 'Sadio Mané',
        'Y. Tielemans': 'Youri Tielemans',
        'N. Kanté': 'N\'Golo Kanté',
        'G. Lo Celso': 'Giovani Lo Celso',
        'Rúben Neves': 'Rúben Neves',
        'A. Robertson': 'Andrew Robertson',
        'T. Partey': 'Thomas Partey',
        'Diogo Jota': 'Diogo Jota',
        'O. Zinchenko': 'Oleksandr Zinchenko',
        'M. Rashford': 'Marcus Rashford',
        'D. van de Beek': 'Donny van de Beek',
        'J. Milner': 'James Milner',
        'J. Ward-Prowse': 'James Ward-Prowse',
        'J. Vardy': 'Jamie Vardy',
        'Jorginho': 'Jorginho',
        'Marcos Alonso': 'Marcos Alonso',
        'L. Shaw': 'Luke Shaw',
        'Bernardo Silva': 'Bernardo Silva',
        'João Moutinho': 'João Moutinho',
        'M. Kovačić': 'Mateo Kovačić',
        'Richarlison': 'Richarlison',
        'A. Oxlade-Chamberlain': 'Alex Oxlade-Chamberlain',
        'T. Ndombele': 'Tanguy Ndombele',
        'D. Alli': 'Dele Alli',
        'M. Antonio': 'Michail Antonio',
        'W. Ndidi': 'Wilfred Ndidi',
        'R. Sterling': 'Raheem Sterling',
        'T. Souček': 'Tomáš Souček',
        'J. McGinn': 'John McGinn',
        'Y. Bissouma': 'Yves Bissouma',
        'V. van Dijk': 'Virgil van Dijk',
        'S. McTominay': 'Scott McTominay',
        'M. Cornet': 'Maxwel Cornet',
        'M. Mount': 'Mason Mount',
        'Raphinha': 'Raphinha',
        'R. James': 'Reece James',
        'Rodri': 'Rodri',
        'Andreas Pereira': 'Andreas Pereira',
        'A. Young': 'Ashley Young',
        'J. Lingard': 'Jesse Lingard',
        'E. Cavani': 'Edinson Cavani',
        'L. Dendoncker': 'Leander Dendoncker',
        'R. Jiménez': 'Raúl Jiménez',
        'P. Højbjerg': 'Pierre-Emile Højbjerg',
        'M. Ødegaard': 'Martin Ødegaard',
        'K. Phillips': 'Kalvin Phillips',
        'J. Maddison': 'James Maddison',
        'A. Doucouré': 'Abdoulaye Doucouré',
        'H. Ziyech': 'Hakim Ziyech',
        'K. Tierney': 'Kieran Tierney',
        'V. Coufal': 'Vladimír Coufal',
        'Azpilicueta': 'César Azpilicueta',
        'N. Keïta': 'Naby Keïta',
        'R. Lukaku': 'Romelu Lukaku',
        'E. Buendía': 'Emiliano Buendía',
        'Gabriel Jesus': 'Gabriel Jesus',
        'Pablo Fornals': 'Pablo Fornals',
        'P. Foden': 'Phil Foden',
        'M. Sissoko': 'Moussa Sissoko',
        'B. Chilwell': 'Ben Chilwell',
        'A. Cresswell': 'Aaron Cresswell',
        'Ayoze Pérez': 'Ayoze Pérez',
        'D. Ings': 'Danny Ings',
        'G. Sigurðsson': 'Gylfi Sigurðsson',
        'M. Ritchie': 'Matt Ritchie',
        'Rodrigo': 'Rodrigo',
        'P. Aubameyang': 'Pierre-Emerick Aubameyang',
        'A. Lacazette': 'Alexandre Lacazette',
        'B. Saka': 'Bukayo Saka',
        'David Luiz': 'David Luiz',
        'P. Vieira': 'Patrick Vieira',
        'K. Havertz': 'Kai Havertz',
        'R. Mahrez': 'Riyad Mahrez',
        'S. Coleman': 'Seamus Coleman',
        'J. Brownhill': 'Josh Brownhill',
        'L. Milivojević': 'Luka Milivojević',
        'P. Groß': 'Pascal Groß',
        'J. Hendrick': 'Jeff Hendrick',
        'J. Grealish': 'Jack Grealish',
        'Thiago Silva': 'Thiago Silva',
        'J. Schlupp': 'Jeffrey Schlupp',
        'M. Doherty': 'Matt Doherty',
        'S. Bergwijn': 'Steven Bergwijn',
        'Cédric': 'Cédric Soares',
        'Nélson Semedo': 'Nélson Semedo',
        'F. Delph': 'Fabian Delph',
        'T. Werner': 'Timo Werner',
        'F. Schär': 'Fabian Schär',
        'L. Bailey': 'Leon Bailey',
        'M. Albrighton': 'Marc Albrighton',
        'J. Sancho': 'Jadon Sancho',
        'A. Wan-Bissaka': 'Aaron Wan-Bissaka',
        'N. Pépé': 'Nicolas Pépé',
        'A. Martial': 'Anthony Martial',
        'B. Mendy': 'Benjamin Mendy',
        'S. Dallas': 'Stuart Dallas',
        'Douglas Luiz': 'Douglas Luiz',
        'A. Lallana': 'Adam Lallana',
        'R. Barkley': 'Ross Barkley',
        'M. Lowton': 'Matthew Lowton',
        'A. Mac Allister': 'Alexis Mac Allister',
        'Ferran Torres': 'Ferran Torres',
        'André Gomes': 'André Gomes',
        'M. Klich': 'Mateusz Klich',
        'M. Elyounoussi': 'Mohamed Elyounoussi',
        'N. Aké': 'Nathan Aké',
        'Diogo Dalot': 'Diogo Dalot',
        'Jonny': 'Jonny Castro',
        'M. Noble': 'Mark Noble',
        'B. Davies': 'Ben Davies',
        'J. McArthur': 'James McArthur',
        'K. McLean': 'Kenny McLean',
        'M. Almirón': 'Miguel Almirón',
        'D. Rose': 'Danny Rose',
        'B. Decordova-Reid': 'Bobby Decordova-Reid',
        'J. Shelvey': 'Jonjo Shelvey',
        'A. Maitland-Niles': 'Ainsley Maitland-Niles',
        'K. Tsimikas': 'Kostas Tsimikas',
        'D. McNeil': 'Dwight McNeil',
        'D. Rice': 'Declan Rice',
        'A. Laporte': 'Aymeric Laporte',
        'Junior Firpo': 'Junior Firpo',
        'T. Cairney': 'Tom Cairney',
        'J. Ayew': 'Jordan Ayew',
        'R. Bertrand': 'Ryan Bertrand',
        'S. Kolašinac': 'Sead Kolašinac',
        'M. Cash': 'Matty Cash',
        'W. Zaha': 'Wilfried Zaha',
        'A. Armstrong': 'Adam Armstrong',
        'H. Winks': 'Harry Winks',
        'H. Barnes': 'Harvey Barnes',
        'Kiko Femenía': 'Kiko Femenía',
        'S. Parker': 'Scott Parker',
        'W. Hughes': 'Will Hughes',
        'B. Traoré': 'Bertrand Traoré',
        'V. Lindelöf': 'Victor Lindelöf',
        'E. Dier': 'Eric Dier',
        'I. Toney': 'Ivan Toney',
        'O. Edouard': 'Odsonne Édouard',
        'Willian': 'Willian',
        'N. Matić': 'Nemanja Matić',
        'A. Townsend': 'Andros Townsend',
        'D. Calvert-Lewin': 'Dominic Calvert-Lewin',
        'G. Xhaka': 'Granit Xhaka',
        'S. March': 'Solly March',
        'O. Watkins': 'Ollie Watkins',
        'J. Stones': 'John Stones',
        'R. Sessegnon': 'Ryan Sessegnon',
        'C. Taylor': 'Charlie Taylor',
        'H. Wilson': 'Harry Wilson',
        'C. Pulisic': 'Christian Pulisic',
        'M. Elneny': 'Mohamed Elneny',
        'T. Davies': 'Tom Davies',
        'N. Mendy': 'Nampalys Mendy',
        'J. Willock': 'Joe Willock',
        'M. Greenwood': 'Mason Greenwood',
        'H. Maguire': 'Harry Maguire',
        'C. Wilson': 'Callum Wilson',
        'T. Castagne': 'Timothy Castagne',
        'J. Bowen': 'Jarrod Bowen',
        'C. Kouyaté': 'Cheikhou Kouyaté',
        'H. Reed': 'Harrison Reed',
        'T. Cleverley': 'Tom Cleverley',
        'Pablo Hernández': 'Pablo Hernández',
        'J. Cork': 'Jack Cork',
        'P. Zabaleta': 'Pablo Zabaleta',
        'R. Giggs': 'Ryan Giggs',
        'N. Clyne': 'Nathaniel Clyne',
        'A. Sambi Lokonga': 'Albert Sambi Lokonga',
        'C. Adams': 'Che Adams',
        'C. Coady': 'Conor Coady',
        'E. Eze': 'Eberechi Eze',
        'K. Iheanacho': 'Kelechi Iheanacho',
        'N. Maupay': 'Neal Maupay',
        'R. Fraser': 'Ryan Fraser',
        'L. Trossard': 'Leandro Trossard',
        'C. Nørgaard': 'Christian Nørgaard',
        'R. Varane': 'Raphaël Varane',
        'A. Barnes': 'Ashley Barnes',
        'A. Yarmolenko': 'Andriy Yarmolenko',
        'D. James': 'Daniel James',
        'R. Saïss': 'Romain Saïss',
        'R. Loftus-Cheek': 'Ruben Loftus-Cheek',
        'D. Gosling': 'Dan Gosling',
        'J. Riedewald': 'Jairo Riedewald',
        'P. Bamford': 'Patrick Bamford',
        'C. Chambers': 'Calum Chambers',
        'M. Vydra': 'Matěj Vydra',
        'A. Knockaert': 'Anthony Knockaert',
        'R. Fredericks': 'Ryan Fredericks',
        'T. Minamino': 'Takumi Minamino',
        'J. Harrison': 'Jack Harrison',
        'J. Kenny': 'Jonjoe Kenny',
        'Adama Traoré': 'Adama Traoré',
        'J. Murphy': 'Jacob Murphy',
        'J. King': 'Joshua King',
        'B. Mbeumo': 'Bryan Mbeumo',
        'M. Targett': 'Matt Targett',
        'C. Jones': 'Curtis Jones',
        'C. Wood': 'Chris Wood',
        'K. Zouma': 'Kurt Zouma',
        'J. Justin': 'James Justin',
        'Sergi Canós': 'Sergi Canós',
        'A. Rüdiger': 'Antonio Rüdiger',
        'Ivan Cavaleiro': 'Ivan Cavaleiro',
        'B. Williams': 'Brandon Williams',
        'N. Chalobah': 'Nathaniel Chalobah',
        'Trincão': 'Francisco Trincão',
        'D. Amartey': 'Daniel Amartey',
        'T. Walcott': 'Theo Walcott',
        'C. Tosun': 'Cenk Tosun',
        'Hwang Hee Chan': 'Hwang Hee-chan',
        'S. Rondón': 'Salomón Rondón',
        'A. El Ghazi': 'Anwar El Ghazi',
        'C. Gallagher': 'Conor Gallagher',
        'A. Forshaw': 'Adam Forshaw',
        'M. Olise': 'Michael Olise',
        'C. Hudson-Odoi': 'Callum Hudson-Odoi',
        'Rúben Dias': 'Rúben Dias',
        'M. Sarr': 'Malang Sarr',
        'M. Holgate': 'Mason Holgate',
        'Gabriel Martinelli': 'Gabriel Martinelli',
        'S. Benrahma': 'Saïd Benrahma',
        'L. Ayling': 'Luke Ayling',
        'E. Dennis': 'Emmanuel Dennis',
        'Juan Mata': 'Juan Mata',
        'E. Pieters': 'Erik Pieters',
        'M. Djenepo': 'Moussa Djenepo',
        'M. Nakamba': 'Marvelous Nakamba',
        'J. Matip': 'Joël Matip',
        'I. Hayden': 'Isaac Hayden',
        'J. Tarkowski': 'James Tarkowski',
        'K. Tete': 'Kenny Tete',
        'Nuno Tavares': 'Nuno Tavares',
        'Daniel Podence': 'Daniel Podence',
        'S. Ghoddos': 'Saman Ghoddos',
        'S. Longstaff': 'Sean Longstaff',
        'M. Lanzini': 'Manuel Lanzini',
        'A. Saint-Maximin': 'Allan Saint-Maximin',
        'Joelinton': 'Joelinton',
        'J. Onomah': 'Josh Onomah',
        'J. Lewis': 'Jamal Lewis',
        'D. Welbeck': 'Danny Welbeck',
        'B. Godfrey': 'Ben Godfrey',
        'J. Veltman': 'Joël Veltman',
        'T. Mings': 'Tyrone Mings',
        'P. Daka': 'Patson Daka',
        'A. Lookman': 'Ademola Lookman',
        'B. Gilmour': 'Billy Gilmour',
        'Trezeguet': 'Mahmoud Hassan',
        'A. Lennon': 'Aaron Lennon',
        'D. Origi': 'Divock Origi',
        'Pedro Neto': 'Pedro Neto',
        'C. Romero': 'Cristian Romero',
        'C. Söyüncü': 'Çağlar Söyüncü',
        'A. Robinson': 'Antonee Robinson',
        'Manquillo': 'Javier Manquillo',
        'C. Christie': 'Cyrus Christie',
        'K. Dewsbury-Hall': 'Kiernan Dewsbury-Hall',
        'Y. Wissa': 'Yoane Wissa',
        'E. Konsa': 'Ezri Konsa',
        'E. Bailly': 'Eric Bailly',
        'M. Keane': 'Michael Keane',
        'I. Sarr': 'Ismaïla Sarr',
        'S. Long': 'Shane Long',
        'H. Choudhury': 'Hamza Choudhury',
        'T. Roberts': 'Tyler Roberts',
        'K. Walker-Peters': 'Kyle Walker-Peters',
        'K. Ajer': 'Kristoffer Ajer',
        'A. Iwobi': 'Alex Iwobi',
        'J. Sargent': 'Josh Sargent',
        'P. Dummett': 'Paul Dummett',
        'T. Lamptey': 'Tariq Lamptey',
        'Oriol Romeu': 'Oriol Romeu',
        'T. Pukki': 'Teemu Pukki',
        'T. Cantwell': 'Todd Cantwell',
        'J. Gomez': 'Joe Gomez',
        'N. Redmond': 'Nathan Redmond',
        'A. Tuanzebe': 'Axel Tuanzebe',
        'M. Aarons': 'Max Aarons',
        'J. Ward': 'Joel Ward',
        'J. Tanganga': 'Japhet Tanganga',
        'S. Alzate': 'Steven Alzate',
        'B. Mee': 'Ben Mee',
        'D. Gray': 'Demarai Gray',
        'Diego Llorente': 'Diego Llorente',
        'T. Ream': 'Tim Ream',
        'W. Smallbone': 'Will Smallbone',
        'E. Smith Rowe': 'Emile Smith Rowe',
        'H. White': 'Harvey White',
        'G. Neville': 'Gary Neville',
        'D. Gayle': 'Dwight Gayle',
        'O. Skipp': 'Oliver Skipp',
        'A. Masina': 'Adam Masina',
        'A. Mitrović': 'Aleksandar Mitrović',
        'L. Dunk': 'Lewis Dunk',
        'W. Fofana': 'Wesley Fofana',
        'C. Benteke': 'Christian Benteke',
        'J. Andersen': 'Joachim Andersen',
        'D. Sánchez': 'Davinson Sánchez',
        'W. Boly': 'Willy Boly',
        'R. Aït Nouri': 'Rayan Aït Nouri',
        'J. Stephens': 'Jack Stephens',
        'T. Tomiyasu': 'Takehiro Tomiyasu',
        'A. Carroll': 'Andy Carroll',
        'Y. Mina': 'Yerry Mina',
        'T. Chalobah': 'Trevoh Chalobah',
        'C. Dawson': 'Craig Dawson',
        'A. Ogbonna': 'Angelo Ogbonna',
        'R. Koch': 'Robin Koch',
        'A. Christensen': 'Andreas Christensen',
        'K. Hause': 'Kortney Hause',
        'J. Evans': 'Jonny Evans',
        'J. Ramsey': 'Jacob Ramsey',
        'J. Mateta': 'Jean-Philippe Mateta',
        'Bryan Gil': 'Bryan Gil',
        'E. Nketiah': 'Edward Nketiah',
        'Gabriel': 'Gabriel Magalhães',
        'R. Holding': 'Rob Holding',
        'J. Vestergaard': 'Jannik Vestergaard',
        'I. Konaté': 'Ibrahima Konaté',
        'P. Jones': 'Phil Jones',
        'M. Kelly': 'Martin Kelly',
        'H. Elliott': 'Harvey Elliott',
        'K. Davis': 'Keinan Davis',
        'M. Kilman': 'Max Kilman',
        'I. Diop': 'Issa Diop',
        'C. Palmer': 'Cole Palmer',
        'N. Williams': 'Neco Williams',
        'B. White': 'Ben White',
        'A. Webster': 'Adam Webster',
        'C. Patino': 'Charlie Patino',
        'B. Johnson': 'Brennan Johnson',
        'P. Jansson': 'Pontus Jansson',
        'L. Cooper': 'Liam Cooper',
        'Zanka': 'Mathias Jørgensen',
        'T. Livramento': 'Tino Livramento',
        'João Pedro': 'João Pedro',
        'J. Terry': 'John Terry',
        'H. Mejbri': 'Hannibal Mejbri',
        'D. Burn': 'Dan Burn',
        'J. Gelhardt': 'Joe Gelhardt',
        'A. Gordon': 'Anthony Gordon',
        'M. Guéhi': 'Marc Guéhi',
        'M. Salisu': 'Mohammed Salisu',
        'C. Musonda': 'Charly Musonda',
        'E. Simms': 'Ellis Simms',
        'P. Crouch': 'Peter Crouch',
        'A. Elanga': 'Anthony Elanga',
        'J. Lascelles': 'Jamaal Lascelles',
        'C. Summerville': 'Crysencio Summerville',
        'C. Chukwuemeka': 'Carney Chukwuemeka',
        'Fábio Silva': 'Fábio Silva',
        'T. Mengi': 'Teden Mengi',
        'J. Clarke': 'Jack Clarke',
        'N. Ferguson': 'Nathan Ferguson',
        'F. Fernández': 'Federico Fernández',
        'T. Adarabioyo': 'Tosin Adarabioyo',
        'F. Carvalho': 'Fábio Carvalho',
        'Pablo Marí': 'Pablo Marí',
        'N. Tella': 'Nathan Tella',
        'Lyanco': 'Lyanco',
        'Ederson': 'Ederson',
        'S. Campbell': 'Sol Campbell',
        'S. Shoretire': 'Shola Shoretire',
        'C. Drameh': 'Cody Drameh',
        'L. Bogarde': 'Lamare Bogarde',
        'Kayky': 'Kayky',
        'R. Lavia': 'Roméo Lavia',
        'J. McAtee': 'James McAtee',
        'J. Bednarek': 'Jan Bednarek',
        'K. Long': 'Kevin Long',
        'J. Carragher': 'Jamie Carragher',
        'L. Delap': 'Liam Delap',
        'C. Cathcart': 'Craig Cathcart',
        'N. Phillips': 'Nat Phillips',
        'A. Mawson': 'Alfie Mawson',
        'P. Mertesacker': 'Per Mertesacker',
        'G. Hanley': 'Grant Hanley',
        'J. Rodon': 'Joe Rodon',
        'B. Gibson': 'Ben Gibson',
        'C. Goode': 'Charlie Goode',
        'N. Collins': 'Nathan Collins',
        'A. Idah': 'Adam Idah',
        'J. Rak-Sakyi': 'Jesurun Rak-Sakyi',
        'W. Troost-Ekong': 'William Troost-Ekong',
        'J. Tomkins': 'James Tomkins',
        'D. Scarlett': 'Dane Scarlett',
        'F. Sierralta': 'Francisco Sierralta',
        'J. Pickford': 'Jordan Pickford',
        'A. Okoflex': 'Armstrong Okoflex',
        'S. Edozie': 'Sam Edozie',
        'L. Mbete': 'Luke Mbete',
        'Hugo Bueno': 'Hugo Bueno',
        'J. Stansfield': 'Jay Stansfield',
        'J. Sarmiento': 'Jeremy Sarmiento',
        'J. Norris': 'James Norris',
        'M. Fagan-Walcott': 'Malachi Fagan-Walcott',
        'Christian Marques': 'Christian Marques',
        'M. Lavinier': 'Marcel Lavinier',
        'M. Bondswell': 'Matthew Bondswell',
        'R. Astley': 'Ryan Astley',
        'B. Hangeland': 'Brede Hangeland',
        'L. Hjelde': 'Leo Hjelde',
        'T. Small': 'Thierry Small',
        'A. Omobamidele': 'Andrew Omobamidele',
        'T. Klose': 'Timm Klose',
        'C. Zimmermann': 'Christoph Zimmermann',
        'E. Martínez': 'Emiliano Martínez',
        'P. Gazzaniga': 'Paulo Gazzaniga',
        'O. Offiah': 'Odel Offiah',
        'O. Solskjær': 'Ole Gunnar Solskjær',
        'De Gea': 'David de Gea',
        'J. Branthwaite': 'Jarrad Branthwaite',
        'Y. Mosquera': 'Yerson Mosquera',
        'Z. Monlouis': 'Zane Monlouis',
        'O. Olufunwa': 'Oludare Olufunwa',
        'A. Alese': 'Aji Alese',
        'L. Dobbin': 'Lewis Dobbin',
        'C. Archer': 'Cameron Archer',
        'Alisson': 'Alisson Becker',
        'C. Cresswell': 'Charlie Cresswell',
        'A. Areola': 'Alphonse Areola',
        'A. Tsoungui': 'Antef Tsoungui',
        'H. Lloris': 'Hugo Lloris',
        'Guaita': 'Vicente Guaita',
        'L. Racic': 'Luka Racic',
        'K. Schmeichel': 'Kasper Schmeichel',
        'B. Leno': 'Bernd Leno',
        'W. Morgan': 'Wes Morgan',
        'J. Furlong': 'James Furlong',
        'M. Dúbravka': 'Martin Dúbravka',
        'B. Lyons-Foster': 'Brooklyn Lyons-Foster',
        'Z. Steffen': 'Zack Steffen',
        'S. Swinkels': 'Sil Swinkels',
        'J. Butland': 'Jack Butland',
        'Fabricio': 'Fabricio Agosto',
        'B. Koumetio': 'Billy Koumetio',
        'Kepa': 'Kepa Arrizabalaga',
        'David Raya': 'David Raya',
        'S. Romero': 'Sergio Romero',
        'M. Vorm': 'Michel Vorm',
        'T. Heaton': 'Tom Heaton',
        'N. Pope': 'Nick Pope',
        'E. van der Sar': 'Edwin van der Sar',
        'Fábio Paim': 'Fábio Paim',
        'K. Darlow': 'Karl Darlow',
        'P. Gollini': 'Pierluigi Gollini',
        'L. Fabiański': 'Łukasz Fabiański',
        'B. Foster': 'Ben Foster',
        'A. Ramsdale': 'Aaron Ramsdale',
        'D. Henderson': 'Dean Henderson',
        'A. McCarthy': 'Alex McCarthy',
        'T. Krul': 'Tim Krul',
        'Adrián': 'Adrián San Miguel',
        'L. Karius': 'Loris Karius',
        'Gomes': 'Heurelho Gomes',
        'A. Gunn': 'Angus Gunn',
        'L. Grant': 'Lee Grant',
        'Robert Sánchez': 'Robert Sánchez',
        'José Sá': 'José Sá',
        'A. Begović': 'Asmir Begović',
        'W. Caballero': 'Willy Caballero',
        'D. Bachmann': 'Daniel Bachmann',
        'D. Randolph': 'Darren Randolph',
        'F. Forster': 'Fraser Forster',
        'I. Meslier': 'Illan Meslier',
        'D. Ward': 'Danny Ward',
        'J. Steer': 'Jed Steer',
        'P. Robinson': 'Paul Robinson',
        'W. Hennessey': 'Wayne Hennessey',
        'M. Rodák': 'Marek Rodák',
        'É. Mendy': 'Édouard Mendy',
        'M. Gillespie': 'Mark Gillespie',
        'P. Čech': 'Petr Čech',
        'J. Steele': 'Jason Steele',
        'F. Woodman': 'Freddie Woodman',
        'D. Martin': 'David Martin',
        'R. Matthews': 'Remi Matthews',
        'R. Elliot': 'Rob Elliot',
        'Álvaro': 'Álvaro Fernández',
        'S. Carson': 'Scott Carson',
        'R. Wright': 'Richard Wright',
        'E. Jakupović': 'Eldin Jakupović',
        'M. Bettinelli': 'Marcus Bettinelli',
        'W. Norris': 'Will Norris',
        'J. Ruddy': 'John Ruddy',
        'Diego Cavalieri': 'Diego Cavalieri',
        'K. Klaesson': 'Kristoffer Klaesson',
        'S. Taylor': 'Steve Taylor',
        'D. Iliev': 'Dejan Iliev',
        'A. Lonergan': 'Andy Lonergan',
        'H. Lewis': 'Harry Lewis',
        'C. Kelleher': 'Caoimhín Kelleher',
        'K. Scherpen': 'Kjell Scherpen',
        'M. McGovern': 'Michael McGovern',
        'A. Mair': 'Aidan Mair',
        'A. Okonkwo': 'Arthur Okonkwo',
        'E. Caprile': 'Elia Caprile',
        'D. van den Heuvel': 'Daan van den Heuvel',
        'K. Hein': 'Karl Hein',
        'Marcelo Pitaluga': 'Marcelo Pitaluga',
        'C. Odunze': 'Chituru Odunze',
        'L. Bergström': 'Lucas Bergström',
        'V. Sinisalo': 'Viljami Sinisalo',
        'L. Ashby-Hammond': 'Luca Ashby-Hammond',
        'Bruno Fernandes': 'Bruno Fernandes',
        'K. De Bruyne': 'Kevin De Bruyne',
        'João Cancelo': 'João Cancelo',
        'T. Alexander-Arnold': 'Trent Alexander-Arnold',
        'M. Salah': 'Mohamed Salah',
        'Thiago': 'Thiago Alcântara',
        'L. Digne': 'Lucas Digne',
        'İ. Gündoğan': 'İlkay Gündoğan',
        'H. Kane': 'Harry Kane',
        'Fred': 'Frederico Rodrigues',
        'K. Trippier': 'Kieran Trippier',
        'Cristiano Ronaldo': 'Cristiano Ronaldo',
        'E. Haaland': 'Erling Haaland',
        'L. Martínez': 'Lisandro Martínez',
        'J. Álvarez': 'Julián Álvarez',
        'Bruno Guimarães': 'Bruno Guimarães',
        'R. Bentancur': 'Rodrigo Bentancur',
        'D. Kulusevski': 'Dejan Kulusevski',
        'Coutinho': 'Philippe Coutinho',
        'D. Núñez': 'Darwin Núñez',
        'L. Díaz': 'Luis Díaz',
        'Gonçalo Guedes': 'Gonçalo Guedes',
        'Palhinha': 'João Palhinha',
        'Emerson Royal': 'Emerson Royal',
        'C. Eriksen': 'Christian Eriksen',
        'B. Kamara': 'Boubacar Kamara',
        'S. Bastien': 'Samuel Bastien',
        'O. Norwood': 'Oliver Norwood',
        'C. Doucouré': 'Cheick Doucouré',
        'J. Cullen': 'Josh Cullen',
        'A. Campbell': 'Alfie Campbell',
        'M. Gibbs-White': 'Morgan Gibbs-White',
        'Héctor Bellerín': 'Héctor Bellerín',
        'K. Mbabu': 'Kevin Mbabu',
        'F. Guilbert': 'Frédéric Guilbert',
        'O. Mangala': 'Orel Mangala',
        'Cafú': 'Cafú',
        'D. Undav': 'Deniz Undav',
        'L. O\'Brien': 'Lewis O\'Brien',
        'Diego Carlos': 'Diego Carlos',
        'K. Mitoma': 'Kaoru Mitoma',
        'M. Caicedo': 'Moisés Caicedo',
        'J. Garner': 'James Garner',
        'Rúben Vinagre': 'Rúben Vinagre',
        'G. Scamacca': 'Gianluca Scamacca',
        'A. Onana': 'André Onana',
        'N. Aguerd': 'Nayef Aguerd',
        'P. Zinckernagel': 'Philip Zinckernagel',
        'S. Twine': 'Scott Twine',
        'H. Lansbury': 'Henri Lansbury',
        'A. Hickey': 'Aaron Hickey',
        'A. Palaversa': 'Adrián Palaversa',
        'J. Bogle': 'Jayden Bogle',
        'P. Sarr': 'Pape Matar Sarr',
        'M. Solomon': 'Manor Solomon',
        'L. Freeman': 'Luke Freeman',
        'M. Batshuayi': 'Michy Batshuayi',
        'M. Niakhaté': 'Moussa Niakhaté',
        'Fábio Vieira': 'Fábio Vieira',
        'M. Longstaff': 'Matty Longstaff',
        'C. Lenglet': 'Clément Lenglet',
        'F. Andone': 'Florin Andone',
        'H. Toffolo': 'Harry Toffolo',
        'B. Rahman': 'Baba Rahman',
        'R. Yates': 'Ryan Yates',
        'I. Maatsen': 'Ian Maatsen',
        'W. Saliba': 'William Saliba',
        'T. Awoniyi': 'Taiwo Awoniyi',
        'J. Lolley': 'Joe Lolley',
        'T. Doyle': 'Tommy Doyle',
        'H. Arter': 'Harry Arter',
        'J. Clark': 'Jordan Clark',
        'M. Lowe': 'Max Lowe',
        'G. Baldock': 'George Baldock',
        'C. Ronan': 'Connor Ronan',
        'E. Ampadu': 'Ethan Ampadu',
        'K. Lewis-Potter': 'Keane Lewis-Potter',
        'K. LuaLua': 'Kazenga LuaLua',
        'C. Coventry': 'Conor Coventry',
        'K. Koulibaly': 'Kalidou Koulibaly',
        'J. Colback': 'Jack Colback',
        'G. Biancone': 'Giulian Biancone',
        'Fábio Carvalho': 'Fábio Carvalho',
        'E. Stevens': 'Enda Stevens',
        'T. Chong': 'Tahith Chong',
        'R. Nelson': 'Reiss Nelson',
        'O. Richards': 'Owen Richards',
        'J. Bree': 'James Bree',
        'E. Adebayo': 'Elijah Adebayo',
        'V. Mykolenko': 'Vitaliy Mykolenko',
        'H. Dervişoğlu': 'Halil Dervişoğlu',
        'Toti Gomes': 'Toti Gomes',
        'Léo Bonatini': 'Léo Bonatini',
        'Vitinho': 'Vitinho',
        'R. Vilca': 'Rodrigo Vilca',
        'S. Botman': 'Sven Botman',
        'K. Mainoo': 'Kobbie Mainoo',
        'A. Broja': 'Armando Broja',
        'C. Woodrow': 'Cauley Woodrow',
        'J. Egan': 'John Egan',
        'I. Kaboré': 'Issa Kaboré',
        'M. Roerslev': 'Mads Roerslev',
        'C. Jerome': 'Cameron Jerome',
        'D. Spence': 'Djed Spence',
        'L. Grabban': 'Lewis Grabban',
        'Marquinhos': 'Marquinhos',
        'A. Ahmedhodžić': 'Anel Ahmedhodžić',
        'R. Norrington-Davies': 'Rhys Norrington-Davies',
        'J. van Hecke': 'Jan Paul van Hecke',
        'G. Murray': 'Glenn Murray',
        'C. McCann': 'Chris McCann',
        'N. Patterson': 'Nathan Patterson',
        'Chiquinho': 'Chiquinho',
        'P. Cutrone': 'Patrick Cutrone',
        'L. Taylor': 'Lyle Taylor',
        'J. Cain': 'Joe Cain',
        'J. Oksanen': 'Jasper Oksanen',
        'L. Colwill': 'Levi Colwill',
        'L. Hall': 'Lewis Hall',
        'H. Cornick': 'Harry Cornick',
        'D. Sarmiento': 'Dereck Sarmiento',
        'D. Sterling': 'Dujon Sterling',
        'K. Kesler Hayden': 'Kaine Kesler Hayden',
        'N. Nkounkou': 'Niels Nkounkou',
        'M. Azeez': 'Miguel Azeez',
        'B. Norton-Cuffy': 'Brooke Norton-Cuffy',
        'S. van den Berg': 'Sepp van den Berg',
        'R. Brewster': 'Rhian Brewster',
        'M. Rogers': 'Morgan Rogers',
        'C. Morris': 'Carlos Morris',
        'D. Potts': 'Dan Potts',
        'I. Ndiaye': 'Iliman Ndiaye',
        'S. Jasper': 'Sammy Jasper',
        'J. O\'Connell': 'Jack O\'Connell',
        'C. Richards': 'Chris Richards',
        'T. O\'Reilly': 'Tom O\'Reilly',
        'T. Harwood-Bellis': 'Taylor Harwood-Bellis',
        'F. Stevanović': 'Filip Stevanović',
        'M. Clarke': 'Matt Clarke',
        'D. Bernard': 'Di\'Shon Bernard',
        'H. Vale': 'Harvey Vale',
        'Marc Jurado': 'Marc Jurado',
        'D. Jebbison': 'Daniel Jebbison',
        'Z. Iqbal': 'Zidane Iqbal',
        'C. Wellens': 'Charlie Wellens',
        'M. Moreno': 'Miguel Moreno',
        'I. Hansen-Aarøen': 'Isak Hansen-Aarøen',
        'J. Turner-Cooke': 'Joe Turner-Cooke',
        'S. Oulad M\'Hand': 'Samir Oulad M\'Hand',
        'O. Hutchinson': 'Omari Hutchinson',
        'M. Ebiowei': 'Malcolm Ebiowei',
        'L. Plange': 'Luke Plange',
        'C. Savage': 'Charlie Savage',
        'Alejandro Garnacho': 'Alejandro Garnacho',
        'B. Hardley': 'Björn Hardley',
        'O. Hesketh': 'Oliver Hesketh',
        'G. McEachran': 'George McEachran',
        'K. Appiah-Forson': 'Keenan Appiah-Forson',
        'L. de Bolle': 'Lucas de Bolle',
        'B. Young': 'Ben Young',
        'S. McAllister': 'Sam McAllister',
        'G. Osho': 'Gabriel Osho',
        'R. Burke': 'Reece Burke',
        'O. Dodgson': 'Owen Dodgson',
        'T. Adaramola': 'Tayo Adaramola',
        'M. Paskotši': 'Maksim Paskotši',
        'D. Gore': 'Dan Gore',
        'J. Enciso': 'Julio Enciso',
        'L. Watson': 'Luke Watson',
        'C. Webster': 'Charlie Webster',
        'L. Pye': 'Louis Pye',
        'J. Giddings': 'Joe Giddings',
        'S. McKenna': 'Scott McKenna',
        'J. Bennett': 'Joe Bennett',
        'R. Thomson': 'Ryan Thomson',
        'C. McNeill': 'Charlie McNeill',
        'N. Emeran': 'Noam Emeran',
        'D. Williams': 'Dylan Williams',
        'T. Collyer': 'Toby Collyer',
        'D. Costelloe': 'Dara Costelloe',
        'A. Mighten': 'Alex Mighten',
        'Z. Brunt': 'Zak Brunt',
        'M. Helm': 'Michael Helm',
        'J. Fevrier': 'Jaden Fevrier',
        'A. Tanimowo': 'Aaron Tanimowo',
        'F. Maguire': 'Festy Maguire',
        'P. Glatzel': 'Paul Glatzel',
        'A. Konaté': 'Abdoulaye Konaté',
        'J. Worrall': 'Joe Worrall',
        'O. Hammond': 'Oliver Hammond',
        'S. Mather': 'Sonny Mather',
        'O. Dacourt': 'Olivier Dacourt',
        'H. Birtwistle': 'Harry Birtwistle',
        'T. Nevers': 'Tristan Nevers',
        'L. McNally': 'Luke McNally',
        'C. Egan-Riley': 'CJ Egan-Riley',
        'G. Lewis': 'George Lewis',
        'F. Potts': 'Felipe Potts',
        'D. Stephenson': 'Dylan Stephenson',
        'E. Ennis': 'Ethan Ennis',
        'S. Bradley': 'Sean Bradley',
        'A. Odubeko': 'Ademipo Odubeko',
        'J. Hodnett': 'Joe Hodnett',
        'D. Lembikisa': 'Dion Lembikisa',
        'J. Hugill': 'Joe Hugill',
        'V. Adedokun': 'Víctor Adedokun',
        'W. Kambwala': 'Willy Kambwala',
        'K. Gordon': 'Kasey Gordon',
        'A. Gilbert': 'Alex Gilbert',
        'A. El Mhassani': 'Anis El Mhassani',
        'J. Scott': 'Joe Scott',
        'E. Ferguson': 'Evan Ferguson',
        'A. Pressley': 'Aaron Pressley',
        'N. Brookwell': 'Niall Brookwell',
        'L. Mbe Soh': 'Loïc Mbe Soh',
        'O. Forson': 'Ollie Forson',
        'L. Harkin': 'Luca Harkin',
        'A. Keto-Diyawa': 'Aaron Keto-Diyawa',
        'J. Barnes': 'Jayden Barnes',
        'J. Walls': 'Joe Walls',
        'C. Barrett': 'Cody Barrett',
        'V. Akinwale': 'Victor Akinwale',
        'M. Ifill': 'Mason Ifill',
        'W. Greenidge': 'Will Greenidge',
        'R. Bennett': 'Rio Bennett',
        'R. Trevitt': 'Ryan Trevitt',
        'J. Oliver': 'Jack Oliver',
        'I. Price': 'Isaac Price',
        'T. Addy': 'Tyler Addy',
        'E. McJannet': 'Ethan McJannet',
        'E. Turns': 'Ed Turns',
        'C. Ferguson': 'Caleb Ferguson',
        'D. Taylor': 'Dylan Taylor',
        'S. Ortega': 'Stefan Ortega',
        'W. Fish': 'Will Fish',
        'M. Woltman': 'Max Woltman',
        'Z. Larkeche': 'Ziyad Larkeche',
        'L. Dendoncker': 'Lars Dendoncker',
        'D. Revan': 'Dimitri Revan',
        'B. Cross': 'Ben Cross',
        'Mateo Mejía': 'Mateo Mejía',
        'J. Hubner': 'Jaden Hubner',
        'J. Larsson': 'Johan Larsson',
        'Gonçalo Cardoso': 'Gonçalo Cardoso',
        'N. Carlyon': 'Noah Carlyon',
        'O. Tipton': 'Owen Tipton',
        'A. Hackford': 'Antwoine Hackford',
        'A. Kirk': 'Aaron Kirk',
        'W. Osula': 'William Osula',
        'S. Beckwith': 'Sam Beckwith',
        'H. Ogunby': 'Harvey Ogunby',
        'Z. Awe': 'Zane Awe',
        'T. Crama': 'Tristan Crama',
        'M. Muir': 'Max Muir',
        'J. Joseph': 'Jaden Joseph',
        'D. Esapa Osong': 'David Esapa Osong',
        'A. Donnelly': 'Alex Donnelly',
        'L. Badley-Morgan': 'Luis Badley-Morgan',
        'K. Lopata': 'Kacper Lopata',
        'B. Peacock-Farrell': 'Bailey Peacock-Farrell',
        'J. Feeney': 'Joe Feeney',
        'J. McGlynn': 'Josh McGlynn',
        'K. Kandola': 'Kameron Kandola',
        'H. Hagan': 'Harry Hagan',
        'A. Koutismouka': 'Alex Koutismouka',
        'A. Jones': 'Alfie Jones',
        'S. Johnstone': 'Sam Johnstone',
        'R. Rama': 'Ronaldo Rama',
        'R. Olsen': 'Robin Olsen',
        'T. Strakosha': 'Thomas Strakosha',
        'M. Thomas-Sadler': 'Max Thomas-Sadler',
        'M. Turner': 'Matt Turner',
        'B. Tricker': 'Ben Tricker',
        'R. Rúnarsson': 'Rúnar Alex Rúnarsson',
        'A. Murić': 'Aro Murić',
        'R. Welch': 'Ryan Welch',
        'C. Ikeme': 'Carl Ikeme',
        'M. Macey': 'Matt Macey',
        'W. Brown': 'Will Brown',
        'N. Trott': 'Nathan Trott',
        'I. Roper': 'Isaac Roper',
        'M. Šarkić': 'Milan Šarkić',
        'P. Banda': 'Peter Banda',
        'B. Winterbottom': 'Ben Winterbottom',
        'G. Shelvey': 'George Shelvey',
        'R. Rees': 'Ryan Rees',
        'B. Austin': 'Brandon Austin',
        'M. Cox': 'Michael Cox',
        'O. Mastný': 'Ondřej Mastný',
        'D. Mee': 'David Mee',
        'T. Wooster': 'Tom Wooster',
        'T. Sharman-Lowe': 'Teddy Sharman-Lowe',
        'J. Amissah': 'Jaden Amissah',
        'J. Young': 'Joe Young',
        'S. Waller': 'Sam Waller',
        'P. Arinbjörnsson': 'Pétur Arinbjörnsson',
        'Á. Onódi': 'Ákos Onódi'
    }
    
    # Appliquer les expansions de noms
    for short_name, full_name in name_expansions.items():
        if name == short_name:
            name = full_name
            break
    
    return name.strip()

def process_season_2018_2021(df, season_name):
    """Traiter les datasets 2018-2019, 2019-2020, 2020-2021"""
    # Renommer les colonnes selon les spécifications
    df = df.rename(columns={
        'long_name': 'Player',
        'age': 'Age', 
        'club_name': 'Club',
        'preferred_foot': 'Preferred Foot',
        'nationality': 'Nationality'
    })
    
    # Pour 2020-2021, renommer value_eur en Value si nécessaire
    if 'value_eur' in df.columns and 'Value' not in df.columns:
        df = df.rename(columns={'value_eur': 'Value'})
    
    # Nettoyer les noms des joueurs
    df['Player'] = df['Player'].apply(clean_player_names)
    
    # Réorganiser les colonnes
    column_order = ['Player', 'Age', 'Nationality', 'Club', 'Value', 'Preferred Foot']
    
    # Ajouter les autres colonnes (sauf Height et Weight)
    other_cols = [col for col in df.columns if col not in column_order + ['Height', 'Weight', 'height', 'weight']]
    column_order.extend(other_cols)
    
    # Ajouter Height et Weight à la fin
    height_col = 'Height' if 'Height' in df.columns else 'height'
    weight_col = 'Weight' if 'Weight' in df.columns else 'weight'
    
    if height_col in df.columns:
        column_order.append('Height')
    if weight_col in df.columns:
        column_order.append('Weight')
    
    df = df[column_order]
    
    # Nettoyer Height et Weight
    if 'Height' in df.columns:
        df['Height'] = df['Height'].apply(clean_height)
    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].apply(clean_weight)
    
    # Convertir les valeurs
    if 'Value' in df.columns:
        df['Value'] = df['Value'].apply(convert_value)
    
    # Ajouter la colonne Season
    df['Season'] = season_name
    
    return df

def process_season_2021_2023(df, season_name):
    """Traiter les datasets 2021-2022 et 2022-2023"""
    # Renommer Name en Player pour uniformité
    df = df.rename(columns={'Name': 'Player'})
    
    # Nettoyer les noms des joueurs
    df['Player'] = df['Player'].apply(clean_player_names)
    
    # Nettoyer Height et Weight
    df['Height'] = df['Height'].apply(clean_height)
    df['Weight'] = df['Weight'].apply(clean_weight)
    
    # Convertir Value
    df['Value'] = df['Value'].apply(convert_value)
    
    # Réorganiser les colonnes pour avoir Player en première position
    cols = ['Player'] + [col for col in df.columns if col != 'Player' and col != 'Season']
    df = df[cols]
    
    # Ajouter la colonne Season
    df['Season'] = season_name
    
    return df

# [Le reste du code reste identique...]
# Charger et traiter tous les datasets
datasets = []

# Dataset 2018-2019
try:
    df_2018_2019 = pd.read_csv('premier_league_cleaned_2018-2019.csv')
    df_2018_2019 = process_season_2018_2021(df_2018_2019, '2018-2019')
    datasets.append(df_2018_2019)
    print("✅ 2018-2019 traité")
except Exception as e:
    print(f"❌ Erreur 2018-2019: {e}")

# Dataset 2019-2020  
try:
    df_2019_2020 = pd.read_csv('premier_league_cleaned_2019-2020.csv')
    df_2019_2020 = process_season_2018_2021(df_2019_2020, '2019-2020')
    datasets.append(df_2019_2020)
    print("✅ 2019-2020 traité")
except Exception as e:
    print(f"❌ Erreur 2019-2020: {e}")

# Dataset 2020-2021
try:
    df_2020_2021 = pd.read_csv('premier_league_cleaned_2020-2021.csv')
    df_2020_2021 = process_season_2018_2021(df_2020_2021, '2020-2021')
    datasets.append(df_2020_2021)
    print("✅ 2020-2021 traité")
except Exception as e:
    print(f"❌ Erreur 2020-2021: {e}")

# Dataset 2021-2022
try:
    df_2021_2022 = pd.read_csv('premier_league_cleaned_2021-2022.csv')
    df_2021_2022 = process_season_2021_2023(df_2021_2022, '2021-2022')
    datasets.append(df_2021_2022)
    print("✅ 2021-2022 traité")
except Exception as e:
    print(f"❌ Erreur 2021-2022: {e}")

# Dataset 2022-2023
try:
    df_2022_2023 = pd.read_csv('premier_league_cleaned_2022-2023.csv')
    df_2022_2023 = process_season_2021_2023(df_2022_2023, '2022-2023')
    datasets.append(df_2022_2023)
    print("✅ 2022-2023 traité")
except Exception as e:
    print(f"❌ Erreur 2022-2023: {e}")

# Fusionner tous les datasets
if datasets:
    final_df = pd.concat(datasets, ignore_index=True)
    
    # Réorganiser l'ordre final des colonnes pour avoir Season à la fin
    cols = [col for col in final_df.columns if col != 'Season']
    cols.append('Season')
    final_df = final_df[cols]
    
    # Assurer que Player est la première colonne
    if 'Player' in final_df.columns:
        player_col = final_df['Player']
        final_df = final_df.drop('Player', axis=1)
        final_df.insert(0, 'Player', player_col)
    
    # Sauvegarder le dataset fusionné
    final_df.to_csv('premier_league_merged_2018-2023.csv', index=False)
    
    print(f"\n🎉 Fusion terminée !")
    print(f"📊 Dataset final: {final_df.shape[0]} lignes, {final_df.shape[1]} colonnes")
    print(f"💾 Sauvegardé sous: premier_league_merged_2018-2023.csv")
    
    # Aperçu des données
    print(f"\n📋 Aperçu des colonnes finales:")
    print(final_df.columns.tolist())
    print(f"\n👀 Aperçu des données:")
    print(final_df.head())
    
    # Vérifier quelques corrections de noms
    print(f"\n🔍 Exemples de noms corrigés:")
    sample_names = ['H. Son', '武藤 嘉紀', '18 S.Taylor', 'J. Henderson']
    for name in sample_names:
        corrected = clean_player_names(name)
        print(f"  {name} → {corrected}")
    
else:
    print("❌ Aucun dataset n'a pu être chargé")
