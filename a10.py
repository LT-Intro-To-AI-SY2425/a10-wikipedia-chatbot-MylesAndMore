import re, string, calendar
from wikipedia import WikipediaPage
import wikipedia
from bs4 import BeautifulSoup
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from match import match
from typing import List, Callable, Tuple, Any, Match


def get_page_html(title: str) -> str:
    """Gets html of a wikipedia page

    Args:
        title - title of the page

    Returns:
        html of the page
    """
    results = wikipedia.search(title)
    return WikipediaPage(results[0]).html()


def get_first_infobox_text(html: str) -> str:
    """Gets first infobox html from a Wikipedia page (summary box)

    Args:
        html - the full html of the page

    Returns:
        html of just the first infobox
    """
    soup = BeautifulSoup(html, "html.parser")
    results = soup.find_all(class_="infobox")

    if not results:
        raise LookupError("Page has no infobox")
    return results[0].text


def clean_text(text: str) -> str:
    """Cleans given text removing non-ASCII characters and duplicate spaces & newlines

    Args:
        text - text to clean

    Returns:
        cleaned text
    """
    only_ascii = "".join([char if char in string.printable else " " for char in text])
    no_dup_spaces = re.sub(" +", " ", only_ascii)
    no_dup_newlines = re.sub("\n+", "\n", no_dup_spaces)
    return no_dup_newlines


def get_match(
    text: str,
    pattern: str,
    error_text: str = "Page doesn't appear to have the property you're expecting",
) -> Match:
    """Finds regex matches for a pattern

    Args:
        text - text to search within
        pattern - pattern to attempt to find within text
        error_text - text to display if pattern fails to match

    Returns:
        text that matches
    """
    p = re.compile(pattern, re.DOTALL | re.IGNORECASE)
    match = p.search(text)

    if not match:
        raise AttributeError(error_text)
    return match


def get_polar_radius(planet_name: str) -> str:
    """Gets the radius of the given planet

    Args:
        planet_name - name of the planet to get radius of

    Returns:
        radius of the given planet
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(planet_name)))
    pattern = r"(?:Polar radius.*?)(?: ?[\d]+ )?(?P<radius>[\d,.]+)(?:.*?)km"
    error_text = "Page infobox has no polar radius information"
    match = get_match(infobox_text, pattern, error_text)

    return match.group("radius")


def get_birth_date(name: str) -> str:
    """Gets birth date of the given person

    Args:
        name - name of the person

    Returns:
        birth date of the given person
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"(?:Born\D*)(?P<birth>\d{4}-\d{2}-\d{2})"
    error_text = (
        "Page infobox has no birth information (at least none in xxxx-xx-xx format)"
    )
    match = get_match(infobox_text, pattern, error_text)

    return match.group("birth")

def get_address(school_name: str) -> str:
    """Gets address of the given school

    Args:
        school_name - name of the school to get address of

    Returns:
        address of the given school
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(school_name)))
    pattern = r"(?:Address\s*:?\s*)(?P<address>[\d\w\s\.,]+?)(?=\s*(?:Street|Coordinates)|$)"
    error_text = "Page infobox has no address information"
    match_obj = get_match(infobox_text, pattern, error_text)

    return match_obj.group("address")

def get_elevation(airport_name: str) -> str:
    """Gets elevation of the given airport

    Args:
        airport_name - name of the airport to get elevation of

    Returns:
        elevation of the given airport
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(airport_name)))
    pattern = r"(?:Elevation AMSL.*?)(?P<elevation>[\d,.]+)(?:.*?)ft"
    error_text = "Page infobox has no elevation information"
    match = get_match(infobox_text, pattern, error_text)

    return match.group("elevation")

def get_runway_length(airport_name: str, runway_name: str) -> str:
    """Gets length of the given runway

    Args:
        airport_name - name of the airport to get runway length of
        runway_name - name of the runway to get length of

    Returns:
        length of the given runway
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(airport_name)))
    pattern = rf"(?i){re.escape(runway_name)}\n(?P<length>[^\n]*)"
    error_text = "Page infobox has no runway length information"
    match = get_match(infobox_text, pattern, error_text)

    return match.group("length")


# below are a set of actions. Each takes a list argument and returns a list of answers
# according to the action and the argument. It is important that each function returns a
# list of the answer(s) and not just the answer itself.


def birth_date(matches: List[str]) -> List[str]:
    """Returns birth date of named person in matches

    Args:
        matches - match from pattern of person's name to find birth date of

    Returns:
        birth date of named person
    """
    return [get_birth_date(" ".join(matches))]


def polar_radius(matches: List[str]) -> List[str]:
    """Returns polar radius of planet in matches

    Args:
        matches - match from pattern of planet to find polar radius of

    Returns:
        polar radius of planet
    """
    return [get_polar_radius(matches[0])]

def address(matches: List[str]) -> List[str]:
    """Returns address of school in matches

    Args:
        matches - match from pattern of school to find address of

    Returns:
        address of school
    """
    return [get_address(" ".join(matches))]

def elevation(matches: List[str]) -> List[str]:
    """Returns elevation of airport in matches

    Args:
        matches - match from pattern of airport to find elevation of

    Returns:
        elevation of airport
    """
    return [get_elevation(" ".join(matches)) + " ft"]

def runway_length(matches: List[str]) -> List[str]:
    """Returns length of runway in matches

    Args:
        matches - match from pattern of airport and runway to find length of

    Returns:
        length of runway
    """
    return [get_runway_length(matches[1], matches[0]) + " ft"]

# dummy argument is ignored and doesn't matter
def bye_action(dummy: List[str]) -> None:
    raise KeyboardInterrupt


# type aliases to make pa_list type more readable, could also have written:
# pa_list: List[Tuple[List[str], Callable[[List[str]], List[Any]]]] = [...]
Pattern = List[str]
Action = Callable[[List[str]], List[Any]]

# The pattern-action list for the natural language query system. It must be declared
# here, after all of the function definitions
pa_list: List[Tuple[Pattern, Action]] = [
    ("when was % born".split(), birth_date),
    ("what is the polar radius of %".split(), polar_radius),
    ("what is the address of %".split(), address),
    ("what is the elevation of %".split(), elevation),
    ("what is the length of runway _ at %".split(), runway_length),
    (["bye"], bye_action),
]


def search_pa_list(src: List[str]) -> List[str]:
    """Takes source, finds matching pattern and calls corresponding action. If it finds
    a match but has no answers it returns ["No answers"]. If it finds no match it
    returns ["I don't understand"].

    Args:
        source - a phrase represented as a list of words (strings)

    Returns:
        a list of answers. Will be ["I don't understand"] if it finds no matches and
        ["No answers"] if it finds a match but no answers
    """
    for pat, act in pa_list:
        mat = match(pat, src)
        if mat is not None:
            answer = act(mat)
            return answer if answer else ["No answers"]

    return ["I don't understand"]


def query_loop() -> None:
    """The simple query loop. The try/except structure is to catch Ctrl-C or Ctrl-D
    characters and exit gracefully"""
    print("Welcome to the wikipedia chatbot!\n")
    while True:
        try:
            print()
            query = input("Your query? ").replace("?", "").lower().split()
            answers = search_pa_list(query)
            for ans in answers:
                print(ans)

        except (KeyboardInterrupt, EOFError):
            break

    print("\nSo long!\n")


# uncomment the next line once you've implemented everything are ready to try it out
query_loop()
