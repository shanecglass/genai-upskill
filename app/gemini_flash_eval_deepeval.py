# from deepeval import assert_test
# from deepeval.metrics import AnswerRelevancyMetric
# from deepeval.test_case import LLMTestCase

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.metrics.ragas import RagasMetric


from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_vertexai import (
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory
)
from model_mgmt import config, prompt

import os
import json
# import testing as testing
import vertexai

project_id = config.project_id
location = config.location


# flake8: noqa --E501


class GoogleVertexAI(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Vertex AI Model"


model_id = "gemini-1.5-flash-002"
# Initilialize safety filters for vertex model
# This is important to ensure no evaluation responses are blocked
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

vertexai.init(project=project_id, location=location)

# TODO : Add values for project and location below
gemini_model = ChatVertexAI(
    model_name=model_id,
    safety_settings=safety_settings,
    project_id=project_id,
    location=location
)

# initiatialize the  wrapper class
vertexai_gemini = GoogleVertexAI(model=gemini_model)
print(vertexai_gemini.generate("Write me a joke"))

input = prompt.generate_template(
    "I want to go to Florence Italy. I have to see the Ponte Vecchio and the Duomo. I want to learn more about the art history of Florence on this trip and see works by Raphael"
)


# Replace this with the actual output from your LLM application
actual_output = """
Ecco il tuo itinerario di 7 giorni a Firenze!

Hotel: The Florence Inn (Via de' Tornabuoni, 20, 50123 Firenze, Italy). A charming and affordable hotel in the heart of the city.

Day 1: Arrival and Ponte Vecchio Charm

Arrive in Florence, check into The Florence Inn.
Stroll across the Ponte Vecchio, admiring the shops and the Arno River. Enjoy the sunset from the bridge – it's breathtaking! (See the image I found: [link to image from image_search_attractions]).
Day 2: Duomo and Baptistery

Visit the magnificent Duomo, climbing to the top of Brunelleschi's Dome for panoramic views of Florence.
Explore the Baptistery with its famous bronze doors.
Day 3: Art History Immersion

Spend the day at the Uffizi Gallery, focusing on the works of Raphael and other Renaissance masters. (Based on the information from doc_search_attractions, I know you'll find many Raphael pieces here!)
Day 4: Beyond the Tourist Trail

Explore the Oltrarno neighborhood, known for its artisan workshops and charming streets.
Consider a cooking class to learn the secrets of Tuscan cuisine.
Day 5: Accademia Gallery and Michelangelo

Visit the Accademia Gallery to see Michelangelo's David and collections of paintings by Florentine artists mostly from the period 1300–1600.
Day 6: Hidden Gems and Piazzas

Discover hidden gems like the Palazzo Pitti and the Boboli Gardens.
Enjoy the lively atmosphere of Piazza della Signoria and Piazza Michelangelo (as suggested by doc_search_attractions).
Day 7: Departure

Enjoy a final Florentine breakfast before departing from Florence.
This itinerary is a suggestion, of course. Feel free to adjust it based on your preferences and energy levels. Remember to book tickets for popular attractions in advance, especially the Uffizi Gallery. And don't forget to indulge in delicious Florentine food and wine throughout your trip! Have a wonderful time in Florence!

https://storage.googleapis.com/genai-l200/images/Ponte-Vecchio-Florence-The-Tour-Guy-Tours-700.jpg

"""

# Replace this with the expected output from your RAG generator
expected_output = """
Florence, the birthplace of the Renaissance, awaits you with its artistic treasures and architectural wonders. Get ready to immerse yourself in the world of Raphael, Michelangelo, and other masters while exploring iconic landmarks like the Ponte Vecchio and the magnificent Duomo.

Day 1:

Morning: Arrive in Florence, check into your hotel, and leave your luggage. I recommend staying at the The Florence Inn, a cozy and affordable hotel located in the heart of the city with warm and welcoming atmosphere and offering views of the bustling streets or the peaceful courtyard. Their address is: Via de' Tornabuoni, 20, 50123 Firenze, Italy
Afternoon: Head to the Ponte Vecchio. Take your time crossing the bridge, admiring the shops and the views of the Arno River. Enjoy lunch at a trattoria with a view of the bridge. Check out this picture I found to get a preview of the Ponte Vecchio: https://storage.googleapis.com/genai-l200/images/Ponte-Vecchio-Florence-The-Tour-Guy-Tours-700.jpg
Evening:  Take a leisurely evening stroll along the Arno River and enjoy the sunset. Find a charming restaurant for dinner and try some authentic Florentine cuisine.

Day 2:

Morning: Visit the Duomo. Be sure to climb to the top of the dome for breathtaking panoramic views of the city. Afterward, explore the Baptistery of San Giovanni and admire Michelangelo's "David."
Afternoon: Visit the Accademia Gallery to see Michelangelo's "David" up close. Take your time to appreciate this masterpiece and other Renaissance sculptures.
Evening: Enjoy a delicious dinner in the Oltrarno neighborhood, known for its artisan workshops and charming atmosphere.

Day 3:

Morning: Immerse yourself in Renaissance art at the Uffizi Gallery. Focus on the works of Raphael, including "The Madonna of the Goldfinch" and "The Portrait of Maddalena Strozzi."
Afternoon: Take a break from museums and enjoy a relaxing walk through the Boboli Gardens, a beautiful park behind the Palazzo Pitti.
Evening:  Attend an opera performance or a classical music concert for a taste of Florentine culture.

Day 4:

Morning: Visit the Palazzo Pitti, focusing on the Gallery of Modern Art and the Museum of Costume and Fashion. Don't miss Raphael's "The Holy Family with the Lamb."
Afternoon: Explore the Oltrarno neighborhood. Wander through its artisan workshops, browse antique shops, and enjoy the laid-back atmosphere.
Evening:  Take a cooking class and learn how to prepare some of your favorite Italian dishes.

Day 5:

Day trip to Siena: Take a day trip to the charming medieval city of Siena. Visit the Piazza del Campo, the main square, and climb the Torre del Mangia for stunning views.
Evening:  Enjoy a farewell dinner in Florence, savoring your last taste of Italian cuisine.

Day 6:

Morning: Visit the Bargello Museum, which houses a collection of Renaissance sculptures, including masterpieces by Donatello and Michelangelo.
Afternoon: Explore the Medici Chapels, a complex of opulent mausoleums dedicated to the powerful Medici family.
Evening:  Take a final evening stroll through Florence, revisiting your favorite spots and soaking up the atmosphere.

Day 7:

Morning: Enjoy a leisurely breakfast and do some last-minute souvenir shopping.
Afternoon: Depart from Florence, taking with you memories of a beautiful and culturally enriching trip.
"""

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = [
    json.dumps({'points_of_interest': ['Ponte Vecchio', 'Duomo'],
                'user_interests': 'art history, Raphael', 'user_destination': 'Florence, Italy'}),
    json.dumps({'hotel_name': 'The Florence Inn', 'hotel_address': "Via de' Tornabuoni, 20, 50123 Firenze, Italy", 'hotel_description':
                "Discover the charm of The Florence Inn, a cozy and affordable hotel located in the heart of the city. Our comfortable rooms offer a warm and welcoming atmosphere, with views of the bustling streets or the peaceful courtyard. Enjoy a traditional Tuscan breakfast in our charming breakfast room, and explore the city's numerous attractions on foot. Ideal for budget-conscious travelers and solo adventurers."}),
    "https://storage.googleapis.com/genai-l200/images/Ponte-Vecchio-Florence-The-Tour-Guy-Tours-700.jpg",
    """# 6. Piazza Michelangelo:\n\nhike the steps to get a beautiful view of Florence. During the summer, it is a great place to go at sunset. Bring some food and drinks and take in the amazing view of the city. 7. Piazza della Signoria and Palazzo Vecchio: Florence\'s most famous square is Piazza della Signoria, the heart of the historic center and a free open-air sculpture exhibit. The Loggia della Signoria holds some important statues including a copy of Michelangelo\'s David. The piazza has been,# 3 Restaurants\n\nWith amazing restaurants on every corner in Florence, we put together a list of places for all types of budgets. To get the full “Italian experience”, make sure to try a pizza, some delicious pasta or some Tuscan red wine before your visit finishes! Buon Apetitto! Go Inspired Traveling Guides,3. Tour the Uffizi Gallery: The Galleria degli Uffizi holds the world\'s most important collection of Renaissance art but it\'s also Italy\'s most crowded museum so it\'s a good idea to buy tickets ahead to avoid long ticket lines. The Uffizi holds thousands of paintings from medieval to modern times and many antique illuminations, sculptures, and tapestries. Artists whose works you\'ll see include Michelangelo, Giotto, Botticelli, Leonardo da Vinci, Perugino, and Raphael. Go Inspired Traveling Guides MUST-HAVE GUIDE TO FLORENCE"""
]
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=vertexai_gemini,
    include_reason=True
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=vertexai_gemini,
    include_reason=True
)

ragas_metric = RagasMetric(
    threshold=0.5, model=vertexai_gemini)


test_case = LLMTestCase(
    input=input,
    expected_output=expected_output,
    retrieval_context=retrieval_context,
    actual_output=actual_output
)

metrics = [relevancy_metric, faithfulness_metric, ragas_metric]

for metric in metrics:
    metric.measure(test_case)
    print(metric.score, metric.reason)
