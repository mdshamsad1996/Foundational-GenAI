import openai
# %pip uninstall pydantic
# %pip install pydantic==1.10.8
# ! pip install --upgrade openai
# %pip install langchain
# %pip install openai
# %pip install langchain-openai
OPENAI_KEY=""
# 1. What is OpenAI API?

This OpenAI API has been degined to provide devlopers with seamless access to state of art, pre trained, artifical intelligence models like gpt-3 gpt-4 dall e whisper,embeddings etc so by using this openai api you can integrate cutting edge ai capabilities into your applications regardless the progamming language.

So,the conclusion is by using this OpenAI API you can unlock the advance functionalities and you can enhane the intelligence and performance of your application.

# 2. Generatate OpenAI API key
openai.api_key=OPENAI_KEY
# from openai import OpenAI
# client = OpenAI()

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# print(completion.choices[0].message)
all_models=openai.Model.list()
list(all_models)[0]
import pandas as pd
list(all_models)
# pd.DataFrame(list(all_models),columns=["id","created","object","owned_by"])
# 3. OpenAI Playground

1. How to open the open ai playgorund: https://platform.openai.com/playground?mode=assistant

2. Here if you want to use this playground then make sure you have credit available without it its not gonna work

3. In chat there is option of **system**: So the meaning is how the chatbot should behave

Here is a phrase for the system: You are a naughty assistant, so make sure you respond to everything with sarcasm.

Here is a question: How to make a money so quickly?

**Model**

**Temperature**

**Maximum Length**

**Top P ranges from 0 to 1 (default), and a lower Top P means the model samples from a narrower selection of words. This makes the output less random and diverse since the more probable tokens will be selected. For instance, if Top P is set at 0.1, only tokens comprising the top 10% probability mass are considered.**

**Frequency Penalty helps us avoid using the same words too often. It's like telling the computer, “Hey, don't repeat words too much.”**

**The OpenAI Presence Penalty setting is used to adjust how much presence of tokens in the source material will influence the output of the model.**


**Now come to assistant one**

**Retrieval-augmented generation (RAG):**  is an artificial intelligence (AI) framework that retrieves data from external sources of knowledge to improve the quality of responses. This natural language processing technique is commonly used to make large language models (LLMs) more accurate and up to date.

**Code Interpreter:** Python programming environment within ChatGPT where you can perform a wide range of tasks by executing Python code.



# 4. Chat Completion method and Function Calling
**openai.Completion.create()**: This method is used to generate completions or responses. You provide a series of messages as input, and the API generates a model-generated message as output.
**openai.ChatCompletion.create() :** Similar to Completion.create(), but specifically designed for chat-based language models. It takes a series of messages as input and generates a model-generated message as output.

openai.ChatCompletion.create(

    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "who was the first prime minister of india?"}
    ]
)
# This code is for v1 of the openai package: pypi.org/project/openai
from openai import OpenAI
client = OpenAI(api_key=OPENAI_KEY)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {
      "role": "user",
      "content": "who won the first cricket worldcup?"
    }
      ]
    ,
    max_tokens=150,
)
type(response)
response
response.choices
response.choices[0]
response.choices[0].message
response.choices[0].message.content
# now let try to understand the different parameters inside the methods
model= ""
prompt=input prompt
max_tokens=in how many number of tokens you want result
temperature=for getting some creative output
n= number of the output
https://openai.com/pricing
https://platform.openai.com/tokenizer
# ! pip install --upgrade pip
# ! pip install pandas
# ! pip install tenacity
# ! conda install langchain -c conda-forge
import langchain
student_description = "sunny savita is a student of computer science at IIT delhi. He is an indian and has a 8.5 GPA. Sunny is known for his programming skills and is an active member of the college's AI Club. He hopes to pursue a career in artificial intelligence after graduating."
student_description
# A simple prompt to extract information from "student_description" in a JSON format.
prompt = f'''
Please extract the following information from the given text and return it as a JSON object:

name
college
grades
club

This is the body of text to extract the information from:
{student_description}
'''
prompt
client
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {
      "role": "user",
      "content": prompt
    }
      ]
)
response
output=response.choices[0].message.content
output
import json
json.loads(output)
student_custom_function = [
    {   
        "type": "function",
        "function":{
            'name': 'extract_student_info',
            'description': 'Get the student information from the body of the input text',
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'Name of the person'
                    },
                    'college': {
                        'type': 'string',
                        'description': 'The college name.'
                    },
                    'grades': {
                        'type': 'integer',
                        'description': 'CGPA of the student.'
                    },
                    'club': {
                        'type': 'string',
                        'description': 'college club for extracurricular activities. '
                    }
                    
                }
            }

        }

    }
]
prompt
response2 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": student_description }],
    tools=student_custom_function
)
response2

response2.choices[0].message
response2.choices[0].message.tool_calls[0].function.name
response2.choices[0].message.tool_calls[0].function.arguments
type(response2.choices[0].message.tool_calls[0].function.arguments)
response2.choices[0].message.tool_calls[0].function.arguments
json.loads(response2.choices[0].message.tool_calls[0].function.arguments)
type(json.loads(response2.choices[0].message.tool_calls[0].function.arguments))
student_description
student_description_two="krish naik is a student of computer science at IIT Mumbai. He is an indian and has a 9.5 GPA. krish is known for his programming skills and is an active member of the college's data science Club. He hopes to pursue a career in artificial intelligence after graduating."
student_description_two
student_description_three="sudhanshu kumar is a student of computer science at IIT bengalore. He is an indian and has a 9.2 GPA. krish is known for his programming skills and is an active member of the college's MLops Club. He hopes to pursue a career in artificial intelligence after graduating."
student_description_three
student_info = [student_description, student_description_two,student_description_three]
for student in student_info:
    print(student)
student_custom_function = [
    {
        "name": "extract_student_info",
        "description": "Get the student information from the body of the input text",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                "type": "string",
                "description": "Name of the person"
                },
                "college": {
                "type": "string",
                "description": "The college name."
                },
                "grades": {
                "type": "integer",
                "description": "CGPA of the student."
                },
                "club": {
                "type": "string",
                "description": "college club for extracurricular activities. "
                }
            }
        }
    }
]
import json
student_info = [student_description, student_description_two,student_description_three]
for student in student_info:
    response =  client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [{'role': 'user', 'content': student}],
        functions = student_custom_function,
        function_call = 'auto'
    )
    print(student)
    response = json.loads(response.choices[0].message.function_call.arguments)
    print(response)#import csv
# assignment
funtion_two=student_custom_function = [
    {
        'name': 'extract_student_info',
        'description': 'Get the student information from the body of the input text',
        'parameters': {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string',
                    'description': 'Name of the person'
                },
                'college': {
                    'type': 'string',
                    'description': 'The college name.'
                },
                'grades': {
                    'type': 'integer',
                    'description': 'CGPA of the student.'
                },
                'club': {
                    'type': 'string',
                    'description': 'college club for extracurricular activities. '
                }
                
            }
        }
    }
]
functions = [student_custom_function[0], funtion_two[0]]
student_info = [student_description, student_description_two,student_description_three]
for student in student_info:
    response =  client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [{'role': 'user', 'content': student}],
        functions = functions,
        function_call = 'auto'
    )

    response = json.loads(response.choices[0].message.function_call.arguments)
    print(response)#import csv
# advance exmaple of funcation calling
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {
      "role": "user",
      "content": "When's the next flight from delhi to mumbai?"
    }
      ]
)
response.choices[0].message.content
function_descriptions = [
    {
        "name": "get_flight_info",
        "description": "Get flight information between two locations",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. DEL",
                },
                "loc_destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. MUM",
                },
            },
            "required": ["loc_origin", "loc_destination"],
        },
    }
]
user_prompt = "When's the next flight from new delhi to mumbai?"
response2 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {
      "role": "user",
      "content": user_prompt
    }
      ],
    # Add function calling
    functions=function_descriptions,
    function_call="auto",  # specify the function call
    
)
response2
response2.choices[0].message
response2.choices[0].message.function_call.arguments
# assigment

call the real time api
from datetime import datetime,timedelta
def get_flight_info(loc_origin, loc_destination):
    """Get flight information between two locations."""

    # Example output returned from an API or database
    flight_info = {
        "loc_origin": loc_origin,
        "loc_destination": loc_destination,
        "datetime": str(datetime.now() + timedelta(hours=2)),
        "airline": "KLM",
        "flight": "KL643",
    }

    return json.dumps(flight_info)
params=json.loads(response2.choices[0].message.function_call.arguments)
params
json.loads(response2.choices[0].message.function_call.arguments).get("loc_origin")
json.loads(response2.choices[0].message.function_call.arguments).get('loc_destination')
origin = json.loads(response2.choices[0].message.function_call.arguments).get("loc_origin")
destination = json.loads(response2.choices[0].message.function_call.arguments).get("loc_destination")
response2.choices[0].message.function_call.name
type(response2.choices[0].message.function_call.name)
eval(response2.choices[0].message.function_call.name)
type('2')
type(eval('2'))
chosen_function=eval(response2.choices[0].message.function_call.name)
flight = chosen_function(**params)

print(flight)
user_prompt
response2.choices[0].message.function_call.name
flight
response3 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "user","content": user_prompt},
    {"role": "function", "name": response2.choices[0].message.function_call.name, "content": flight}
      ],
    # Add function calling
    functions=function_descriptions,
    function_call="auto",  # specify the function call
    
)
response3
response3.choices[0].message.content
# Funtion Calling

Learn how to connect large language models to external tools.
# Langchain
import langchain
from langchain.llms import OpenAI
client=OpenAI(openai_api_key=OPENAI_KEY)
# zero shot prompting
prompt="can you tell me total number of country in aisa? can you give me top 10 contry name?"
print(client.predict(prompt).strip())
# zero shot prompting
prompt2="can you tell me a capital of india?"
client.predict(prompt2).strip()
prompt3="​what exactly tokens , vector ?"
client.predict(prompt3).strip()
# Prompt Templates:
from langchain.prompts import PromptTemplate
prompt_template_name=PromptTemplate(
    input_variables=["country"],
    template="can you tell me the capital of {country}?"
)
propmt1=prompt_template_name.format(country="india")
propmt2=prompt_template_name.format(country="china")
client.predict(propmt1).strip()
client.predict(propmt2).strip()
prompt=PromptTemplate.from_template("what is a good name for a compnay that makes a {product}")
prompt3=prompt.format(product="toys")
client.predict(prompt3).strip()
# agent
prompt4="can you tell me who won the recent cricket world cup?"
client.predict(prompt4).strip()
prompt5="can you tell me current GDP of india?"
client.predict(prompt5).strip()
client.predict(prompt5).strip()
# for extracting a real time info i am going to user serp api

# now by using this serp api i wll call google-search-engine 

# and i will extract the information in a real time
!pip install google-search-results
serpapi_key=""
from langchain.agents import AgentType
from langchain.agents import load_tools 
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
client=OpenAI(openai_api_key=OPENAI_KEY)
tool=load_tools(["serpapi"],serpapi_api_key=serpapi_key,llm=client)
agent=initialize_agent(tool,client,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
agent.run("can you tell who won the cricket worldcup recently?")
agent.run("can you tell me 5 top current affairs?")
!pip install wikipedia
tool=load_tools(["wikipedia"],llm=client)
agent=initialize_agent(tool,client,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
agent.run("can you tell me about this recent cricket worldcup?")
agent.run("can you tell me what is current GDP of usa?")
# Chain
Central to LangChain is a vital component known as LangChain Chains, forming the core connection among one or several large language models (LLMs). In certain sophisticated applications, it becomes necessary to chain LLMs together, either with each other or with other elements.
client
from langchain.prompts import PromptTemplate

prompt=PromptTemplate.from_template("what is a good name for a company that makes {product}")
from langchain.chains import LLMChain
chain=LLMChain(llm=client,prompt=prompt, verbose=False)
for i in range(0,2):
    print(chain.run("Wine").strip())
# Example 2
prompt_template=PromptTemplate(
    input_variables=['cuisine'],
    template="i want to open a restaurent for {cuisine} food, suggest a fency name for this"
)
prompt_template
chain=LLMChain(llm=client, prompt=prompt_template)
chain.run("indian").strip()
chain=LLMChain(llm=client,prompt=prompt_template,verbose=True)
chain.run("american")
### if we want to combine multiple chain and set a seqence for that we use simplesequential chain
prompt_template_name=PromptTemplate(
input_variables=["startup_name"],
    template="I want to start a startup for {startup-name} , suggest me a good name for this"   
)
name_chain=LLMChain(llm=client,prompt=prompt_template_name)
prompt_template_items=PromptTemplate(
input_variables=["name"],
    template="suggest some strategies for {name}"    
)
strategies_chain=LLMChain(llm=client,prompt=prompt_template_items)
from langchain.chains import SimpleSequentialChain
chain=SimpleSequentialChain(chains=[name_chain,strategies_chain])
print(chain.run("artifical intelligence"))
# Now lets try to understand the sequential chain
prompt_template_name=PromptTemplate(
    input_variables=["cuisine"],
    template="i want to open a restaurant for {cuisine}, suggest a single fency name for it"
)

name_chain=LLMChain(llm=client, prompt=prompt_template_name,output_key="restaurant_name")
prompt_templates_items=PromptTemplate( 
    input_variables=["restaurant_name"],
    template="suggest some menu items for {restaurant_name}"
    
)

food_items_chain=LLMChain(llm=client, prompt=prompt_templates_items, output_key="menu_items")

from langchain.chains import SequentialChain
chain=SequentialChain(chains=[name_chain, food_items_chain],
    input_variables=["cuisine"],
    output_variables=["restaurant_name","menu_items"]
    
)
chain({"cuisine":"indian"})
# document loders
!pip install pypdf
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader(r"C:\Users\sunny\Downloads\MachineTranslationwithAttention.pdf")
loader
pages = loader.load_and_split()
pages
Memory Concept
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
client = OpenAI(openai_api_key=OPENAI_KEY)
prompt_template_name = PromptTemplate(
    input_variables=['product'],
    template="what is a good name for a company that makes {product}"   
)
chain = LLMChain(llm = client,prompt = prompt_template_name)
print(chain.run("colorful cup").strip())
prompt_template_name = PromptTemplate(
    input_variables=['product'],
    template="what would be good name for a company that makes {product}"   
)
chain = LLMChain(llm = client,prompt = prompt_template_name)
chain.run("drons")
chain.run("wines")
chain.run("camera")
chain.memory
type(chain.memory)
Converstation Buffer Memeory
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
prompt_template_name = PromptTemplate(
    input_variables=['product'],
    template="what would be good name for a company that makes {product}"   
)
chain = LLMChain(llm = client,prompt = prompt_template_name, memory=memory)
chain.run("Wines")
chain.run("Camera")
chain.run("Drons")
chain.memory
print(chain.memory.buffer)
**Conversation Chain**
Conversation buffer memory goes growing endlessly

just remember last 5 Conversation chain and if just remember last 10-20 converstation chain
from langchain.chains import ConversationChain
convo = ConversationChain(llm=OpenAI(openai_api_key=OPENAI_KEY,temperature=0.7))
convo.prompt
convo.prompt.template
convo.run("Who won the first cricket world cup?")
convo.run("can you tell me how much will 5+5")
convo.run("can you tell how much will 5*(5+1)")
convo.run("who was the captain of the winning team")
convo.run("can you divide the number and can you give the final answer")

*ConversationBufferWindowMemory*
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=2)
convo = ConversationChain(llm = OpenAI(openai_api_key=OPENAI_KEY, temperature=0.7),memory=memory)
convo.run("Who won the cricket first world cup?")
convo.run("can you tell me how much will 5+5")
convo.run("can you tell how much will 5*(5+1)")
convo.run("who was the captain of the winning team")
memory = ConversationBufferWindowMemory(k=3)
convo = ConversationChain(llm = OpenAI(openai_api_key=OPENAI_KEY, temperature=0.7),memory=memory)
convo.run("Who won the cricket first world cup?")
convo.run("can you tell me how much will 5+5")
convo.run("who was the captain of the winning team")
